# Inference Script
# INPUTS:
# - filepath to model artifact (.pt)
# - filepath to dataset file (.parquet)
# OUTPUTS: 
# - tabular inferences  (.parquet)

# OUTPUTS
# parquet file of inferences which includes columns of:
# - data features
# - model output: p(methylation)

import torch
import polars as pl
import os
import pyarrow.parquet as pq
import sys
import json
import numpy as np
import torch.nn.functional as F
import argparse
import yaml
import copy

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Dict, List, Any, Optional
from torch import nn
from tqdm import tqdm
from enum import Enum

from methyl_jepa.dataset import MethylIterableDataset
from methyl_jepa.model import MethylCNNv2, FeatureSet, MODEL_REGISTRY

# --- Configuration ---
# printing long strings config
pl.Config(fmt_str_lengths=50)
# for the workers in the iterable dataset
torch.multiprocessing.set_start_method('spawn', force=True)



def parse_artifact(artifact_path):
    """Loads a model with its corresponding architecture and configuration."""
    checkpoint = torch.load(artifact_path)

    config = checkpoint['config']
    model_state_dict = checkpoint['model_state_dict']

    return config, model_state_dict
    
def parse_stats(stats_path):
        with open(stats_path, 'r') as f:
            stats = yaml.safe_load(f)
        return stats 

def load_model(model_state_dict, config):
    config = copy.deepcopy(config)
    ModelClass = MODEL_REGISTRY[config['model']['architecture']]
    model_params = config['model'].get('params', {}).copy()
    feature_set_str = model_params.pop('feature_set')
    feature_set_enum = FeatureSet(feature_set_str)

    model = ModelClass(feature_set=feature_set_enum, **model_params)
    model.load_state_dict(model_state_dict)
    return model

# --- Get Inputs ----
def get_args():
    parser = argparse.ArgumentParser(description="Run inference on a methylation dataset.")
    
    # Required arguments
    parser.add_argument('--artifact-path', type=str, required=True, help='Path to the trained model weights (.pt file).')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the input Parquet dataset.')
    parser.add_argument('--stats-path', type=str, required=True, help='Path to the JSON file with normalization stats.')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the output Parquet file.')

    # Optional arguments
    parser.add_argument('--num-workers',
                         type=int,
                         default=8,
                         help='Number of workers for the DataLoader.')
    parser.add_argument('--restrict-row-groups',
                        type=int,
                        default=0,
                        help='For debugging, restrict to a certain number of row groups. 0 means no restriction.')
    return parser.parse_args()


def infer(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    ) -> torch.Tensor:
    model.eval()
    probs_all: List[torch.Tensor] = []
    strand_all = []
    pos_all = []
    read_name_all = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs = {
                'seq': batch['seq'].to(device),
                'kinetics': batch['kinetics'].to(device)
            }

            logits: torch.Tensor = model(inputs)
            probs = torch.sigmoid(logits)

            probs_all.append(probs.cpu())

            strand_all.extend(batch['metadata']['strand'])
            pos_all.extend(batch['metadata']['position'].tolist())
            read_name_all.extend(batch['metadata']['read_name'])

    return pl.DataFrame({
        'read_name': read_name_all,
        'strand':strand_all,
        'pos': pos_all,
        'prob': torch.cat(probs_all).numpy(),
    })

def make_dataloader(config, stats, data_path, args, device):
    dataset_params = config['data']
    training_params = config['training']
    feature_set = FeatureSet(config['model']['params']['feature_set'])
    single_strand = feature_set == FeatureSet.HEMI
    pin_memory = device == 'gpu'
    dataset = MethylIterableDataset(
        data_path,
        means= stats['means'],
        stds=stats['stds'],
        context=config['model']['params']['sequence_length'],
        restrict_row_groups=dataset_params['restrict_row_groups'],
        single_strand=single_strand,
        inference=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=training_params['batch_size'],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=32 if args.num_workers > 0 else None
    )
    
    return dataloader

def main():
    args = get_args()
    config, model_state_dict = parse_artifact(args.artifact_path)
    stats = parse_stats(args.stats_path)['log_norm']
    print('loaded paramters for inference')
    # --- Set up device ---
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # --- Setup Model ---
    model = load_model(model_state_dict, config)
    model.to(device)
    print ("instantiated model")
    # --- Set up DataLoader --- 
    dl = make_dataloader(config, stats, args.data_path, args, device)
    print("instantiated dataloader")
    # --- Run Inference ---
    df = infer(model, dl, device)
    print('ran inference')
    # --- Save Results ---
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.write_parquet(args.output_path)
    print("wrote out inference file")

if __name__ == '__main__':
    main()


