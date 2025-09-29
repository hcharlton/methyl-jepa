# Inference Script
# INPUTS:
# - filepath to model artifact (.pt)
# - filepath to dataset file (.parquet)
# OUTPUTS: 
# - tabular inferences  (.parquet)

# OUTPUTS
# parquet file of inferences which includes columns of:
# - sample features
# - loss
# - p(methylation)

import torch
import polars as pl
import altair as alt
import os
import pyarrow.parquet as pq
import sys
import json
import numpy as np
import torch.nn.functional as F
import argparse

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Dict, List, Any, Optional
from torch import nn
from tqdm import tqdm
from enum import Enum

from src.dataset import MethylIterableDataset
from src.model import MethylCNNv2, FeatureSet, MODEL_REGISTRY

# --- Configuration ---
# for large datasets (vastly improves max size)
alt.data_transformers.enable("vegafusion")
# printing long strings config
pl.Config(fmt_str_lengths=50)
# for the workers in the iterable dataset
torch.multiprocessing.set_start_method('spawn', force=True)



def parse_artifact(artifact_path, device):
    """Loads a model with its corresponding architecture and configuration."""
    checkpoint = torch.load(artifact_path, map_location=device)

    config = checkpoint['config']
    model_state_dict = checkpoint['model_state_dict']

    return config, model_state_dict
    
def load_model(model_state_dict, config, device):
    # 1. Get architecture name and arguments from the file
    ModelClass = MODEL_REGISTRY[config['model']['architecture']]
    model_params = config['model'].get('params', {})
    feature_set_enum = model_params.pop('feature_set')
    model = ModelClass(FeatureSet(feature_set_enum), **model_params)
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
    parser.add_argument('--batch-size', 
                        type=int, 
                        default=8192, 
                        help='Batch size for inference.')
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
    losses_all: List[torch.Tensor] = []
    probs_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []
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

def main():
    args = get_args()
    config, model_state_dict = parse_artifact(args.artifact_path)
    # --- Set up device ---
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # --- Load Train-DS Normalization Stats
    with open(args.stats_path, 'r') as f:
        stats = json.load(f)
    train_means = stats['means']
    train_stds = stats['stds']

    # --- Setup Model ---
    model = load_model(model_state_dict, config, device)
    model.to(device)

    # --- Set up DataLoader --- 
    ds = MethylIterableDataset(args.data_path,
                                    means=train_means,
                                    stds=train_stds,
                                    context=32,
                                    restrict_row_groups=args.restrict_row_groups,
                                    single_strand=args.single_strand)
    dl = DataLoader(ds,
                            batch_size=args.batch_size,
                            drop_last=True,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=32)

    # --- Run Inference ---

    # --- Save Results ---





