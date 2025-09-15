import torch
import argparse
import yaml
import os
import polars as pl
from enum import StrEnum, auto
from torch import nn
from torch.utils.data import DataLoader
# from torch.amp import GradScaler, autocast
from tqdm import tqdm
from typing import Dict, List, Any, Optional
from .evaluate import evaluate 
from .model import FeatureSet, MethylCNNv2, MODEL_REGISTRY 
from .dataset import MethylIterableDataset
from .evaluate import evaluate

def get_args():
    """Parses command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train a methylation classifier.")
    parser.add_argument('config_path', type=str)
    parser.add_argument('--output-model-path', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=8)
    return parser.parse_args()

def parse_config(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config 

def parse_stats(stats_path):
        with open(stats_path, 'r') as f:
            stats = yaml.safe_load(f)
        return stats 

class Split(StrEnum):
    TRAIN = auto() # Value will be 'train'
    TEST = auto()  # Value will be 'test'

def make_dataloader(split, config, stats, args):
    dataset_params = config['data']
    training_params = config['training']
    data_path = config['data'][f'{split}_path']
    
    dataset = MethylIterableDataset(
        data_path,
        means= stats['means'],
        stds=stats['stds'],
        context=dataset_params['context'],
        restrict_row_groups=dataset_params['restrict_row_groups'],
        single_strand=dataset_params['single_strand']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=training_params['batch_size'],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=32 if args.num_workers > 0 else None
    )
    
    return dataloader


def make_criterion(criterion_config):
    """Instantiates a criterion from a config dictionary."""
    name = criterion_config['name']
    params = criterion_config.get('params', {})
    CriterionClass = getattr(nn, name)
    return CriterionClass(**params)

def make_optimizer(optimizer_config, model):
    """Instantiates an optimizer from a config dictionary."""
    name = optimizer_config['name']
    params = optimizer_config.get('params', {})
    OptimizerClass = getattr(torch.optim, name)
    return OptimizerClass(model.parameters(), **params)

def train(config, device, model, optimizer, criterion, train_dl, test_dl):
    # tracking statistics
    epoch_train_losses = []
    epoch_test_losses = []
    epoch_test_acc = []

    for _ in range(config['epochs']):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for _, batch in enumerate(tqdm(train_dl), 0):
            # remove the label from batch
            labels = batch.pop('label').to(device)
            # dictionary of features, with features on device
            # this doesn't work with the additional metadata info
            # inputs = {k: v.to(device) for k, v in batch.items()}
            inputs = {
                'seq': batch['seq'].to(device),
                'kinetics': batch['kinetics'].to(device)
            }
            # zero grads
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            # store training loss
            running_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
        # calculate avg training epoch loss
        avg_epoch_loss = running_loss/total_samples
        # add to running list
        epoch_train_losses.append(avg_epoch_loss)
        # get test set evaluation stats
        eval_dict = evaluate(model, test_dl, criterion, device)
        test_loss = eval_dict['loss']
        test_acc = eval_dict['accuracy']
        epoch_test_losses.append(test_loss)
        epoch_test_acc.append(test_acc)
        # print stats after each epoch
        print(f' avg epoch train loss: {round(avg_epoch_loss, 4)}\n \
        test set loss: {round(test_loss,4)}\n test set accuracy: {round(test_acc,4)}')

    print(f'Completed training for {config['training']['epochs']} epochs')
    return {'train_losses': epoch_train_losses, 'test_losses': epoch_test_losses, 'test_acc': epoch_test_acc}

def get_featureset_class(config):
    try:
        # Pop the string value from the params dict
        feature_set_str = config['model'].pop('feature_set')
        # Convert the string to a FeatureSet Enum member
        feature_set_enum = FeatureSet(feature_set_str)
        return feature_set_enum
    except KeyError:
        raise ValueError("Config missing required 'feature_set' under model.params")
    except ValueError:
        raise ValueError(f"'{feature_set_str}' is not a valid FeatureSet. Check config.yaml.")

def main():
    # get set up information
    args = get_args()
    config = parse_config(args.config_path)
    feature_set = get_featureset_class(config)
    stats = parse_stats(args.stats_path)['log_norm']
    # convert string featureset to enum
    
    # choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ### INSTANTIATION ###
    # dataloaders
    train_dl = make_dataloader(Split.TRAIN, config, stats, args)
    test_dl = make_dataloader(Split.TEST, config, stats, args)
    # model
    ModelClass = MODEL_REGISTRY[config['model']['architecture']]
    model_params = config['model'].get('params', {})
    model = ModelClass(feature_set = feature_set, **model_params)
    model.to(device)
    # criterion
    criterion = make_criterion(config['training']['criterion'])
    # optimizer 
    optimizer = make_optimizer(config['training']['optimizer'], model)

    train_stats = train(config, 
                           device,
                           model, 
                           optimizer, 
                           criterion, 
                           train_dl, 
                           test_dl
                           )
    stats_df = pl.DataFrame({
        'epoch': range(1, args.epochs + 1),
        'train_loss': train_stats['train_losses'],
        'test_loss': train_stats['test_losses'],
        'test_accuracy': train_stats['test_acc']
    })

    # Derive the log path from the model path
    log_path = args.output_model_path.replace('.pt', '_log.csv')
    print(f"Saving training log to {log_path}")
    stats_df.write_csv(log_path)
    # ensure output path existence
    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
    saved_config = config.copy()

    torch.save({
        'config': saved_config,
        'model_state_dict': model.state_dict(),
    }, args.output_model_path)
    print("Model saved successfully.")

if __name__ == '__main__':
    main()