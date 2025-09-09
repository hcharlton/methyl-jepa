import torch
import polars as pl
import altair as alt
import os
import pyarrow.parquet as pq
import sys
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Dict, List, Any, Optional
from torch import nn
from tqdm import tqdm
from enum import Enum

from src.dataset import MethylIterableDataset
from src.model import MethylCNN

# for large datasets (vastly improves max size)
alt.data_transformers.enable("vegafusion")
# printing long strings config
pl.Config(fmt_str_lengths=50)
# for the workers in the iterable dataset
torch.multiprocessing.set_start_method('spawn', force=True)

KINETICS_FEATURES = ['fi', 'fp', 'ri', 'rp']

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f'{device} operating on {torch.cuda.get_device_name(0)}')


data_path = /home/chcharlton/mutationalscanning/Workspaces/chcharlton/methyl-jepa/data/processed/martin_1m.parquet



it_workers=8
batch_size=2**13
restrict_row_groups=0
single_strand=True
#train
ds = MethylIterableDataset(data_path,
                                    means=train_means,
                                    stds=train_stds,
                                    context=32,
                                    restrict_row_groups=restrict_row_groups,
                                    single_strand=single_strand)
dl = DataLoader(ds,
                         batch_size=batch_size,
                         drop_last=True,
                         num_workers=it_workers,
                         pin_memory=True,
                         persistent_workers=True,
                         prefetch_factor=32)


