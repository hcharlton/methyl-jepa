# dataset_class.py

import torch
import polars as pl
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from torch.utils.data import IterableDataset
from typing import Dict
import torch.nn.functional as F



def compute_log_normalization_stats(df, features, epsilon=1):
    means = {col: (df[col].explode() + epsilon).log().mean() for col in features}
    stds = {col: (df[col].explode() + epsilon).log().explode().std() for col in features}
    return means, stds

class MethylIterableDataset(IterableDataset):
  '''
  Iterable dataset for parquet format methylation samples.
  Processes the parquet file in row_group's for memory efficiency.
  '''

  def __init__(
      self,
      data_path: Path,
      means: Dict,
      stds: Dict,
      context: int,
      # batch_size
      ):
    super().__init__()
    self.data_path = data_path
    self.means = means
    self.stds = stds
    self.context = context
    self.kinetics_features = ['fi', 'fp', 'ri', 'rp']
    self.vocab = {'A':0, 'T':1, 'C':2, 'G':3}
    self.vocab_size = len(self.vocab)
    # self.batch_size = batch_size
    try:
      pq_metadata = pq.read_metadata(self.data_path)
      self.num_row_groups = pq_metadata.num_row_groups
      self.len = pq_metadata.num_rows
    except:
      print('Failed to read given parquet file.')
      self.num_row_groups = 0
      self.len = 0

  def __len__(self):
    return self.len

  def _process_row_group(self, row_group_df):
    # nucleotide sequence -> list(chars) -> numpy array
    seq_int_array = np.stack(
        row_group_df['seq']
        .str.split("")
        .list.eval(
             pl.element().replace_strict(self.vocab)
        )
        .to_numpy()
        )
    # convert to torch tensor
    seq_tensor = torch.tensor((seq_int_array), dtype=torch.long) # (count(rows), len(row)) 
    # convert to one-hot, permute to shape for cat with kinetics
    seq_one_hot = F.one_hot(seq_tensor, num_classes=self.vocab_size).permute(0, 2, 1) # (count(rows), len(row), len(vocab)).permute(0,2,1)
    # kinetics
    kinetics_array = np.stack(
      [(np.log(row_group_df[col].to_numpy()+1)-self.means[col])/self.stds[col] for col in self.kinetics_features], 
      axis=1
      )
    kinetics_tensor = torch.tensor(kinetics_array, dtype=torch.float32)
    # labels
    label_tensor = torch.tensor(row_group_df['label'].to_numpy(), dtype=torch.long)

    for i in range(len(row_group_df)):
            yield {
                'seq': seq_one_hot[i],
                'kinetics': kinetics_tensor[i],
                'label': label_tensor[i]
            }
        
  def __iter__(self):
    pq_file = pq.ParquetFile(self.data_path)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info == None:
      iter_start = 0
      iter_end = self.num_row_groups
    else:
      per_worker = int(np.ceil(self.num_row_groups / float(worker_info.num_workers)))
      worker_id = worker_info.id
      iter_start = worker_id * per_worker
      iter_end = min(iter_start + per_worker, self.num_row_groups)
    row_group_indices = range(iter_start, iter_end)
    
    for i in row_group_indices:
      row_group = pq_file.read_row_group(i, use_threads=False)
      row_group_df = pl.from_arrow(row_group).with_columns([
          pl.col(c).list.to_array(self.context) for c in self.kinetics_features
          ])
      yield from self._process_row_group(row_group_df)

