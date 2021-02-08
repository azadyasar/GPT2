import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from gpt2.utils import fusing
from gpt2.modeling import Transformer
from gpt2.data import Dataset, VocabSP, VocabYTTM, TokenizedCorpus
from gpt2.training import TrainConfig, TrainingSpec, Trainer
from typing import Tuple, Iterator, Dict

class GPT2TrainingSpec(TrainingSpec):
  def __init__(self, train_corpus: str, eval_corpus: str, vocab_path: str,
                seq_len: int, layers: int, heads: int, dims: int, rate: int,
                dropout: float, base_lr: float, wd_rate: float,
                total_steps: int, use_grad_ckpt: bool, is_sentencepiece: bool = True):
    self.train_corpus = train_corpus
    self.eval_corpus = eval_corpus
    self.vocab_path = vocab_path
    self.seq_len = seq_len
    self.layers = layers
    self.heads = heads
    self.dims = dims
    self.rate = rate
    self.dropout = dropout
    self.base_lr = base_lr
    self.wd_rate = wd_rate
    self.total_steps = total_steps
    self.use_grad_ckpt = use_grad_ckpt
    self.is_sentencepiece = is_sentencepiece
  
  def initialize(self):
    if self.is_sentencepiece:
      self.vocab = VocabSP(tokenizer_path=self.vocab_path)
    else:
      self.vocab = VocabYTTM(tokenizer_path=self.vocab_path)
    
    self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_idx,
                                         reduction='mean')
  
  def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
    train_dataset = TokenizedCorpus(corpus_path=self.train_corpus,
                                    vocab=self.vocab,
                                    seq_len=self.seq_len)
    eval_dataset = TokenizedCorpus(corpus_path=self.eval_corpus,
                                   vocab=self.vocab,
                                   seq_len=self.seq_len)
    return train_dataset, eval_dataset

  def construct_model(self) -> nn.Module:
    return Transformer(layers=self.layers, pad_idx=self.vocab.pad_idx,
                       words=len(self.vocab), seq_len=self.seq_len,
                       heads=self.heads, dims=self.dims, rate=self.rate,
                       dropout=self.dropout, bidirectional=False)
  
  def create_optimizer(self, params: Iterator[nn.Parameter]
                       ) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    optimizer = fusing.Adam(
      params, lr=self.base_lr, weight_decay=self.wd_rate)
    scheduler = optim.lr_scheduler.LambdaLR(
      optimizer, lambda step: 1 - step / self.total_steps)
    return optimizer, scheduler

  def train_objective(self, data: Dict[str, torch.Tensor], model: nn.Module)