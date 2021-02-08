import torch
import threading
import time
from gpt2.data import Dataset, VocabYTTM, VocabSP
from typing import Dict, Any, List, Optional, Union

class MTCorpus(Dataset):
    def __init__(self,
                corpus_path: str,
                vocab: Union[VocabSP, VocabYTTM],
                seq_len: int,
                repeat: bool = True):
        self.corpus_fp = open(corpus_path, 'r', encoding='utf-8')
        self.vocab = vocab
        self.seq_len = seq_len
        self.repeat = repeat

    def skip(self, count: int):
        for _ in range(count):
            if not self.corpus_fp.readline():
                # Raise error when all sequences are fetched.
                if not self.repeat:
                    raise StopIteration()

                # Or, move to the first of the corpus.
                self.corpus_fp.seek(0)
                self.corpus_fp.readline()
    
    def _fetch_one(self) -> Dict[str, List[int]]:
        while True:
            # Read subword-tokenized sequence from corpus.
            line = self.corpus_fp.readline()
            if not line or len(line) == 0:
                print(f"Consumed all of the corpus: {self.corpus_fp.name}")
                if not self.repeat:
                    raise StopIteration()
                
                print("Rewinding")
                self.corpus_fp.seek(0)
                return self._fetch_one()
            
            sentences = line.lower().split('\t')
            src_sent, trg_sent = sentences[0], sentences[1]
            src_sent_indices = self.vocab.encode(src_sent)
            src_sent_indices = [self.vocab.bos_idx] + src_sent_indices + [self.vocab.sep_idx]
            trg_sent_indices = self.vocab.encode(trg_sent)
            
            indices = src_sent_indices + trg_sent_indices
            if len(indices) + 1 > self.seq_len:
                indices = indices[:self.seq_len - 1]

            # Decorate the sequence with additional tokens.
            indices = indices + [self.vocab.eos_idx]
            indices += [self.vocab.pad_idx] * (self.seq_len - len(indices) + 1)

            return {'input': indices[:-1], 'output': indices[1:]}
    
    def fetch(self, batch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if batch is None:
            data = self._fetch_one()
        else:
            data = [self._fetch_one() for _ in range(batch)]
            data = {k: [d[k] for d in data] for k in data[0]}

        return {k: torch.tensor(v, dtype=torch.long) for k, v in data.items()}

    def where(self) -> Dict[str, Any]:
        return {'offset': self.corpus_fp.tell()}

    def assign(self, where: Dict[str, Any]):
        self.corpus_fp.seek(where['offset'])
  