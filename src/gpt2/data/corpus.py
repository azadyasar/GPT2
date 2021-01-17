import torch
from gpt2.data import Dataset, Vocab
from typing import Dict, Any, List, Optional


class TokenizedCorpus(Dataset):
    def __init__(self,
                 corpus_path: str,
                 vocab: Vocab,
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
            indices = self._read_n_tokens(self.seq_len - 2)
            if len(indices) + 2 > self.seq_len:
                continue

            # Decorate the sequence with additional tokens.
            indices = [self.vocab.bos_idx] + indices + [self.vocab.eos_idx]
            indices += [self.vocab.pad_idx] * (self.seq_len - len(indices) + 1)

            return {'input': indices[:-1], 'output': indices[1:]}
        
    def _read_n_tokens(self, n: int) -> List[int]:
        count = 0
        text = ""
        while True:
            char = self.corpus_fp.read(1)
            if not char or len(char) == 0:
                # Raise error when all sequences are read.
                if not self.repeat:
                    raise StopIteration()
                
                # Or, move to the beginning of the corpus.
                self.corpus_fp.seek(0)
                text = ""
                count = 0
                continue
            if char.isspace():
                count += 1
                if count >= n:
                    return [int(idx) for idx in text.split()]
            text += char
        

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
