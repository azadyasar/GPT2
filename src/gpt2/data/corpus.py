import torch
import threading
import time
from gpt2.data import Dataset, VocabYTTM, VocabSP
from typing import Dict, Any, List, Optional, Union
import multiprocessing as mp
from multiprocessing import Value, Array

class TokenizedCorpus(Dataset):
    def __init__(self,
                 corpus_path: str,
                 vocab: Union[VocabSP, VocabYTTM],
                 seq_len: int,
                 repeat: bool = True):
        self.corpus_fp = open(corpus_path, 'r', encoding='utf-8')
        self.vocab = vocab
        self.seq_len = seq_len
        self.repeat = repeat
        self.buffer = []
        self.buffer_pointer = 0
        
        self.tmp_buffer = Array('i', [])
        self.refill = Value('b', True)
        p = mp.Process(target=self._fill_buffer_mp)
        p.start()
        
        # self.refill_lock = threading.Lock()
        # self.read_event = threading.Event()
        # self.t = threading.Thread(target=self._fill_buffer_in_bg)
        # self.t.setDaemon(True)
        # self.t.start()
        # self.read_event.set()

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
        if (self.buffer_pointer + n) >= len(self.buffer):
            print("Asking for data")
            self.refill.acquire()
            while self.refill.value is True: time.sleep(0.00001)
            self.refill.value = True
            self.refill.release()
            
            self.tmp_buffer.acquire()
            self.buffer_pointer = 0
            self.buffer = [self.tmp_buffer[i] for i in range(len(self.tmp_buffer))]
            print("buffer len = ", len(self.buffer))
            p = mp.Process(target=self._fill_buffer_mp)
            p.start()
            # while self.read_event.is_set(): time.sleep(0.0001)
            # self.buffer = self.tmp_buffer
            # self.buffer_pointer = 0
            # self.read_event.set()
            print("Got continuing")
            
        res = self.buffer[self.buffer_pointer : self.buffer_pointer + n]
        self.buffer_pointer += n
        return res
        # count = 0
        # text = ""
        # while True:
        #     if self.buffer_pointer >= len(self.buffer):
        #         self._fill_buffer()
        #         text = ""
        #         count = 0
        #     char = self.buffer[self.buffer_pointer]
        #     self.buffer_pointer += 1
        #     if char.isspace():
        #         count += 1
        #         if count >= n:
        #             return [int(idx) for idx in text.split()]
        #     text += char
        
    def _fill_buffer_mp(self, char_count: int = 2097152):
        print("Reading")
        text = self.corpus_fp.read(char_count)
        if len(text) < char_count:
            print("Consumed all of the corpus.")
            # Raise error when all sequences are read.
            if not self.repeat:
                raise StopIteration()
            print("Rewinding")
            # Or, reset current tokens and move to the beginning of the corpus.
            self.corpus_fp.seek(0)
            self._fill_buffer_mp()
        print("Read")
        self.tmp_buffer = self.vocab.encode(text)
        print("Indexed len = ", len(self.tmp_buffer))
        self.refill.acquire()
        self.refill.value = False
        self.refill.release()
        # time.sleep(0.000001)
          
    # def _fill_buffer_in_bg(self, char_count: int = 2097152):
    #     while True:
    #         self.read_event.clear()
    #         self.read_event.wait(60)
    #         print("Reading")
    #         text = self.corpus_fp.read(char_count)
    #         if len(text) < char_count:
    #             print("Consumed all of the corpus.")
    #             # Raise error when all sequences are read.
    #             if not self.repeat:
    #                 raise StopIteration()
    #             print("Rewinding")
    #             # Or, reset current tokens and move to the beginning of the corpus.
    #             self.corpus_fp.seek(0)
    #             continue
    #         print("Read")
    #         self.tmp_buffer = self.vocab.encode(text)
    #         print("Indexed")
            # time.sleep(0.000001)
    
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
