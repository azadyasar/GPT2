from typing import Union, List
import youtokentome as yttm
import sentencepiece as spm

class VocabSP(object):
    def __init__(self,
                 tokenizer_path: str,
                 unk_token: str = '<unk>',
                 bos_token: str = '<s>',
                 eos_token: str = '</s>',
                 pad_token: str = '<pad>'):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.LoadFromFile(tokenizer_path)
        self.sep_id = 4
        self.pad_token = self.tokenizer.IdToPiece(self.tokenizer.pad_id())
        self.unk_token = self.tokenizer.IdToPiece(self.tokenizer.unk_id())
        self.bos_token = self.tokenizer.IdToPiece(self.tokenizer.bos_id())
        self.eos_token = self.tokenizer.IdToPiece(self.tokenizer.eos_id())
        self.sep_token = self.tokenizer.IdToPiece(self.sep_id)

        self.vocab = {self.tokenizer.IdToPiece(id): id 
                      for id in range(self.tokenizer.piece_size())}
        self.words = list(self.vocab.keys())

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.Encode(text.lower())

    def decode_from_ids(self, token_ids: List[int]) -> str:
        return self.tokenizer.Decode(token_ids)
        
    def encode_as_pieces(self, text: str) -> List[int]:
        return self.tokenizer.EncodeAsPieces(text.lower())
    
    def id_to_piece(self, id: int) -> str:
        return self.tokenizer.IdToPiece(id)

    def __getitem__(self, idx_or_token: Union[int, str]) -> Union[str, int]:
        if isinstance(idx_or_token, str):
            return self.vocab[idx_or_token]
        else:
            return self.tokenizer.IdToPiece(idx_or_token)

    def __contains__(self, token: str) -> bool:
        return token in self.vocab

    def __len__(self) -> int:
        # Note that vocabulary size must be a multiple of 8 although the actual
        # number of words is less than it.
        return (self.tokenizer.vocab_size() + 7) // 8 * 8

    @property
    def unk_idx(self) -> int:
        return self.vocab[self.unk_token]

    @property
    def bos_idx(self) -> int:
        return self.vocab[self.bos_token]

    @property
    def eos_idx(self) -> int:
        return self.vocab[self.eos_token]

    @property
    def pad_idx(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def sep_idx(self) -> int:
        return self.vocab[self.sep_token]


class VocabYTTM(object):
    def __init__(self,
                 tokenizer_path: str,
                 unk_token: str = '<unk>',
                 bos_token: str = '<s>',
                 eos_token: str = '</s>',
                 pad_token: str = '<pad>'):
        self.tokenizer = yttm.BPE(tokenizer_path)
        self.pad_token = self.tokenizer.id_to_subword(0)
        self.unk_token = self.tokenizer.id_to_subword(1)
        self.bos_token = self.tokenizer.id_to_subword(2)
        self.eos_token = self.tokenizer.id_to_subword(3)

        self.words = self.tokenizer.vocab()
        self.vocab = {word: i for i, word in enumerate(self.words)}

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text.lower())

    def decode_from_ids(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)
        

    def __getitem__(self, idx_or_token: Union[int, str]) -> Union[str, int]:
        if isinstance(idx_or_token, str):
            return self.vocab[idx_or_token]
        else:
            return self.tokenizer.id_to_subword(idx_or_token)

    def __contains__(self, token: str) -> bool:
        return token in self.vocab

    def __len__(self) -> int:
        # Note that vocabulary size must be a multiple of 8 although the actual
        # number of words is less than it.
        return (self.tokenizer.vocab_size() + 7) // 8 * 8

    @property
    def unk_idx(self) -> int:
        return self.vocab[self.unk_token]

    @property
    def bos_idx(self) -> int:
        return self.vocab[self.bos_token]

    @property
    def eos_idx(self) -> int:
        return self.vocab[self.eos_token]

    @property
    def pad_idx(self) -> int:
        return self.vocab[self.pad_token]
