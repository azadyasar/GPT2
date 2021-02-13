import argparse
import torch.nn as nn
from gpt2.data import VocabSP, VocabYTTM
from gpt2.modeling import Transformer
from gpt2.translation import TranslationSpec, TranslationConfig, Translator
from typing import List
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from random import random

class GPT2TranslationSpec(TranslationSpec):
    def __init__(self, vocab_path: str, seq_len: int, layers: int, heads: int,
                 dims: int, rate: int, is_sentencepiece: bool = True):
        self.vocab_path = vocab_path
        self.seq_len = seq_len
        self.layers = layers
        self.heads = heads
        self.dims = dims
        self.rate = rate
        self.is_sentencepiece = is_sentencepiece

    def initialize(self):
        if self.is_sentencepiece:
            self.vocab = VocabSP(tokenizer_path=self.vocab_path)
        else:
            self.vocab = VocabYTTM(tokenizer_path=self.vocab_path)
        # self.tokenizer = Tokenizer(vocab=self.vocab)

    def construct_model(self) -> nn.Module:
        return Transformer(layers=self.layers, pad_idx=self.vocab.pad_idx,
                           words=len(self.vocab), seq_len=self.seq_len,
                           heads=self.heads, dims=self.dims, rate=self.rate,
                           dropout=0, bidirectional=False)

    def encode_context(self, context: str) -> List[int]:
        return [self.vocab.bos_idx] + self.vocab.encode(context) + [self.vocab.sep_idx]

    def decode_tokens(self, tokens: List[int]) -> str:
        if self.vocab.eos_idx in tokens:
            tokens = tokens[:tokens.index(self.vocab.eos_idx) + 1]
        return self.vocab.decode_from_ids(tokens)

def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):
    # assert n_rows * n_cols == n_heads

    if isinstance(sentence, str):
        # sentence = preprocess_and_tokenize(sentence)
        sentence = sentence

    fig = plt.figure(figsize=(32,32))
    # plt.title("Input = " + tr_sp.decode(sentence) + " | Translation = " + en_sp.decode(translation), y=0.5)
    plt.axis('off')
    show_n_heads = 2
    # plt.rc('grid', linestyle="-", color='white')
    # plt.grid(True)
    for i in range(n_heads//show_n_heads):
        ax = fig.add_subplot(n_rows/show_n_heads, n_cols, i+1)
        # plt.grid(True, linestyle="--", color='pink', linewidth=0.5)
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()
        cax = ax.matshow(_attention, cmap='bone', aspect='auto')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + sentence,
                            rotation=45)
        ax.set_yticklabels([''] + translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.tight_layout()
    # fig.subplots_adjust(hspace=0.25, wspace=0.25)
    plt.savefig(str(random() * 1000) + ".png")
    # plt.show()
    
def translate_and_viz(args: argparse.Namespace):
    spec = GPT2TranslationSpec(
        vocab_path=args.vocab_path, seq_len=args.seq_len, layers=args.layers,
        heads=args.heads, dims=args.dims, rate=args.rate, is_sentencepiece=args.is_sp == 1)
    config = TranslationConfig(
        seq_len=args.seq_len, nucleus_prob=args.nucleus_prob,
        use_gpu=args.use_gpu)

    translator = Translator(spec, config)
    translator.initialize(from_model=args.model_path)

    while True:
        input_seq = input('>>')
        result, words, attn, input_attn = translator.translate_with_attn(input_seq)
        tokens = [spec.vocab.id_to_piece(w) for w in words]
        display_attention(tokens, tokens, attn, n_heads=spec.heads, n_rows=spec.heads//2, n_cols=2)
        # ids = spec.encode_context(input_seq)
        # tokens = [spec.vocab.id_to_piece(id) for id in ids]
        # display_attention(tokens, tokens, input_attn, n_heads=spec.heads, n_rows=spec.heads//2, n_cols=2)
        print(result)
        

def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        'transviz', help='translate sentences from English to Turkish with GPT-2 model')

    parser.add_argument('--vocab_path', required=True,
                        help='vocabulary file path')
    parser.add_argument('--is_sp', type=int,
                       help='is tokenizer a sentencepiece model')
    parser.add_argument('--model_path', required=True,
                        help='trained GPT-2 model file path')

    group = parser.add_argument_group('Model configurations')
    group.add_argument('--seq_len', default=64, type=int,
                       help='maximum sequence length')
    group.add_argument('--layers', default=12, type=int,
                       help='number of transformer layers')
    group.add_argument('--heads', default=16, type=int,
                       help='number of multi-heads in attention layer')
    group.add_argument('--dims', default=784, type=int,
                       help='dimension of representation in each layer')
    group.add_argument('--rate', default=4, type=int,
                       help='increase rate of dimensionality in bottleneck')

    group = parser.add_argument_group('Generation options')
    group.add_argument('--nucleus_prob', default=0.85, type=float,
                       help='probability threshold for nucleus sampling')
    group.add_argument('--use_gpu', action='store_true',
                       help='use gpu device in inferencing')

    parser.set_defaults(func=translate_and_viz)