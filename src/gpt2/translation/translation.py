import torch
from gpt2.modeling import Past
from gpt2.translation import TranslationSpec, TranslationConfig
from typing import List, Optional, Tuple

class Translator(object):
    def __init__(self, spec: TranslationSpec, config: TranslationConfig):
        self.spec = spec
        self.config = config
    
    def initialize(self, from_model: Optional[str] = None):
        self.spec.initialize()
        self.model = self.spec.construct_model().eval()
        
        if from_model:
            ckpt = torch.load(from_model, map_location='cpu')
            self.model.load_state_dict(ckpt['model'])
        
        if self.config.use_gpu:
            self.model.cuda().half()
    
    def translate(self, source: str) -> str:
        words = self.spec.encode_context(source.lower())
        
        current, past = words, None
        while len(words) < self.config.seq_len:
            # Predict the next word token from the given context.
            probs, past = self._predict_probs(current, past)
            next_word = self._sample_from_top_p(probs)
            
            # Change the context to the predicted word.
            words.append(next_word)
            current = [next_word]
        return self.spec.decode_tokens(words)
    
    @torch.no_grad()
    def _predict_probs(self,
                       words: List[int],
                       past: Optional[List[Past]] = None
                       ) -> Tuple[torch.Tensor, List[Past]]:
        x = torch.tensor(words, dtype=torch.long)
        x = self.spec.decorate_sequence(
            x, offset=past[0][0].size(-2) if past is not None else 0)
        
        if self.config.use_gpu:
            logits, past = self.model(x.cuda(), past)
            logits = logits.cpu().float()
        else:
            logits, past = self.model(x, past)
            
        return logits[-1, :].softmax(-1), past
    
    def _sample_from_top_p(self, probs: torch.Tensor) -> int:
        return probs.argmax().item()