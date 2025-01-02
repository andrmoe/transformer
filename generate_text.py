from transformer import Transformer
import torch
from byte_pair_encoding import Tokenizer
import torch.nn.functional as F


class TextGenerator:
    def __init__(self):
        with open("datasets/gutenberg_tokens.txt", "r", encoding="utf-8") as f:
            list_str = f.read()
        self.tokens = eval(list_str)
        self.tokenizer = Tokenizer(self.tokens)
        self.model = Transformer(vocab_size=len(self.tokens), embedding_dim=128, n_heads=8,
                                 n_layers=8, max_context_window=128, key_query_dim=32,
                                 mlp_dim=512)
        self.model.eval()
        self.model.load_state_dict(torch.load("transformer_big.pth", weights_only=True))

    def generate_next_token(self, prompt_tokens: [int], temp: float = 1.0) -> int:
        with torch.no_grad():
            output = self.model(torch.tensor([prompt_tokens], dtype=torch.long))
            next_logit = output[0, -1, :]
            distribution = F.softmax(next_logit/temp, 0)
            return int(torch.multinomial(distribution, 1))

    def generate(self, prompt: str, length: int = 128, sep="", temp=1.0):
        prompt_tokens = list(self.tokenizer.tokenize(prompt))
        for i in range(length):
            next_token = self.generate_next_token(prompt_tokens, temp)
            prompt_tokens.append(next_token)
        return sep.join(list(self.tokenizer.detokenize(prompt_tokens)))


t = TextGenerator()
