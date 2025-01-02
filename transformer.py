import pickle

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim, einsum
from preprocessing import create_lm_dataloaders
from byte_pair_encoding import Tokenizer
from typing import Any
#from torchviz import make_dot


class Attention(nn.Module):
    def __init__(self, n_heads: int = 1, embedding_dim: int = 256, key_query_dim: int = 64):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.key_query_dim = key_query_dim
        self.key_matrices = nn.Parameter(0.1*torch.randn((n_heads, embedding_dim, self.key_query_dim)))
        self.query_matrices = nn.Parameter(0.1*torch.randn((n_heads, embedding_dim, self.key_query_dim)))
        self.value_in_matrices = nn.Parameter(0.1*torch.randn((n_heads, embedding_dim, self.key_query_dim)))
        self.value_out_matrices = nn.Parameter(0.1*torch.randn((n_heads, self.key_query_dim, embedding_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        queries = einsum("nkl, ijk -> nijl", self.query_matrices, x)
        keys = einsum("nkl, ijk -> nijl", self.key_matrices, x)
        key_query_products = einsum("nijl, nikl-> nijk", keys, queries)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        key_query_products = key_query_products.masked_fill(mask, float('-inf'))
        attention_pattern = F.softmax(key_query_products, 3)
        return einsum("nikj, ijp, npl, nlm -> ikm", attention_pattern, x,
                      self.value_in_matrices, self.value_out_matrices)


class MLP(nn.Module):
    def __init__(self, embedding_dim: int, inner_dim: int, drop_out_p=0.5):
        super(MLP, self).__init__()
        self.mlp_in = nn.Linear(embedding_dim, inner_dim)
        self.drop_out = torch.nn.Dropout(p=drop_out_p)
        self.mlp_out = nn.Linear(inner_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.mlp_in(x))
        x = self.drop_out(x)
        x = self.mlp_out(x)
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 256, max_context_window: int = 512,
                 n_heads=1,  n_layers: int = 1, key_query_dim=128, mlp_dim=512):
        super(Transformer, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_context_window = max_context_window
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(0.001*torch.randn((max_context_window, embedding_dim)))

        self.attention_layers = nn.ModuleList([Attention(n_heads=n_heads, embedding_dim=embedding_dim, key_query_dim=key_query_dim) for _ in range(n_layers)])
        self.multilayer_perceptrons = nn.ModuleList([MLP(embedding_dim, mlp_dim) for _ in range(n_layers)])

        self.unembedding = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.numel() == 0:
            raise ValueError("Input must have at least one token")
        if tokens.shape[1] > self.max_context_window:
            tokens = tokens[:, -self.max_context_window:]
        x = self.embedding(tokens)
        encoded_positions = self.positional_encoding[:tokens.shape[1]]
        x = x + encoded_positions
        for attention_layer, mlp_layer in zip(self.attention_layers, self.multilayer_perceptrons):
            x = x - torch.mean(x, dim=2).unsqueeze(2)
            x = x / torch.sqrt(torch.var(x, dim=2).unsqueeze(2))
            x = x + attention_layer(x)
            x = x + mlp_layer(x)
        x = self.unembedding(x)
        return x


def transformer_loss(data: torch.Tensor, model_output: torch.Tensor, loss_fn):
    targets = data[:, 1:].detach()
    if model_output.shape[1] < targets.shape[1]:
        targets = targets[:, -model_output.shape[1]:]
    output = model_output.view(-1, model_output.shape[-1])
    targets = targets.reshape(-1)
    return loss_fn(output, targets)


def show_training_info(batch_index: int, epoch: Any, data: torch.Tensor, output: torch.Tensor,
                       batch_size: int, dataset_size, loss, tokenizer: Tokenizer):
    num_tokens_to_display = 32
    truncated_output = output[0, :num_tokens_to_display, :]
    truncated_data = data[0, :num_tokens_to_display]
    probabilities = F.softmax(truncated_output[-1, :], dim=0)
    most_likely_token = int(torch.argmax(probabilities))
    highest_probability = probabilities[most_likely_token]
    correct_token = int(data[0][num_tokens_to_display])
    correct_token_probability = probabilities[correct_token]
    print(f"Epoch: {epoch}, [{batch_index * batch_size}/{dataset_size}, "
          f"loss={float(loss):.2f}, text_length={min(len(text) for text in data)}]"
          f" '{"".join(tokenizer.detokenize(truncated_data[num_tokens_to_display-8:])).replace('\n', '\\n')}' :: "
          f"'{tokenizer.token_to_str(most_likely_token).replace('\n', '\\n')}'; {highest_probability * 100:.1f}%', "
          f"Correct: '{tokenizer.token_to_str(correct_token).replace('\n', '\\n')}'; {correct_token_probability * 100:.1f}%")


def model_train(epoch, loaders, model, optimiser, loss_fn, device, tokenizer: Tokenizer):
    model.train()
    for batch_index, data in enumerate(loaders["train"]):
        data = data.to(device)
        optimiser.zero_grad()
        output = model(data[:, :-1])
        loss = transformer_loss(data, output, loss_fn)

        show_training_info(batch_index, epoch, data, output, loaders["train"].batch_size,
                           len(loaders["train"].dataset), loss, tokenizer)
        loss.backward()
        optimiser.step()


def model_test(test_data_loader, model, loss_fn, device, tokenizer):
    model.eval()

    with torch.no_grad():
        for idx, data in enumerate(test_data_loader):
            data = data.to(device)
            output = model(data[:, :-1])
            loss = transformer_loss(data, output, loss_fn)
            show_training_info(idx, "test", data, output, test_data_loader.batch_size,
                               len(test_data_loader.dataset), loss, tokenizer)


def main():
    loaders, tokens = create_lm_dataloaders()
    tokenizer = Tokenizer(tokens)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(vocab_size=len(tokens), embedding_dim=128, n_heads=8,
                        n_layers=8, max_context_window=128, key_query_dim=32,
                        mlp_dim=512).to(device)

    model.load_state_dict(torch.load(f"transformer_big.pth", weights_only=True))

    optimiser = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    #model_test(loaders["test"], model, loss_fn, device, tokenizer)
    for epoch in range(1000):
        model_train(epoch, loaders, model, optimiser, loss_fn, device, tokenizer)
        if epoch % 1 == 0:
            model_test(loaders["test"], model, loss_fn, device, tokenizer)
            torch.save(model.state_dict(), "transformer_big.pth")


if __name__ == "__main__":
    main()
