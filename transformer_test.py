from language_model_dataset import LanguageModelDataSet
from transformer import Attention, Transformer
import torch
from torch import nn

def test_lm_dataset_len():
    corpus = [[5,8,3,9],
              [0,5],
              [63,201, 2]]
    lm_dataset = LanguageModelDataSet(corpus)
    # Each text can only produce n-1 training examples,
    # because we need at least one data token and one target token
    assert len(lm_dataset) == 6


def test_lm_dataset_getitem():
    corpus = [[5, 8, 3, 9],
              [0, 5],
              [63, 201, 2]]

    lm_dataset = LanguageModelDataSet(corpus)
    # Each text can only produce n-1 training examples,
    # because we need at least one data token and one target token
    data, target = lm_dataset[5]
    assert torch.equal(data, torch.tensor([63, 201], dtype=torch.long))
    assert torch.equal(target, torch.tensor(2, dtype=torch.long))


def test_lm_dataset_completeness():
    corpus = [[5, 8, 3, 9],
              [0, 5],
              [63, 201, 2]]
    lm_dataset = LanguageModelDataSet(corpus)
    lm_dataset_list = [lm_dataset[idx] for idx in range(len(lm_dataset))]
    i = 0
    for text in corpus:
        for target_idx, target in enumerate(text):
            if target_idx == 0:
                continue
            sample_data = torch.tensor(text[:target_idx], dtype=torch.long)
            sample_target = torch.tensor(target, dtype=torch.long)
            data, target = lm_dataset_list[i]
            assert torch.equal(sample_data, data) and torch.equal(sample_target, target)
            i += 1


def test_attention():
    with torch.no_grad():
        attention_layer = Attention(n_heads=1, embedding_dim=2, key_query_dim=2)
        attention_layer.key_matrices.data = torch.unsqueeze(torch.eye(2), 0)  # Identity matrix
        attention_layer.query_matrices.data = torch.ones((1, 2, 2))
        attention_layer.value_in_matrices.data = torch.unsqueeze(torch.eye(2), 0)
        attention_layer.value_out_matrices.data = torch.unsqueeze(torch.eye(2), 0)
        data = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float)

        # Queries will be [1, 1] for both tokens
        # Dot product of keys and queries will be 1 in both cases
        # Token 0 should only attend to itself (autoregressive)
        # Token 1 should attend equally to both tokens
        # Result should therefore be [[[1, 0], [0.5, 0.5]]]

        expected_output = torch.tensor([[[1, 0], [0.5, 0.5]]], dtype=torch.float)
        output = attention_layer(data)
        assert expected_output.equal(output)


def test_parameter_count():
    with torch.no_grad():
        vocab_size = 2048
        embedding_dim = 128
        n_heads = 8
        n_layers = 4
        max_context_window = 64
        key_query_dim = 32
        mlp_dim = 256
        model = Transformer(vocab_size=vocab_size, embedding_dim=embedding_dim, n_heads=n_heads,
                            n_layers=n_layers, max_context_window=max_context_window,
                            key_query_dim=key_query_dim, mlp_dim=mlp_dim)

        def count_parameters(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        embedding_param_count = vocab_size * embedding_dim
        assert embedding_param_count == count_parameters(model.embedding)

        positional_enc_params = embedding_dim * max_context_window
        assert positional_enc_params == model.positional_encoding.numel()

        attention_params_per_layer = 4 * n_heads * embedding_dim * key_query_dim  # Four stacks of matrices per layer
        for attention_layer in model.attention_layers:
            assert attention_params_per_layer == count_parameters(attention_layer)

        mlp_params_per_layer = embedding_dim * mlp_dim + mlp_dim  # weights and biases for expansion
        mlp_params_per_layer += mlp_dim * embedding_dim + embedding_dim  # weights and biases for reduction
        for mlp_layer in model.multilayer_perceptrons:
            assert mlp_params_per_layer == count_parameters(mlp_layer)

        unembedding_params = embedding_dim * vocab_size  # Unembedding layer
        assert unembedding_params == count_parameters(model.unembedding)

        parameter_count = embedding_param_count
        parameter_count += positional_enc_params
        parameter_count += n_layers * attention_params_per_layer
        parameter_count += n_layers * mlp_params_per_layer
        parameter_count += unembedding_params
        assert parameter_count == count_parameters(model)

