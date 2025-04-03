import tiktoken
import torch
from torch.utils.data import DataLoader
from gpt_dataset import GPTDataset
from self_attention import SelfAttention

def load_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return raw_text

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

def main():
    raw_text = load_file("the-verdict.txt")

    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    max_length = 4
    dataloader = create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print(inputs)
    print(targets)

    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)

    # 4x256 pos_embeddings added to each 4x256 token embedding in batch
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)

def test_attention():
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
         [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],
         [0.77, 0.25, 0.10],
         [0.05, 0.80, 0.55]]
    )
    torch.manual_seed(789)
    d_in = inputs.shape[-1]
    d_out = 2
    sa = SelfAttention(d_in, d_out)
    print(sa(inputs))

if __name__ == "__main__":
    test_attention()
