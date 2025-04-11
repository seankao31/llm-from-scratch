import tiktoken
import torch
from torch.utils.data import DataLoader
from gpt_dataset import GPTDataset
from gpt_config import GPTConfig
from gpt_model import GPTModel

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

def test_model(cfg: GPTConfig):
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    # print(batch)

    torch.manual_seed(123)
    model = GPTModel(cfg)
    logits = model(batch)
    print("Output shape:", logits.shape)
    # print(logits)

    att_params = sum(p.numel() for p in model.trf_blocks[0].att.parameters())
    print(f"Number of parameters in one attention module: {att_params:,}")
    ff_params = sum(p.numel() for p in model.trf_blocks[0].ff.parameters())
    print(f"Number of parameters in one feed forward module: {ff_params:,}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    # Assume float32, 4 bytes per parameter
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")

def main():
    # GPT-2 small
    GPT_CONFIG_124M = GPTConfig(
        vocab_size=50257,
        context_length=1024,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        drop_rate=0.1,
        qkv_bias=False
    )

    GPT_CONFIG_MEDIUM = GPTConfig(
        vocab_size=50257,
        context_length=1024,
        emb_dim=1024,
        n_heads=16,
        n_layers=24,
        drop_rate=0.1,
        qkv_bias=False
    )

    GPT_CONFIG_LARGE = GPTConfig(
        vocab_size=50257,
        context_length=1024,
        emb_dim=1280,
        n_heads=20,
        n_layers=36,
        drop_rate=0.1,
        qkv_bias=False
    )

    GPT_CONFIG_XL = GPTConfig(
        vocab_size=50257,
        context_length=1024,
        emb_dim=1600,
        n_heads=25,
        n_layers=48,
        drop_rate=0.1,
        qkv_bias=False
    )

    configs = [GPT_CONFIG_124M, GPT_CONFIG_MEDIUM, GPT_CONFIG_LARGE, GPT_CONFIG_XL]
    for cfg in configs:
        test_model(cfg)

if __name__ == "__main__":
    main()
