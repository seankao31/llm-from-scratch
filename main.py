import tiktoken
import torch
from torch.utils.data import DataLoader
from gpt_dataset import GPTDataset
from gpt_config import GPTConfig
from gpt_model import GPTModel
from segmented_timer import SegmentedTimer

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Smaller model for training on single laptop
GPT_CONFIG_124M_MINI = GPTConfig(
    vocab_size=50257,
    context_length=256,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    drop_rate_attn=0.1,
    drop_rate_shortcut=0.1,
    drop_rate_emb=0.1,
    qkv_bias=False
)

# GPT-2 small
GPT_CONFIG_124M = GPTConfig(
    vocab_size=50257,
    context_length=1024,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    drop_rate_attn=0.1,
    drop_rate_shortcut=0.1,
    drop_rate_emb=0.1,
    qkv_bias=False
)

GPT_CONFIG_MEDIUM = GPTConfig(
    vocab_size=50257,
    context_length=1024,
    emb_dim=1024,
    n_heads=16,
    n_layers=24,
    drop_rate_attn=0.1,
    drop_rate_shortcut=0.1,
    drop_rate_emb=0.1,
    qkv_bias=False
)

GPT_CONFIG_LARGE = GPTConfig(
    vocab_size=50257,
    context_length=1024,
    emb_dim=1280,
    n_heads=20,
    n_layers=36,
    drop_rate_attn=0.1,
    drop_rate_shortcut=0.1,
    drop_rate_emb=0.1,
    qkv_bias=False
)

GPT_CONFIG_XL = GPTConfig(
    vocab_size=50257,
    context_length=1024,
    emb_dim=1600,
    n_heads=25,
    n_layers=48,
    drop_rate_attn=0.1,
    drop_rate_shortcut=0.1,
    drop_rate_emb=0.1,
    qkv_bias=False
)

TOKEN_END_OF_TEXT = '<|endoftext|>'

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

def generate_text(model, idx, max_new_tokens, context_size):
    """
    idx: torch.Tensor (batch, n_tokens)
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx

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

def test_models():
    configs = [GPT_CONFIG_124M, GPT_CONFIG_MEDIUM, GPT_CONFIG_LARGE, GPT_CONFIG_XL]
    for cfg in configs:
        test_model(cfg)

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={TOKEN_END_OF_TEXT})
    # Add batch dimension
    return torch.tensor(encoded).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    total_loss = 0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches

def main():
    file_path = "the-verdict.txt"
    text_data = load_file(file_path)

    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    config = GPT_CONFIG_124M_MINI

    train_loader = create_dataloader(
        train_data,
        batch_size=2,
        max_length=config.context_length,
        stride=config.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader(
        val_data,
        batch_size=2,
        max_length=config.context_length,
        stride=config.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    model = GPTModel(config)
    devices = [
        torch.device("cpu"),
        torch.device("mps")
    ]
    for device in devices:
        print(f"\nTesting on device: {device}")
        with SegmentedTimer() as timer:
            model.to(device)
            timer.mark("Model moved to device")
            with torch.no_grad():
                train_loss = calc_loss_loader(train_loader, model, device)
                val_loss = calc_loss_loader(val_loader, model, device)
            timer.mark("Loss caculated")
        print("Training loss:", train_loss)
        print("Validation loss:", val_loss)

if __name__ == "__main__":
    torch.manual_seed(123)
    main()
