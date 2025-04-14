import tiktoken
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader
from gpt_dataset import GPTDataset
from gpt_config import GPTConfig
from gpt_model import GPTModel
from segmented_timer import SegmentedTimer

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")
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

def generate_text(model, idx, max_new_tokens, context_size,
                  temperature=0.0, top_k=None, eos_id=None):
    """
    idx: torch.Tensor (batch, n_tokens)
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            # Set the logit values of tokens that are below the top-k selection to -inf
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        if temperature > 0.0:
            # Temperature scaling
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Stop generating early if end-of-sequence token is encountered
        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=-1)
    return idx

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

def evaluate_model(model: GPTModel, train_loader: DataLoader, val_loader: DataLoader,
                   device: torch.device, eval_iter: int):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model: GPTModel, tokenizer: tiktoken.Encoding, device: torch.device,
                              start_context: str):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    # Compact print format
    print(decoded_text.replace("\n", " "))
    model.train()

def train_model(
        model: GPTModel, train_loader: DataLoader, val_loader: DataLoader,
        optimizer:torch.optim.Optimizer, device: torch.device, tokenizer: tiktoken.Encoding,
        num_epochs: int, eval_freq: int, eval_iter: int, start_context: int):
    # For tracking progress
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    # Invisble plot for aligning ticks
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

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
    tokenizer = tiktoken.get_encoding("gpt2")
    device = DEVICE
    torch.manual_seed(123)
    print(f"\nTesting on device: {device}")
    with SegmentedTimer() as timer:
        model.to(device)
        timer.mark("Model moved to device")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
        num_epochs = 10
        timer.skip()
        train_losses, val_losses, tokens_seen = train_model(
            model, train_loader, val_loader, optimizer, device, tokenizer,
            num_epochs=num_epochs, eval_freq=5, eval_iter=5,
            start_context="Every effort moves you"
        )
        timer.mark("Train completed")
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

if __name__ == "__main__":
    main()
