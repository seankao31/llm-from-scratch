import tiktoken
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from gpt_dataset import GPTDataset
from gpt_download import download_and_load_gpt2
from gpt_config import GPTConfig
from gpt_model import GPTModel
from segmented_timer import SegmentedTimer
from spam_dataset import SpamDataset
from spam_download import DATA_FILE_PATH

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Smaller context length for training on single laptop
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

GPT_CONFIGS = {
    # Small
    "124M": GPTConfig(
        vocab_size=50257,
        context_length=1024,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        drop_rate_attn=0.1,
        drop_rate_shortcut=0.1,
        drop_rate_emb=0.1,
        qkv_bias=False
    ),
    # Medium
    "355M": GPTConfig(
        vocab_size=50257,
        context_length=1024,
        emb_dim=1024,
        n_heads=16,
        n_layers=24,
        drop_rate_attn=0.1,
        drop_rate_shortcut=0.1,
        drop_rate_emb=0.1,
        qkv_bias=False
    ),
    # Large
    "774M": GPTConfig(
        vocab_size=50257,
        context_length=1024,
        emb_dim=1280,
        n_heads=20,
        n_layers=36,
        drop_rate_attn=0.1,
        drop_rate_shortcut=0.1,
        drop_rate_emb=0.1,
        qkv_bias=False
    ),
    # XL
    "1558M": GPTConfig(
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
}

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

def test_train():
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

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt: GPTModel, params: dict[str, list[dict]]):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # Weight tying. Reuse token embedding weights in the output layer
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def test_pretrained_models():
    model_size = "124M"
    for model_size, cfg in GPT_CONFIGS.items():
        print("Loading ", model_size)

        tokenizer = tiktoken.get_encoding("gpt2")
        device = DEVICE

        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
        # Bias vectors are not commonly used in LLMs anymore.
        # However, it was used for the pretrained GPT2 weights.
        cfg.qkv_bias = True

        gpt = GPTModel(cfg)
        gpt.eval()
        load_weights_into_gpt(gpt, params)
        gpt.to(device)

        torch.manual_seed(123)
        token_ids = generate_text(
            model=gpt,
            idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
            max_new_tokens=25,
            context_size=cfg.context_length,
            top_k=50,
            temperature=1.5
        )
        print("Output text:")
        print(token_ids_to_text(token_ids, tokenizer))

        file_path = "the-verdict.txt"
        text_data = load_file(file_path)

        data_loader = create_dataloader(
            text_data,
            batch_size=2,
            max_length=cfg.context_length,
            stride=cfg.context_length,
            drop_last=True,
            shuffle=True,
            num_workers=0
        )
        loss = calc_loss_loader(data_loader, gpt, device)
        print("Loss: ", loss)

def create_balanced_dataset(df: pd.DataFrame):
    """
    Simply undersample the dataset. There are other methods to handle class imbalances.
    """
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df

def random_split(df: pd.DataFrame, train_frac, validation_frac):
    """
    Split the dataset into three parts: training, validation, testing.
    """
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df

def prepare_spam():
    df = pd.read_csv(DATA_FILE_PATH, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)

def spam_dataloader():
    tokenizer = tiktoken.get_encoding("gpt2")
    # pad_to_full_context_length=False
    train_dataset = SpamDataset("train.csv", tokenizer)
    validation_dataset = SpamDataset(
        "validation.csv", tokenizer, max_length=train_dataset.max_length)
    test_dataset = SpamDataset("test.csv", tokenizer, max_length=train_dataset.max_length)

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )

    for input_batch, target_batch in train_loader:
        pass
    print(input_batch.shape)
    print(target_batch.shape)

    print(f"{len(train_loader)} train batches")
    print(f"{len(validation_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

def fine_tune_spam(cfg: GPTConfig, model: GPTModel, fine_tune_whole=False):
    if not fine_tune_whole:
        for param in model.parameters():
            param.requires_grad = False
        # Make last transformer block and final layer normalization trainable as well
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True

    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=cfg.emb_dim, out_features=num_classes)

    # first_token=False

def main():
    model_size = "124M"
    tokenizer = tiktoken.get_encoding("gpt2")
    settings, params = download_and_load_gpt2(model_size, "gpt2")
    cfg = GPT_CONFIGS["124M"]
    cfg.qkv_bias = True
    # drop out 0?
    model = GPTModel(cfg)
    load_weights_into_gpt(model, params)
    model.eval()
    text_1 = "Every effort moves you"
    token_ids = generate_text(model, text_to_token_ids(text_1, tokenizer), 15, cfg.context_length)
    print(token_ids_to_text(token_ids, tokenizer))

    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )
    token_ids = generate_text(model, text_to_token_ids(text_2, tokenizer), 23, cfg.context_length)
    print(token_ids_to_text(token_ids, tokenizer))

    fine_tune_spam(cfg, model)

if __name__ == "__main__":
    main()
