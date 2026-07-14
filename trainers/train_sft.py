"""
Train SFT for mulun model — text + decision joint training.
"""
import os, sys, json, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_here = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_here)  # mulun root
sys.path.insert(0, _ROOT)
sys.path.insert(0, _here)

from transformers import PreTrainedTokenizerFast
from mulun_model import MulunConfig, MulunForCausalLM


class SecuritySFTDataset(Dataset):
    """Simple SFT dataset from parquet or jsonl."""

    def __init__(self, data_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if data_path.endswith('.parquet'):
            import pyarrow.parquet as pq
            t = pq.read_table(data_path)
            self.data = t.to_pylist()
        elif data_path.endswith('.jsonl'):
            with open(data_path, encoding='utf-8') as f:
                self.data = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported format: {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if isinstance(sample, dict) and 'conversations' in sample:
            conversations = sample['conversations']
            prompt = self.tokenizer.apply_chat_template(
                conversations, tokenize=False, add_generation_prompt=False
            )
        elif isinstance(sample, dict) and 'text' in sample:
            prompt = sample['text']
        elif isinstance(sample, dict) and 'prompt' in sample:
            prompt = sample['prompt']
        else:
            prompt = str(sample)

        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
        )
        input_ids = encoding['input_ids']
        labels = input_ids.copy()

        # Structured labels for decision head (optional)
        structured = sample.get('structured', {})
        strategy = structured.get('strategy', -1)
        action = structured.get('action', -1)
        value = structured.get('value', -1.0)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'strategy': torch.tensor(strategy, dtype=torch.long),
            'action': torch.tensor(action, dtype=torch.long),
            'value': torch.tensor(value, dtype=torch.float),
        }


class TrajectoryDataset(Dataset):
    """
    Dataset for trajectory-level data (multi-step conversations).

    Each sample contains:
      - conversations: multi-turn dialogue with multiple assistant <think> blocks
      - trajectory_steps: [{strategy, action, value}, ...] per decision step

    Decision positions are identified by finding <think> tokens (id=25) in input_ids.
    The k-th <think> token corresponds to the k-th trajectory_steps entry.
    """

    THINK_TOKEN_ID = 25

    def __init__(self, data_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        conversations = sample['conversations']
        steps = sample.get('trajectory_steps', [])

        prompt = self.tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False
        )
        encoding = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        input_ids = encoding['input_ids']
        labels = input_ids.copy()

        # Find decision step positions via <think> token
        think_positions = [i for i, t in enumerate(input_ids) if t == self.THINK_TOKEN_ID]

        # Pad/truncate step targets to match number of think positions
        n_steps = len(steps)
        n_think = len(think_positions)

        if n_think == 0:
            # No decision steps in this sample (shouldn't happen for trajectory data)
            strategies, actions, values = [], [], []
        else:
            # Match steps to think positions
            steps_used = steps[:n_think]
            if n_steps < n_think:
                steps_used.extend([steps[-1]] * (n_think - n_steps))

            strategies = [s['strategy'] for s in steps_used]
            actions = [s['action'] for s in steps_used]
            values = [s['value'] for s in steps_used]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'strategies': torch.tensor(strategies, dtype=torch.long),
            'actions': torch.tensor(actions, dtype=torch.long),
            'values': torch.tensor(values, dtype=torch.float),
        }


def _traj_collate(batch):
    """Collate function for TrajectoryDataset — pads step tensors to max length in batch."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        if k in ('strategies', 'actions', 'values'):
            # Pad to max steps in batch
            max_len = max(b[k].shape[0] for b in batch)
            padded = [torch.nn.functional.pad(b[k], (0, max_len - b[k].shape[0]), value=-1) for b in batch]
            result[k] = torch.stack(padded)
        else:
            result[k] = torch.stack([b[k] for b in batch])
    return result


def train_sft(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Tokenizer ──
    tokenizer_path = args.tokenizer_dir or os.path.join(_ROOT, 'tokenizer')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"Tokenizer loaded: {len(tokenizer)} vocab")

    # ── Config ──
    config = MulunConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        vocab_size=len(tokenizer),
        state_dim=args.state_dim,
        n_actions=args.n_actions,
        max_position_embeddings=args.max_seq_len,
    )
    print(f"Config: {config.hidden_size}/{config.num_hidden_layers}, "
          f"vocab={config.vocab_size}, decision={config.n_actions} actions")

    # ── Model ──
    model = MulunForCausalLM(config, init_decision_head=True).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"  Decision head: {sum(p.numel() for p in model.decision_head.parameters())/1e3:.1f}K")

    # Try loading pretrained backbone
    if args.from_weight and os.path.exists(args.from_weight):
        print(f"Loading backbone from {args.from_weight} ...")
        ckpt = torch.load(args.from_weight, map_location=device, weights_only=False)
        # Only load matching backbone keys
        model_sd = model.state_dict()
        for k, v in ckpt.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v.to(dtype=model_sd[k].dtype, device=device)
        model.load_state_dict(model_sd, strict=False)
        print(f"  Loaded {sum(1 for k in ckpt if k in model_sd)}/{len(ckpt)} keys")

    # ── Data (auto-detect trajectory format) ──
    with open(args.data_path, encoding='utf-8') as _f:
        _first = json.loads(_f.readline())
    is_trajectory = 'trajectory_steps' in _first

    if is_trajectory:
        from trainers.train_sft import TrajectoryDataset
        dataset = TrajectoryDataset(args.data_path, tokenizer, args.max_seq_len)
        print(f"Using TrajectoryDataset ({len(dataset)} samples)")
    else:
        dataset = SecuritySFTDataset(args.data_path, tokenizer, args.max_seq_len)
        print(f"Using SecuritySFTDataset ({len(dataset)} samples)")
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers,
                         collate_fn=_traj_collate if is_trajectory else None)

    # ── Optimizer ──
    # lm_head.weight is tied to embed_tokens.weight, so it's already in backbone
    optimizer = optim.AdamW([
        {'params': model.model.parameters(), 'lr': args.lr * 0.1},  # backbone
        {'params': model.decision_head.parameters(), 'lr': args.lr},  # decision head
    ], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader))

    # ── Training ──
    scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}', ncols=100)

        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            kwargs = {}
            if is_trajectory:
                kwargs['step_strategies'] = batch['strategies'].to(device)
                kwargs['step_actions'] = batch['actions'].to(device)
                kwargs['step_values'] = batch['values'].to(device)
            else:
                kwargs['step_strategies'] = batch['strategy'].unsqueeze(1).to(device)
                kwargs['step_actions'] = batch['action'].unsqueeze(1).to(device)
                kwargs['step_values'] = batch['value'].unsqueeze(1).to(device)

            with torch.amp.autocast(device_type='cuda', enabled=(args.dtype != 'float32')):
                out = model(
                    input_ids, labels=labels, mode='decision', **kwargs
                )
                loss = out.loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * args.accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * args.accumulation_steps:.4f}'})
            scheduler.step()

        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch+1}: avg_loss={avg_loss:.4f}')

        # Save checkpoint
        tag = f'_{args.run_name}' if args.run_name else ''
        if (epoch + 1) % args.save_every == 0:
            save_path = f'{args.save_dir}/mulun{tag}_epoch{epoch+1}.pth'
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            config.save_pretrained(args.save_dir)
            print(f'  Saved: {save_path}')

    # Final save
    final_path = f'{args.save_dir}/mulun{tag}_final.pth'
    torch.save(model.state_dict(), final_path)
    config.save_pretrained(args.save_dir)
    print(f'\nFinal model: {final_path}')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mulun SFT training")
    parser.add_argument('--data-path', type=str, required=True, help='Training data (parquet/jsonl)')
    parser.add_argument('--tokenizer-dir', type=str, default=None, help='Tokenizer directory')
    parser.add_argument('--from-weight', type=str, default=None, help='Pretrained weight path')
    parser.add_argument('--save-dir', type=str, default='../out', help='Save directory')
    parser.add_argument('--hidden-size', type=int, default=768, help='Transformer hidden size')
    parser.add_argument('--num-hidden-layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--state-dim', type=int, default=64, help='Decision state dimension')
    parser.add_argument('--n-actions', type=int, default=8, help='Number of actions')
    parser.add_argument('--max-seq-len', type=int, default=1024, help='Max sequence length')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Per-device batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--accumulation-steps', type=int, default=4, help='Gradient accumulation')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--use-moe', type=int, default=0, help='Use MoE')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Training dtype')
    parser.add_argument('--device', type=str, default='cuda:0', help='Training device')
    parser.add_argument('--num-workers', type=int, default=2, help='Data loader workers')
    parser.add_argument('--save-every', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--run-name', type=str, default='', help='Tag for output filenames (e.g. state64)')
    args = parser.parse_args()

    train_sft(args)
