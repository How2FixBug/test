"""Train the custom model from plain-text data. No pretrained weights are used."""
import argparse
import os
import time
import torch
from model import TinyGPT, encode, VOCAB_SIZE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"Training text not found: {args.data}\nCreate it with plain text or chat examples, then run again.")
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = torch.tensor(encode(text), dtype=torch.long)
    if len(tokens) < args.context + 2:
        raise SystemExit(f"Need at least {args.context + 2} encoded tokens; found {len(tokens)}.")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training from scratch | device={device} | bytes={len(text.encode('utf-8')):,} | tokens={len(tokens):,}")
    model = TinyGPT(args.width, args.layers, args.heads, args.context).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    model.train()
    start = time.time()
    for step in range(1, args.steps + 1):
        starts = torch.randint(0, len(tokens) - args.context - 1, (args.batch_size,))
        x = torch.stack([tokens[i:i + args.context] for i in starts]).to(device)
        y = torch.stack([tokens[i + 1:i + args.context + 1] for i in starts]).to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f"step {step:>5}/{args.steps} | loss {loss.item():.4f} | {time.time()-start:.1f}s")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model": model.cpu().state_dict(), "config": vars(args)}, "checkpoints/roger-small.pt")
    print("Saved checkpoints/roger-small.pt")

if __name__ == "__main__":
    main()
