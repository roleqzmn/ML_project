import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram

from batching import RatioBatchSampler
from metrics import best_f1_threshold, metrics

BASE = Path(__file__).resolve().parent
SR = 16000
SECONDS = 2


def read_data(filename):
    filename = Path(filename).resolve()
    with filename.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    groups, files, segments = {}, {}, set()
    for row in rows:
        row["path"] = str((filename.parent / row["path"]).resolve())
        row["label"] = int(row["label"])
        row["start_frame"] = int(row.get("start_frame") or 0)
        row["num_frames"] = int(row["num_frames"]) if row.get("num_frames") else -1
        if row["label"] not in (0, 1) or row["split"] not in ("train", "val", "test"):
            raise ValueError("Expected labels 0/1 and splits train/val/test")
        if row["start_frame"] < 0 or row["num_frames"] == 0 or row["num_frames"] < -1:
            raise ValueError("Invalid audio segment")
        segment = (row["path"], row["start_frame"], row["num_frames"])
        if segment in segments:
            raise ValueError(f"Repeated audio segment: {segment}")
        segments.add(segment)
        for mapping, key in [(groups, row["group_id"]), (files, row["path"])]:
            if mapping.setdefault(key, row["split"]) != row["split"]:
                raise ValueError(f"Audio source occurs in multiple splits: {key}")
    subsets = {split: [r for r in rows if r["split"] == split]
               for split in ("train", "val", "test")}
    for split, subset in subsets.items():
        if {r["label"] for r in subset} != {0, 1}:
            raise ValueError(f"Both classes are required in {split}")
    return subsets


class AudioDataset(Dataset):
    def __init__(self, rows, training=False):
        self.rows = rows
        self.training = training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x, sr = sf.read(row["path"], start=row["start_frame"], frames=row["num_frames"],
                        dtype="float32", always_2d=True)
        if len(x) == 0 or not np.isfinite(x).all():
            raise ValueError(f"Invalid audio: {row['path']}")
        x = x.mean(axis=1)
        if sr != SR:
            divisor = math.gcd(sr, SR)
            x = resample_poly(x, SR // divisor, sr // divisor)
        length = SR * SECONDS
        if len(x) > length:
            start = random.randint(0, len(x) - length) if self.training else (len(x) - length) // 2
            x = x[start:start + length]
        x = x - x.mean()
        x = x * (0.1 / max(float(np.sqrt(np.mean(x ** 2))), 1e-4))
        x = np.pad(x, (0, max(0, length - len(x))))
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(row["label"], dtype=torch.float32)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = MelSpectrogram(sample_rate=SR, n_fft=400, hop_length=160,
                                  n_mels=64, f_min=40, f_max=7600)
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 1)), nn.Flatten(),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, audio):
        spec = self.mel(audio).clamp_min(1e-6).log()
        mean = spec.mean(dim=(-2, -1), keepdim=True)
        std = spec.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        return self.network(((spec - mean) / std).unsqueeze(1)).squeeze(1)


@torch.inference_mode()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    labels, scores = [], []
    total_loss, seen = 0.0, 0
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        total_loss += loss_fn(logits, y.to(device, non_blocking=True)).item() * len(y)
        seen += len(y)
        scores.extend(logits.sigmoid().cpu().tolist())
        labels.extend(y.tolist())
    return labels, scores, total_loss / seen


def save_plot(history, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = [row["epoch"] for row in history]
    panels = [
        ("loss", "val_loss", "Strata"),
        ("train_accuracy", "val_accuracy", "Accuracy"),
        ("train_macro_f1", "val_macro_f1", "Macro-F1"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, (train_key, val_key, title) in zip(axes, panels):
        axis.plot(epochs, [row[train_key] for row in history], label="Trening", color="#2563eb", linewidth=2)
        axis.plot(epochs, [row[val_key] for row in history], label="Walidacja", color="#ea580c", linewidth=2)
        axis.set(title=title, xlabel="Epoka", ylabel=title)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.2)
        if train_key != "loss":
            axis.set_ylim(0, 1.02)
        axis.legend()
    axes[2].axhline(0.6, color="#64748b", linestyle="--", linewidth=1)
    fig.suptitle("Przebieg treningu CNN", fontsize=16)
    fig.savefig(filename, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=BASE / "prepared_data" / "manifest.csv")
    parser.add_argument("--output", type=Path, default=BASE / "results")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.workers < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("epochs >= 1, workers >= 0 and lr > 0 are required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; install PyTorch with CUDA support")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    subsets = read_data(args.csv)
    sampler = RatioBatchSampler([r["label"] for r in subsets["train"]], args.batch_size)
    loader_options = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(AudioDataset(subsets["train"], training=True),
                              batch_sampler=sampler, **loader_options)
    evaluation_loaders = {
        split: DataLoader(AudioDataset(rows), batch_size=args.batch_size, **loader_options)
        for split, rows in subsets.items()
    }
    model = CNN().to(device)
    weight = sampler.denied_per_batch / sampler.allowed_per_batch
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "model.pt"
    best = -1.0
    history = []
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"Batches: {len(sampler)}; allowed/not_allowed: "
          f"{sampler.allowed_per_batch}/{sampler.denied_per_batch}; pos_weight: {weight:g}", flush=True)
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "val_loss", "train_accuracy",
                                              "val_accuracy", "train_macro_f1", "val_macro_f1", "threshold"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss, seen = 0.0, 0
            for step, (x, y) in enumerate(train_loader, 1):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y)
                seen += len(y)
                if step % 50 == 0:
                    print(f"Epoch {epoch}/{args.epochs}, batch {step}/{len(train_loader)}", flush=True)
            val_y, val_scores, val_loss = evaluate(model, evaluation_loaders["val"], device, loss_fn)
            threshold = best_f1_threshold(val_y, val_scores)
            val_result = metrics(val_y, val_scores, threshold)
            train_y, train_scores, _ = evaluate(model, evaluation_loaders["train"], device, loss_fn)
            train_result = metrics(train_y, train_scores, threshold)
            row = dict(epoch=epoch, loss=total_loss / seen, val_loss=val_loss,
                       train_accuracy=train_result["accuracy"], val_accuracy=val_result["accuracy"],
                       train_macro_f1=train_result["macro_f1"],
                       val_macro_f1=val_result["macro_f1"], threshold=threshold)
            writer.writerow(row)
            history.append(row)
            f.flush()
            print(f"Epoch {epoch:02}: loss={row['loss']:.4f}, "
                  f"train acc={row['train_accuracy']:.4f}, val acc={row['val_accuracy']:.4f}, "
                  f"train macro-F1={row['train_macro_f1']:.4f}, val macro-F1={row['val_macro_f1']:.4f}", flush=True)
            if row["val_macro_f1"] > best:
                best = row["val_macro_f1"]
                torch.save(dict(model=model.state_dict(), threshold=threshold, epoch=epoch,
                                sample_rate=SR, seconds=SECONDS), checkpoint)
    save_plot(history, args.output / "training.png")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    report = dict(best_epoch=saved["epoch"], threshold=saved["threshold"], splits={})
    for split, loader in evaluation_loaders.items():
        labels, scores, loss = evaluate(model, loader, device, loss_fn)
        result = metrics(labels, scores, saved["threshold"])
        result["loss"] = loss
        report["splits"][split] = result
        print(f"{split}: accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f}", flush=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "settings.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
