from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


"""Colab-friendly training script for the self-pruning CIFAR-10 case study.

Recommended Colab command:

python self_pruning_cifar10_colab.py --epochs 3 --lambdas 0 1e-4 2e-4 5e-4
"""


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_lambda(value: float) -> str:
    return f"{value:.0e}" if value != 0 else "0"


def resolve_device(requested_device: str) -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def maybe_subset(dataset: Dataset, subset_size: int | None, seed: int) -> Dataset:
    if subset_size is None or subset_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_subset: int | None = None,
    test_subset: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    try:
        train_dataset = datasets.CIFAR10(
            root=str(data_dir),
            train=True,
            transform=train_transform,
            download=True,
        )
        test_dataset = datasets.CIFAR10(
            root=str(data_dir),
            train=False,
            transform=test_transform,
            download=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to load CIFAR-10. Download the dataset or rerun the script with network access."
        ) from exc

    train_dataset = maybe_subset(train_dataset, train_subset, seed)
    test_dataset = maybe_subset(test_dataset, test_subset, seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


class PrunableLinear(nn.Module):
    """Linear layer whose weights are modulated by learnable sigmoid gates."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gate_init_prob: float = 0.9,
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 < gate_init_prob < 1.0:
            raise ValueError("gate_init_prob must be in (0, 1).")
        if gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be positive.")

        self.in_features = in_features
        self.out_features = out_features
        self.gate_temperature = gate_temperature
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        initial_gate_score = math.log(gate_init_prob / (1.0 - gate_init_prob))
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), initial_gate_score)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def gates(self) -> Tensor:
        # Lower temperatures make the sigmoid steeper, which helps gates
        # move closer to a true binary pruning decision during training.
        return torch.sigmoid(self.gate_scores / self.gate_temperature)

    def l1_gate_penalty(self) -> Tensor:
        return self.gates().sum()

    def forward(self, inputs: Tensor, hard_threshold: float | None = None) -> Tensor:
        gates = self.gates()
        if hard_threshold is not None:
            gates = (gates >= hard_threshold).to(dtype=inputs.dtype)
        effective_weight = self.weight * gates
        return F.linear(inputs, effective_weight, self.bias)


class SelfPruningMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 32 * 32 * 3,
        hidden_dims: Sequence[int] = (512, 256, 128),
        num_classes: int = 10,
        dropout: float = 0.2,
        gate_init_prob: float = 0.9,
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, num_classes]
        self.layers = nn.ModuleList()

        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            self.layers.append(
                PrunableLinear(
                    in_features=in_dim,
                    out_features=out_dim,
                    gate_init_prob=gate_init_prob,
                    gate_temperature=gate_temperature,
                )
            )
            is_last_layer = index == len(dims) - 2
            if not is_last_layer:
                self.layers.append(nn.BatchNorm1d(out_dim))
                self.layers.append(nn.GELU())
                self.layers.append(nn.Dropout(dropout))

    def prunable_layers(self) -> list[PrunableLinear]:
        return [module for module in self.layers if isinstance(module, PrunableLinear)]

    def forward(self, images: Tensor, hard_threshold: float | None = None) -> Tensor:
        x = torch.flatten(images, start_dim=1)
        for module in self.layers:
            if isinstance(module, PrunableLinear):
                x = module(x, hard_threshold=hard_threshold)
            else:
                x = module(x)
        return x

    def sparsity_loss(self) -> Tensor:
        return torch.stack([layer.l1_gate_penalty() for layer in self.prunable_layers()]).sum()

    @torch.no_grad()
    def gate_values(self) -> Tensor:
        return torch.cat([layer.gates().flatten() for layer in self.prunable_layers()])

    @torch.no_grad()
    def sparsity_stats(self, threshold: float) -> dict[str, float | int]:
        gate_values = self.gate_values()
        total = gate_values.numel()
        below_threshold = (gate_values < threshold).sum().item()
        return {
            "threshold": threshold,
            "total_gates": total,
            "pruned_gates": below_threshold,
            "active_gates": total - below_threshold,
            "sparsity_level": 100.0 * below_threshold / total,
            "mean_gate": gate_values.mean().item(),
            "median_gate": gate_values.median().item(),
        }


@dataclass
class EpochMetrics:
    epoch: int
    split: str
    total_loss: float
    classification_loss: float
    sparsity_loss: float
    accuracy: float
    sparsity_level: float
    mean_gate: float


@dataclass
class ExperimentResult:
    lambda_sparse: float
    best_epoch: int
    best_hard_test_accuracy: float
    soft_test_accuracy: float
    hard_test_accuracy: float
    sparsity_level: float
    mean_gate: float
    pruned_gates: int
    total_gates: int
    checkpoint_path: str
    history_path: str
    gate_histogram_path: str


def run_epoch(
    model: SelfPruningMLP,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    lambda_sparse: float,
    optimizer: torch.optim.Optimizer | None,
    sparsity_threshold: float,
) -> EpochMetrics:
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_examples = 0
    correct_predictions = 0
    summed_total_loss = 0.0
    summed_cls_loss = 0.0
    summed_sparse_loss = 0.0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        classification_loss = criterion(logits, labels)
        sparsity_loss = model.sparsity_loss()
        total_loss = classification_loss + lambda_sparse * sparsity_loss

        if is_training:
            total_loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_examples += batch_size
        correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
        summed_total_loss += total_loss.item() * batch_size
        summed_cls_loss += classification_loss.item() * batch_size
        summed_sparse_loss += sparsity_loss.item() * batch_size

    stats = model.sparsity_stats(threshold=sparsity_threshold)
    return EpochMetrics(
        epoch=-1,
        split="train" if is_training else "eval",
        total_loss=summed_total_loss / total_examples,
        classification_loss=summed_cls_loss / total_examples,
        sparsity_loss=summed_sparse_loss / total_examples,
        accuracy=100.0 * correct_predictions / total_examples,
        sparsity_level=float(stats["sparsity_level"]),
        mean_gate=float(stats["mean_gate"]),
    )


@torch.no_grad()
def evaluate_hard_pruned_accuracy(
    model: SelfPruningMLP,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> float:
    model.eval()
    total_examples = 0
    correct_predictions = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images, hard_threshold=threshold)
        total_examples += labels.size(0)
        correct_predictions += (logits.argmax(dim=1) == labels).sum().item()

    return 100.0 * correct_predictions / total_examples


def save_gate_histogram(gate_values: Tensor, threshold: float, output_path: Path) -> None:
    values = gate_values.cpu().numpy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(values, bins=60, color="#19647E", edgecolor="white")
    axes[0].axvline(
        threshold,
        color="#C1121F",
        linestyle="--",
        linewidth=2,
        label=f"threshold={threshold}",
    )
    axes[0].set_title("Full Range")
    axes[0].set_xlabel("Gate value")
    axes[0].set_ylabel("Count")
    axes[0].set_yscale("log")
    axes[0].legend()

    axes[1].hist(values, bins=60, range=(0.0, 0.1), color="#1F7A8C", edgecolor="white")
    axes[1].axvline(
        threshold,
        color="#C1121F",
        linestyle="--",
        linewidth=2,
        label=f"threshold={threshold}",
    )
    axes[1].set_title("Zoomed Near Zero")
    axes[1].set_xlabel("Gate value")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    figure.suptitle("Final Gate Value Distribution", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_history(history: list[EpochMetrics], output_path: Path) -> None:
    fieldnames = list(asdict(history[0]).keys())
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in history:
            writer.writerow(asdict(entry))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_for_lambda(
    args: argparse.Namespace,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    lambda_sparse: float,
) -> ExperimentResult:
    model = SelfPruningMLP(
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        gate_init_prob=args.gate_init_prob,
        gate_temperature=args.gate_temperature,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    base_parameters: list[nn.Parameter] = []
    gate_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if name.endswith("gate_scores"):
            gate_parameters.append(parameter)
        else:
            base_parameters.append(parameter)

    optimizer = torch.optim.Adam(
        [
            {
                "params": base_parameters,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": gate_parameters,
                "lr": args.learning_rate * args.gate_lr_multiplier,
                "weight_decay": 0.0,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    lambda_tag = format_lambda(lambda_sparse)
    checkpoint_path = args.output_dir / f"model_lambda_{lambda_tag}.pt"
    history_path = args.output_dir / f"history_lambda_{lambda_tag}.csv"
    histogram_path = args.output_dir / f"gate_hist_lambda_{lambda_tag}.png"

    history: list[EpochMetrics] = []
    best_epoch = 1
    best_hard_accuracy = -1.0

    print(
        f"\n[lambda={lambda_sparse}] parameters={parameter_count(model):,} "
        f"device={device.type} gate_lr={args.learning_rate * args.gate_lr_multiplier:g}"
    )

    for epoch in range(1, args.epochs + 1):
        started_at = time.time()

        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            lambda_sparse=lambda_sparse,
            optimizer=optimizer,
            sparsity_threshold=args.sparsity_threshold,
        )
        train_metrics.epoch = epoch
        train_metrics.split = "train"

        eval_metrics = run_epoch(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            lambda_sparse=lambda_sparse,
            optimizer=None,
            sparsity_threshold=args.sparsity_threshold,
        )
        eval_metrics.epoch = epoch
        eval_metrics.split = "test_soft"

        hard_accuracy = evaluate_hard_pruned_accuracy(
            model=model,
            data_loader=test_loader,
            device=device,
            threshold=args.sparsity_threshold,
        )
        history.extend([train_metrics, eval_metrics])

        if hard_accuracy > best_hard_accuracy:
            best_hard_accuracy = hard_accuracy
            best_epoch = epoch

        elapsed = time.time() - started_at
        print(
            "  "
            f"epoch={epoch:02d}/{args.epochs} "
            f"train_acc={train_metrics.accuracy:6.2f}% "
            f"soft_test_acc={eval_metrics.accuracy:6.2f}% "
            f"hard_test_acc={hard_accuracy:6.2f}% "
            f"sparsity={eval_metrics.sparsity_level:6.2f}% "
            f"mean_gate={eval_metrics.mean_gate:0.4f} "
            f"time={elapsed:0.1f}s"
        )
        scheduler.step()

    final_stats = model.sparsity_stats(threshold=args.sparsity_threshold)
    soft_test_metrics = run_epoch(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        lambda_sparse=lambda_sparse,
        optimizer=None,
        sparsity_threshold=args.sparsity_threshold,
    )
    soft_test_metrics.epoch = best_epoch
    soft_test_metrics.split = "test_soft"
    hard_test_accuracy = evaluate_hard_pruned_accuracy(
        model=model,
        data_loader=test_loader,
        device=device,
        threshold=args.sparsity_threshold,
    )

    save_history(history, history_path)
    gate_values = model.gate_values()
    save_gate_histogram(gate_values=gate_values, threshold=args.sparsity_threshold, output_path=histogram_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dims": list(args.hidden_dims),
            "dropout": args.dropout,
            "gate_init_prob": args.gate_init_prob,
            "gate_lr_multiplier": args.gate_lr_multiplier,
            "gate_temperature": args.gate_temperature,
            "lambda_sparse": lambda_sparse,
            "sparsity_threshold": args.sparsity_threshold,
            "best_epoch": best_epoch,
            "best_hard_test_accuracy": best_hard_accuracy,
            "soft_test_accuracy": soft_test_metrics.accuracy,
            "hard_test_accuracy": hard_test_accuracy,
            "sparsity_level": final_stats["sparsity_level"],
        },
        checkpoint_path,
    )

    return ExperimentResult(
        lambda_sparse=lambda_sparse,
        best_epoch=best_epoch,
        best_hard_test_accuracy=best_hard_accuracy,
        soft_test_accuracy=soft_test_metrics.accuracy,
        hard_test_accuracy=hard_test_accuracy,
        sparsity_level=float(final_stats["sparsity_level"]),
        mean_gate=float(final_stats["mean_gate"]),
        pruned_gates=int(final_stats["pruned_gates"]),
        total_gates=int(final_stats["total_gates"]),
        checkpoint_path=str(checkpoint_path),
        history_path=str(history_path),
        gate_histogram_path=str(histogram_path),
    )


def write_summary(results: list[ExperimentResult], output_dir: Path) -> None:
    summary_json = output_dir / "experiment_results.json"
    summary_csv = output_dir / "experiment_results.csv"

    payload = [asdict(result) for result in results]
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(payload[0].keys()))
        writer.writeheader()
        for row in payload:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a self-pruning MLP with learnable gates on CIFAR-10."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/content/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("/content/self_pruning_outputs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gate-lr-multiplier", type=float, default=100.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--gate-init-prob", type=float, default=0.7)
    parser.add_argument("--gate-temperature", type=float, default=0.5)
    parser.add_argument("--sparsity-threshold", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1e-4, 2e-4, 5e-4])
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Ignoring notebook arguments: {unknown_args}")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    torch.set_num_threads(max(1, min(os.cpu_count() or 1, 16)))
    device = resolve_device(args.device)

    print(
        f"Preparing CIFAR-10 loaders with batch_size={args.batch_size}, "
        f"num_workers={args.num_workers}, device={device.type}"
    )
    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
    )

    results: list[ExperimentResult] = []
    for lambda_sparse in args.lambdas:
        results.append(
            train_for_lambda(
                args=args,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                lambda_sparse=lambda_sparse,
            )
        )

    write_summary(results, args.output_dir)

    print("\nFinal summary")
    for result in results:
        print(
            "  "
            f"lambda={result.lambda_sparse:<8g} "
            f"best_hard_acc={result.best_hard_test_accuracy:6.2f}%@{result.best_epoch:02d} "
            f"soft_test_acc={result.soft_test_accuracy:6.2f}% "
            f"hard_test_acc={result.hard_test_accuracy:6.2f}% "
            f"sparsity={result.sparsity_level:6.2f}% "
            f"pruned={result.pruned_gates:,}/{result.total_gates:,}"
        )


if __name__ == "__main__":
    main()
