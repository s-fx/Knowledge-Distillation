import os
import sys
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_l_16
from torchvision.models.vision_transformer import VisionTransformer

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    top_k_accuracy_score,
    log_loss,
)
from tqdm import tqdm

import config as cfg
from utils import set_seed, make_train_val_split, TemperatureScaler

# Import DeiT model class
from train_deit_distill import DeiTTinyDistilled

# Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
set_seed(cfg.SEED)
torch.backends.cudnn.benchmark = True

# Data
test_transform = transforms.Compose([
    transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg.CIFAR100_MEAN, std=cfg.CIFAR100_STD),
])

# Val-Set (aus Train) für Temperature Scaling – Testset wird nicht angefasst sonst test leakage!!!!
val_base = torchvision.datasets.CIFAR100(cfg.DATA_DIR, train=True, download=True, transform=test_transform)
_, val_set = make_train_val_split(val_base, cfg.VAL_SIZE, seed=cfg.SEED)
val_loader = DataLoader(val_set, batch_size=cfg.EVAL_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

test_dataset = torchvision.datasets.CIFAR100(
    root=cfg.DATA_DIR, train=False, download=True, transform=test_transform
)
test_loader = DataLoader(
    test_dataset, batch_size=cfg.EVAL_BATCH_SIZE,
    shuffle=False, num_workers=4, pin_memory=True
)

# Model Loading Functions
def load_vit_large():
    model = vit_l_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, cfg.NUM_CLASSES)
    model.load_state_dict(torch.load(cfg.VIT_L_PATH, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    return model


def load_vit_tiny():
    model = VisionTransformer(
        image_size=cfg.IMAGE_SIZE,
        patch_size=cfg.VIT_T_PATCH_SIZE,
        num_layers=cfg.VIT_T_NUM_LAYERS,
        num_heads=cfg.VIT_T_NUM_HEADS,
        hidden_dim=cfg.VIT_T_HIDDEN_DIM,
        mlp_dim=cfg.VIT_T_MLP_DIM,
        num_classes=cfg.NUM_CLASSES,
        dropout=cfg.VIT_T_DROPOUT,
        attention_dropout=cfg.VIT_T_DROPOUT,
    )
    model.load_state_dict(torch.load(cfg.VIT_T_PATH, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    return model


def load_vit_tiny_kd():
    model = VisionTransformer(
        image_size=cfg.IMAGE_SIZE,
        patch_size=cfg.VIT_T_PATCH_SIZE,
        num_layers=cfg.VIT_T_NUM_LAYERS,
        num_heads=cfg.VIT_T_NUM_HEADS,
        hidden_dim=cfg.VIT_T_HIDDEN_DIM,
        mlp_dim=cfg.VIT_T_MLP_DIM,
        num_classes=cfg.NUM_CLASSES,
        dropout=cfg.VIT_T_DROPOUT,
        attention_dropout=cfg.VIT_T_DROPOUT,
    )
    model.load_state_dict(torch.load(cfg.VIT_T_KD_PATH, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    return model


def load_deit_tiny():
    model = DeiTTinyDistilled(
        image_size=cfg.IMAGE_SIZE,
        patch_size=cfg.VIT_T_PATCH_SIZE,
        num_layers=cfg.VIT_T_NUM_LAYERS,
        num_heads=cfg.VIT_T_NUM_HEADS,
        hidden_dim=cfg.VIT_T_HIDDEN_DIM,
        mlp_dim=cfg.VIT_T_MLP_DIM,
        num_classes=cfg.NUM_CLASSES,
        dropout=cfg.VIT_T_DROPOUT,
    )
    model.load_state_dict(torch.load(cfg.VIT_T_DEIT_PATH, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    return model

# NEW
@torch.no_grad()
def collect_logits(model, loader, is_deit=False):
    model.eval()
    logits_list, labels_list = [], []
    for images, labels in tqdm(loader, desc="  logits"):
        images = images.to(DEVICE)
        out = model.forward_inference(images) if is_deit else model(images)
        logits_list.append(out.float().cpu())        # fp32
        labels_list.append(labels)
    return torch.cat(logits_list), torch.cat(labels_list)

# NEW
def metrics_from_logits(logits, labels, scaler=None):
    if scaler is not None:
        with torch.no_grad():
            logits = scaler(logits.to(next(scaler.parameters()).device)).cpu()

    probs = F.softmax(logits, dim=1).numpy()
    preds = logits.argmax(1).numpy()
    tgts = labels.numpy()

    top1 = 100.0 * np.mean(preds == tgts)
    top5 = top_k_accuracy_score(tgts, probs, k=5, labels=np.arange(cfg.NUM_CLASSES)) * 100
    precision = precision_score(tgts, preds, average="macro", zero_division=0)
    recall = recall_score(tgts, preds, average="macro", zero_division=0)
    f1 = f1_score(tgts, preds, average="macro", zero_division=0)
    ce = float(F.cross_entropy(logits, labels).item())   # == NLL, jetzt konsistent (fp32)
    nll = log_loss(tgts, probs, labels=np.arange(cfg.NUM_CLASSES))
    ece = compute_ece(probs, tgts)

    return {
        "top1_accuracy": round(top1, 2), "top5_accuracy": round(top5, 2),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "cross_entropy_loss": round(ce, 4), "nll": round(float(nll), 4),
        "ece": round(float(ece), 4), "predictions": preds,
    }

# Metrics Functions
def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            ece += (np.sum(mask) / len(labels)) * abs(bin_acc - bin_conf)

    return float(ece)


# wieviele trainable und gesamte params
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 ** 2)
    return 0.0



# Benchmark Inference Speed
def benchmark_speed(model, model_name, is_deit=False):
    model.eval()
    single = torch.randn(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, device=DEVICE)
    batch  = torch.randn(cfg.EVAL_BATCH_SIZE, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, device=DEVICE)
    fwd = (lambda x: model.forward_inference(x)) if is_deit else (lambda x: model(x))

    for _ in range(cfg.NUM_WARMUP_RUNS):
        with torch.no_grad(): fwd(single)
    torch.cuda.synchronize()

    starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)
    starter.record()
    for _ in range(cfg.NUM_BENCHMARK_RUNS):
        with torch.no_grad(): fwd(single)
    ender.record(); torch.cuda.synchronize()
    ms = starter.elapsed_time(ender) / cfg.NUM_BENCHMARK_RUNS

    for _ in range(cfg.NUM_WARMUP_RUNS):
        with torch.no_grad(): fwd(batch)
    torch.cuda.synchronize()
    starter.record()
    for _ in range(cfg.NUM_BENCHMARK_RUNS):
        with torch.no_grad(): fwd(batch)
    ender.record(); torch.cuda.synchronize()
    total_s = starter.elapsed_time(ender) / 1000.0
    throughput = (cfg.NUM_BENCHMARK_RUNS * cfg.EVAL_BATCH_SIZE) / total_s

    return {"latency_ms": round(ms, 2), "fps": round(1000.0/ms, 1),
            "throughput_img_per_sec": round(throughput, 1)}



# Teacher-Student Agreement
def compute_agreement(teacher_preds, student_preds):
    return round(float(np.mean(teacher_preds == student_preds)) * 100, 2)


def print_comparison_table(all_results):
    print(f"\n{'='*100}")
    print(f"{'FINAL COMPARISON':^100}")
    print(f"{'='*100}")

    header = f"{'Metric':<28}"
    for name in all_results:
        header += f"| {name:<16}"
    print(header)
    print("─" * len(header))

    metrics = [
        ("Top-1 Acc (%)", "top1_accuracy"),
        ("Top-5 Acc (%)", "top5_accuracy"),
        ("Precision (macro)", "precision_macro"),
        ("Recall (macro)", "recall_macro"),
        ("F1 (macro)", "f1_macro"),
        ("CE Loss", "cross_entropy_loss"),
        ("NLL", "nll"),
        ("ECE", "ece"),
        ("Parameters", "parameters"),
        ("FLOPs (GMACs)", "flops_gmacs"),
        ("Model Size (MB)", "model_size_mb"),
        ("Latency (ms)", "latency_ms"),
        ("FPS", "fps"),
        ("Throughput (img/s)", "throughput_img_per_sec"),
        ("Peak GPU Mem (MB)", "peak_gpu_memory_mb"),
        ("Teacher Agreement (%)", "teacher_agreement"),
        ("Temperature", "temperature"),
        ("ECE after TS", "ece_after_ts"),
        ("NLL after TS", "nll_after_ts"),
    ]

    for display_name, key in metrics:
        row = f"{display_name:<28}"
        for name in all_results:
            val = all_results[name].get(key, "—")
            if val is None:
                val = "—"
            elif isinstance(val, float):
                if val > 1000:
                    val = f"{val:,.0f}"
                else:
                    val = f"{val}"
            elif isinstance(val, int):
                val = f"{val:,}"
            row += f"| {str(val):<16}"
        print(row)

    print("─" * len(header))


if __name__ == "__main__":

    all_results = {}

    models_config = [
        {
            "name": "ViT-L (Teacher)",
            "loader": load_vit_large,
            "path": cfg.VIT_L_PATH,
            "is_deit": False,
        },
        {
            "name": "ViT-T (Baseline)",
            "loader": load_vit_tiny,
            "path": cfg.VIT_T_PATH,
            "is_deit": False,
        },
        {
            "name": "ViT-T (KD)",
            "loader": load_vit_tiny_kd,
            "path": cfg.VIT_T_KD_PATH,
            "is_deit": False,
        },
        {
            "name": "DeiT-T (Distill)",
            "loader": load_deit_tiny,
            "path": cfg.VIT_T_DEIT_PATH,
            "is_deit": True,
        },
    ]

    teacher_preds = None

    for model_cfg in models_config:
        name = model_cfg["name"]
        path = model_cfg["path"]

        # Check if checkpoint exists
        if not os.path.exists(path):
            print(f"\nCheckpoint not found: {path}")
            print(f"Skipping {name}. Train the model first.")
            continue

        # Load model
        model = model_cfg["loader"]()
        val_logits, val_labels   = collect_logits(model, val_loader,  model_cfg["is_deit"])
        test_logits, test_labels = collect_logits(model, test_loader, model_cfg["is_deit"])

        scaler = TemperatureScaler()
        T_opt = scaler.fit(val_logits.to(DEVICE), val_labels.to(DEVICE))

        raw = metrics_from_logits(test_logits, test_labels, scaler=None)
        cal = metrics_from_logits(test_logits, test_labels, scaler=scaler)
        preds = raw.pop("predictions"); cal.pop("predictions")
        raw["temperature"]  = round(T_opt, 3)
        raw["ece_after_ts"] = cal["ece"]
        raw["nll_after_ts"] = cal["nll"]

        total_params, _ = count_parameters(model)
        flops = compute_flops(model, is_deit=model_cfg["is_deit"])
        model_size = get_model_size_mb(path)
        speed_results = benchmark_speed(model, name, is_deit=model_cfg["is_deit"])

        if "Teacher" in name:
            teacher_preds = preds; agreement = 100.0
        else:
            agreement = compute_agreement(teacher_preds, preds) if teacher_preds is not None else None

        all_results[name] = {
            **raw,
            "parameters": total_params,
            "flops_gmacs": round(flops / 1e9, 2) if flops else None,
            "model_size_mb": round(model_size, 1),
            **speed_results,
            "teacher_agreement": agreement,
        }

        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
            _ = collect_logits(model, test_loader, model_cfg["is_deit"])
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
            all_results[name]["peak_gpu_memory_mb"] = round(peak_mb, 1)

        # Free GPU memory
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Print final comparison
    if all_results:
        print_comparison_table(all_results)

        # Save to JSON
        output_path = os.path.join(cfg.RESULTS_DIR, "comparison_results.json")
        json_results = {}
        for name, res in all_results.items():
            json_results[name] = {
                k: v for k, v in res.items()
                if not isinstance(v, np.ndarray)
            }

        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    else:
        print("\nNo models were evaluated. Train at least one model first.")

    print("\nDone.")
