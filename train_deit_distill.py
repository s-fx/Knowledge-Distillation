"""
Train DeiT Tiny on cifar100 like in the papaer. Teacher is vit-l finetuned on imagenet... (was a bad
idea...)
"""

import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_l_16
from torchvision.models.vision_transformer import VisionTransformer

from tqdm import tqdm

import config as cfg
from utils import set_seed, make_train_val_split


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Model: DeiT-Tiny with distillation token
class DeiTTinyDistilled(VisionTransformer):
    """
    torchvision VisionTransformer mit dem distill token!

    Token layout: [CLS] [DIST] [patch_1] ... [patch_N]
      - CLS -> head_cls  : trained with CE on ground-truth labels
      - DIST-> head_dist : trained against the teacher (hard or soft)
      - inference: mean of both heads
    """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=3,
        hidden_dim=192,
        mlp_dim=768,
        num_classes=100,
        dropout=0.1,
        attention_dropout=0.1,
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,
        )

        num_patches = (image_size // patch_size) ** 2

        # distillation token 
        self.dist_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.dist_token, std=0.02)

        # pos embedd musst du einen slot einfuegen fuer [DIST] Toklen
        old_pos = self.encoder.pos_embedding.data          # (1, N+1, D)
        new_pos = torch.empty(1, num_patches + 2, hidden_dim).normal_(std=0.02)
        new_pos[:, 0:1] = old_pos[:, 0:1]                  # CLS position
        new_pos[:, 2:] = old_pos[:, 1:]                    # patch positions
        self.encoder.pos_embedding = nn.Parameter(new_pos)
        self.seq_length = num_patches + 2

        # two heads und dann mean nehmen!
        self.heads = nn.Identity()
        self.head_cls = nn.Linear(hidden_dim, num_classes)
        self.head_dist = nn.Linear(hidden_dim, num_classes)
        for head in (self.head_cls, self.head_dist):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward_heads(self, x):
        """Returns (cls_logits, dist_logits)."""
        x = self._process_input(x)                         # (B, N, D)
        b = x.shape[0]
        cls_tokens = self.class_token.expand(b, -1, -1)
        dist_tokens = self.dist_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        x = self.encoder(x) 
        return self.head_cls(x[:, 0]), self.head_dist(x[:, 1])

    def forward(self, x):
        cls_logits, dist_logits = self.forward_heads(x)
        if self.training:
            return cls_logits, dist_logits
        return (cls_logits + dist_logits) / 2.0

    @torch.no_grad()
    def forward_inference(self, x):
        cls_logits, dist_logits = self.forward_heads(x)
        return (cls_logits + dist_logits) / 2.0


def build_dataloaders():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.CIFAR100_MEAN, cfg.CIFAR100_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.CIFAR100_MEAN, cfg.CIFAR100_STD),
    ])

    full_train = torchvision.datasets.CIFAR100(
        root=cfg.DATA_DIR, train=True, download=True, transform=train_transform
    )
    val_base = torchvision.datasets.CIFAR100(
        root=cfg.DATA_DIR, train=True, download=True, transform=test_transform
    )

    train_set, _ = make_train_val_split(full_train, cfg.VAL_SIZE, seed=cfg.SEED)
    _, val_set = make_train_val_split(val_base, cfg.VAL_SIZE, seed=cfg.SEED)  # same indices

    train_loader = DataLoader(
        train_set, batch_size=cfg.DEIT_BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.DEIT_BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    return train_loader, val_loader


def load_teacher(device):
    print("Loading teacher model (ViT-Large)...")
    teacher = vit_l_16(weights=None)
    teacher.heads.head = nn.Linear(teacher.heads.head.in_features, cfg.NUM_CLASSES)
    teacher.load_state_dict(
        torch.load(cfg.VIT_L_PATH, map_location=device, weights_only=True)
    )
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("Teacher loaded and frozen.")
    return teacher


def build_student(device):
    student = DeiTTinyDistilled(
        image_size=cfg.IMAGE_SIZE,
        patch_size=cfg.VIT_T_PATCH_SIZE,
        num_layers=cfg.VIT_T_NUM_LAYERS,
        num_heads=cfg.VIT_T_NUM_HEADS,
        hidden_dim=cfg.VIT_T_HIDDEN_DIM,
        mlp_dim=cfg.VIT_T_MLP_DIM,
        num_classes=cfg.NUM_CLASSES,
        dropout=cfg.VIT_T_DROPOUT,
        attention_dropout=cfg.VIT_T_DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in student.parameters())
    print(f"DeiT-Tiny (distilled) Parameters: {n_params:,}")
    return student


# DeiT loss
def deit_loss(cls_logits, dist_logits, teacher_logits, labels, T, alpha, hard=True):
    """
    CLS head : CE against ground truth (with label smoothing)
    DIST head: hard  -> CE against argmax(teacher)   (so wie im paper!)
               soft  -> KL against temperature-scaled teacher
    """
    ce_loss = F.cross_entropy(
        cls_logits, labels, label_smoothing=cfg.DEIT_LABEL_SMOOTHING
    )

    if hard:
        teacher_labels = teacher_logits.argmax(dim=1)
        dist_loss = F.cross_entropy(dist_logits, teacher_labels)
    else:
        dist_log_p = F.log_softmax(dist_logits / T, dim=1)
        teacher_p = F.softmax(teacher_logits / T, dim=1)
        dist_loss = F.kl_div(dist_log_p, teacher_p, reduction="batchmean") * (T * T)

    return (1.0 - alpha) * ce_loss + alpha * dist_loss


def train_one_epoch(student, teacher, loader, optimizer, scaler, device):
    student.train()
    running_loss = 0.0
    correct = total = 0

    amp = (device == "cuda")
    pbar = tqdm(loader, desc="Training (DeiT)")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp):
            with torch.no_grad():
                teacher_logits = teacher(images)
            cls_logits, dist_logits = student(images)
            loss = deit_loss(
                cls_logits, dist_logits, teacher_logits, labels,
                T=cfg.DEIT_TEMPERATURE, alpha=cfg.DEIT_ALPHA,
                hard=cfg.DEIT_HARD_DISTILL,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        avg_logits = (cls_logits.detach().float() + dist_logits.detach().float()) / 2.0
        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        correct += avg_logits.argmax(1).eq(labels).sum().item()

        pbar.set_postfix(
            loss=f"{running_loss/total:.4f}",
            acc=f"{100.*correct/total:.2f}%",
        )

    return running_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct = total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model.forward_inference(images)          # fp32, averaged heads
        loss = F.cross_entropy(outputs, labels)
        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
    return running_loss / total, 100. * correct / total


def main():
    print(f"Device: {DEVICE}")
    set_seed(cfg.SEED)

    train_loader, val_loader = build_dataloaders()
    teacher = load_teacher(DEVICE)
    student = build_student(DEVICE)

    optimizer = optim.AdamW(
        student.parameters(), lr=cfg.DEIT_LR, weight_decay=cfg.DEIT_WEIGHT_DECAY
    )

    def lr_lambda(epoch):
        if epoch < cfg.DEIT_WARMUP_EPOCHS:
            return (epoch + 1) / cfg.DEIT_WARMUP_EPOCHS
        progress = ((epoch - cfg.DEIT_WARMUP_EPOCHS)
                    / max(1, cfg.DEIT_EPOCHS - cfg.DEIT_WARMUP_EPOCHS))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    best_acc = 0.0
    mode = "hard" if cfg.DEIT_HARD_DISTILL else f"soft (T={cfg.DEIT_TEMPERATURE})"

    print(f"\n{'='*60}")
    print("DeiT-Style Distillation")
    print("Teacher: ViT-Large | Student: DeiT-Tiny (torchvision backbone)")
    print(f"Epochs: {cfg.DEIT_EPOCHS} | distill: {mode} | alpha: {cfg.DEIT_ALPHA}")
    print(f"{'='*60}\n")

    for epoch in range(cfg.DEIT_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{cfg.DEIT_EPOCHS}] | LR: {scheduler.get_last_lr()[0]:.6f}")

        train_loss, train_acc = train_one_epoch(
            student, teacher, train_loader, optimizer, scaler, DEVICE
        )
        val_loss, val_acc = evaluate(student, val_loader, DEVICE)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val NLL:  {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), cfg.VIT_T_DEIT_PATH)
            print(f"Best model saved! ({best_acc:.2f}%)")

    print(f"\n{'='*60}")
    print(f"Training Complete! Best Val Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {cfg.VIT_T_DEIT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
