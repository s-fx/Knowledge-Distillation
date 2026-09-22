"""
vit tiny on cifar from scratch is the baseline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.models.vision_transformer import VisionTransformer

from tqdm import tqdm

import config as cfg
from utils import set_seed, make_train_val_split


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
set_seed(cfg.SEED)


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
    transforms.Normalize(mean=cfg.CIFAR100_MEAN, std=cfg.CIFAR100_STD),
])


full_train = torchvision.datasets.CIFAR100(
    root=cfg.DATA_DIR, train=True, download=True, transform=train_transform
)

val_base = torchvision.datasets.CIFAR100(
    root=cfg.DATA_DIR, train=True, download=True, transform=test_transform
)
test_dataset = torchvision.datasets.CIFAR100(
    root=cfg.DATA_DIR, train=False, download=True, transform=test_transform
)

train_set, _ = make_train_val_split(full_train, cfg.VAL_SIZE, seed=cfg.SEED)
_, val_set   = make_train_val_split(val_base,   cfg.VAL_SIZE, seed=cfg.SEED)

train_loader = DataLoader(train_set, batch_size=cfg.VIT_T_BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=cfg.VIT_T_BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)



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

model = model.to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())
print(f"ViT-Tiny Parameters: {num_params:,}")

criterion = nn.CrossEntropyLoss(label_smoothing=cfg.VIT_T_LABEL_SMOOTHING)
eval_criterion = nn.CrossEntropyLoss()   # plain NLL, comparable across all scripts

optimizer = optim.AdamW(
    model.parameters(),
    lr=cfg.VIT_T_LR,
    weight_decay=cfg.VIT_T_WEIGHT_DECAY
)

# Linear warmup + cosine decay
def lr_lambda(epoch):
    if epoch < cfg.VIT_T_WARMUP_EPOCHS:
        return (epoch + 1) / cfg.VIT_T_WARMUP_EPOCHS
    else:
        import math
        progress = (epoch - cfg.VIT_T_WARMUP_EPOCHS) / (cfg.VIT_T_EPOCHS - cfg.VIT_T_WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

def train_one_epoch(model, loader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(
            loss=f"{running_loss/total:.4f}",
            acc=f"{100.*correct/total:.2f}%"
        )

    return running_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = eval_criterion(outputs, labels)
        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
    return running_loss / total, 100. * correct / total

if __name__ == "__main__":
    best_acc = 0.0

    print(f"\n{'='*60}")
    print(f"Training ViT-Tiny on CIFAR-100 (from scratch)")
    print(f"Epochs: {cfg.VIT_T_EPOCHS} | LR: {cfg.VIT_T_LR}")
    print(f"Warmup: {cfg.VIT_T_WARMUP_EPOCHS} epochs")
    print(f"{'='*60}\n")

    for epoch in range(cfg.VIT_T_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{cfg.VIT_T_EPOCHS}] | LR: {scheduler.get_last_lr()[0]:.6f}")

        train_loss, train_acc = train_one_epoch(model, train_loader)
        val_loss, val_acc = evaluate(model, val_loader)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg.VIT_T_PATH)
            print(f"Best model saved! ({best_acc:.2f}%)")

    print(f"\n{'='*60}")
    print(f"Training Complete! Best Val Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {cfg.VIT_T_PATH}")
    print(f"{'='*60}")
