"""
train teacher vit l on cifar
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_l_16, ViT_L_16_Weights

from tqdm import tqdm

import config as cfg
from utils import set_seed, make_train_val_split


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
set_seed(cfg.SEED)


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
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
_, val_set   = make_train_val_split(val_base,   cfg.VAL_SIZE, seed=cfg.SEED)  # gleiche Indizes!

train_loader = DataLoader(train_set, batch_size=cfg.VIT_L_BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=cfg.VIT_L_BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)


print("Loading ViT-L/16 pretrained on ImageNet...")
model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)

# replace head with new for cifar
model.heads.head = nn.Linear(model.heads.head.in_features, cfg.NUM_CLASSES)

model = model.to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")


criterion = nn.CrossEntropyLoss(label_smoothing=cfg.VIT_L_LABEL_SMOOTHING)
eval_criterion = nn.CrossEntropyLoss()


optimizer = optim.AdamW(
    model.parameters(),
    lr=cfg.VIT_L_LR,
    weight_decay=cfg.VIT_L_WEIGHT_DECAY
)

steps_per_epoch = math.ceil(len(train_loader) / cfg.VIT_L_ACCUM_STEP)
total_steps = steps_per_epoch * cfg.VIT_L_EPOCHS

def lr_lambda(step):
    if step < cfg.VIT_L_WARMUP_STEPS:
        return (step + 1) / cfg.VIT_L_WARMUP_STEPS
    progress = ((step - cfg.VIT_L_WARMUP_STEPS)
                / max(1, total_steps - cfg.VIT_L_WARMUP_STEPS))
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

def train_one_epoch(model, loader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    accum = cfg.VIT_L_ACCUM_STEP
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Training")
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels) / accum

        scaler.scale(loss).backward()

        is_last = (i + 1) == len(loader)
        if (i + 1) % accum == 0 or is_last:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        running_loss += loss.item() * labels.size(0) * accum
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
    print(f"Training ViT-Large/16 on CIFAR-100")
    print(f"Epochs: {cfg.VIT_L_EPOCHS} | LR: {cfg.VIT_L_LR}")
    print(f"{'='*60}\n")

    for epoch in range(cfg.VIT_L_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{cfg.VIT_L_EPOCHS}] | LR: {scheduler.get_last_lr()[0]:.6f}")

        train_loss, train_acc = train_one_epoch(model, train_loader)
        val_loss, val_acc = evaluate(model, val_loader)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg.VIT_L_PATH)
            print(f"Best model saved! ({best_acc:.2f}%)")

    print(f"\n{'='*60}")
    print(f"Training Complete! Best Val Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {cfg.VIT_L_PATH}")
    print(f"{'='*60}")
