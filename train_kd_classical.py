"""
Train vit tiny with classical KD
Teacher vit large , Student: vit tiny.

Loss = alpha * KL(student_soft || teacher_soft) * T^2 + (1-alpha) * CE(student, labels)
hope is correct haha
"""

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
# Val braucht Test-Transform (keine Augmentation):
val_base = torchvision.datasets.CIFAR100(
    root=cfg.DATA_DIR, train=True, download=True, transform=test_transform
)
test_dataset = torchvision.datasets.CIFAR100(
    root=cfg.DATA_DIR, train=False, download=True, transform=test_transform
)

train_set, _ = make_train_val_split(full_train, cfg.VAL_SIZE, seed=cfg.SEED)
_, val_set   = make_train_val_split(val_base,   cfg.VAL_SIZE, seed=cfg.SEED)  # gleiche Indizes!

train_loader = DataLoader(train_set, batch_size=cfg.KD_BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=cfg.KD_BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)


# Load Teacher froooozen
print("Loading teacher model (ViT-Large)...")
teacher = vit_l_16(weights=None)
teacher.heads.head = nn.Linear(teacher.heads.head.in_features, cfg.NUM_CLASSES)
teacher.load_state_dict(torch.load(cfg.VIT_L_PATH, map_location=DEVICE, weights_only=True))
teacher = teacher.to(DEVICE)
teacher.eval()

for param in teacher.parameters():
    param.requires_grad = False

print("Teacher loaded and frozen.")

# student
student = VisionTransformer(
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

student = student.to(DEVICE)

num_params = sum(p.numel() for p in student.parameters())
print(f"Student (ViT-Tiny) Parameters: {num_params:,}")

# KD Loss
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    klassische kd loss, sollte richtig sein..

        student_logits: raw logits from student
        teacher_logits: raw logits from teacher
        labels: ground truth labels
        T: temperature
        alpha: weight for soft loss
    """
    # Soft targets KL divergence
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)

    # Hard targets CE
    ce_loss = F.cross_entropy(student_logits, labels, label_smoothing=cfg.KD_LABEL_SMOOTHING)

    return alpha * kd_loss + (1.0 - alpha) * ce_loss

# Training Setup
optimizer = optim.AdamW(
    student.parameters(),
    lr=cfg.KD_LR,
    weight_decay=cfg.KD_WEIGHT_DECAY
)

def lr_lambda(epoch):
    if epoch < cfg.KD_WARMUP_EPOCHS:
        return (epoch + 1) / cfg.KD_WARMUP_EPOCHS
    else:
        import math
        progress = (epoch - cfg.KD_WARMUP_EPOCHS) / (cfg.KD_EPOCHS - cfg.KD_WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
eval_criterion = nn.CrossEntropyLoss()

scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))


def train_one_epoch(student, teacher, loader):
    student.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training (KD)")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            student_logits = student(images)

            with torch.no_grad():
                teacher_logits = teacher(images)

            loss = distillation_loss(
                student_logits, teacher_logits, labels,
                T=cfg.KD_TEMPERATURE, alpha=cfg.KD_ALPHA
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
        _, predicted = student_logits.max(1)
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
    print(f"Classical Knowledge Distillation")
    print(f"Teacher: ViT-Large | Student: ViT-Tiny")
    print(f"Epochs: {cfg.KD_EPOCHS} | T: {cfg.KD_TEMPERATURE} | α: {cfg.KD_ALPHA}")
    print(f"{'='*60}\n")

    for epoch in range(cfg.KD_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{cfg.KD_EPOCHS}] | LR: {scheduler.get_last_lr()[0]:.6f}")

        train_loss, train_acc = train_one_epoch(student, teacher, train_loader)
        val_loss, val_acc = evaluate(student, val_loader)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), cfg.VIT_T_KD_PATH)
            print(f"Best model saved! ({best_acc:.2f}%)")

    print(f"\n{'='*60}")
    print(f"Training Complete! Best Val Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {cfg.VIT_T_KD_PATH}")
    print(f"{'='*60}")
