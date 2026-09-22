import os

DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model checkpoint paths
VIT_L_PATH = os.path.join(CHECKPOINT_DIR, "vit_l16_cifar100_best.pth")
VIT_T_PATH = os.path.join(CHECKPOINT_DIR, "vit_tiny_cifar100_best.pth")
VIT_T_KD_PATH = os.path.join(CHECKPOINT_DIR, "vit_tiny_kd_classical_best.pth")
VIT_T_DEIT_PATH = os.path.join(CHECKPOINT_DIR, "vit_tiny_deit_distill_best.pth")


NUM_CLASSES = 100
IMAGE_SIZE = 224

SEED = 42
VAL_SIZE = 5000          # 45k train / 5k val aus dem 50k Trainingsset


# ViT-Large Teacher Finetuning pretrained
VIT_L_BATCH_SIZE = 16
VIT_L_ACCUM_STEP = 2 # 16 * 2 = 32 effektiv batch size
VIT_L_LR = 2e-5
VIT_L_WEIGHT_DECAY = 0.05
VIT_L_EPOCHS = 10
VIT_L_WARMUP_STEPS = 500
VIT_L_LABEL_SMOOTHING = 0.1

# ViT-Tiny Student Baseline From scratch
VIT_T_BATCH_SIZE = 128
VIT_T_EPOCHS = 150
VIT_T_LR = 5e-4
VIT_T_WEIGHT_DECAY = 0.05
VIT_T_WARMUP_EPOCHS = 10
VIT_T_LABEL_SMOOTHING = 0.1

# ViT-Tiny architecture
VIT_T_PATCH_SIZE = 16
VIT_T_NUM_LAYERS = 12
VIT_T_NUM_HEADS = 3
VIT_T_HIDDEN_DIM = 192
VIT_T_MLP_DIM = 768
VIT_T_DROPOUT = 0.1

# Classical Knowledge Distillation
KD_BATCH_SIZE = 128
KD_EPOCHS = 150
KD_LR = 5e-4
KD_WEIGHT_DECAY = 0.05
KD_WARMUP_EPOCHS = 10
KD_TEMPERATURE = 3.0
KD_ALPHA = 0.5  # weight for soft loss (1-alpha for hard loss)
KD_LABEL_SMOOTHING = 0.1

# DeiT-Style Distillation
DEIT_BATCH_SIZE = 128
DEIT_EPOCHS = 150
DEIT_LR = 5e-4
DEIT_WEIGHT_DECAY = 0.05
DEIT_WARMUP_EPOCHS = 10
DEIT_TEMPERATURE = 3.0
DEIT_ALPHA = 0.5  # balance between cls and dist loss
DEIT_LABEL_SMOOTHING = 0.1
DEIT_HARD_DISTILL = True # like in the paper


# Data Augmentation
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# Evaluation
EVAL_BATCH_SIZE = 128
NUM_BENCHMARK_RUNS = 100
NUM_WARMUP_RUNS = 20
