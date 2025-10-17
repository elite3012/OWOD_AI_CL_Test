# Quick Start Guide - Enhanced OWOD

This guide will get you started with the Enhanced OWOD system in under 30 minutes.

## 🚀 5-Minute Setup

### 1. Clone and Setup Environment

```powershell
# Clone repository (if not already done)
cd OWOD_AI_CL

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```powershell
# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import open_clip; print('OpenCLIP: OK')"
```

## 📖 Understanding the Project

### What is OWOD?

**Open World Object Detection** is a system that:
- ✅ Detects known objects (trained classes)
- ✅ Identifies unknown objects (never seen before)
- ✅ Learns new classes incrementally without forgetting old ones

### Research Focus

This project demonstrates:
1. **Foundation Models**: Using CLIP for robust visual features
2. **PEFT**: LoRA for efficient fine-tuning (0.8% trainable parameters)
3. **Continual Learning**: Memory replay + knowledge distillation
4. **Multi-Modality**: Vision-language integration
5. **Efficiency**: Hardware-aware optimizations

## 🎯 Core Components

### 1. CLIP Backbone with LoRA

**Location**: `models/backbones/clip_backbone.py`

```python
# How it works:
# 1. Load pre-trained CLIP vision encoder
# 2. Freeze base weights
# 3. Add LoRA adapters (only 0.8% parameters trainable)
# 4. Extract multi-scale features for detection

# Example usage:
from models.backbones.clip_backbone import build_clip_backbone
from configs.defaults import get_cfg

cfg = get_cfg()
cfg.FOUNDATION_MODEL.TYPE = "clip"
cfg.PEFT.ENABLED = True
cfg.PEFT.METHOD = "lora"

backbone = build_clip_backbone(cfg)
```

### 2. Memory Replay System

**Location**: `models/continual_learning/memory_replay.py`

```python
# How it works:
# 1. Store exemplars from previous tasks
# 2. Select representative samples (herding strategy)
# 3. Replay during training to prevent forgetting

# Example usage:
from models.continual_learning.memory_replay import MemoryBuffer

buffer = MemoryBuffer(
    memory_size=2000,
    selection_strategy="herding"
)

# During training:
buffer.update(images, targets, features)
replay_images, replay_targets = buffer.sample(batch_size=16)
```

### 3. Knowledge Distillation

**Location**: `models/continual_learning/knowledge_distillation.py`

```python
# How it works:
# 1. Keep previous model as "teacher"
# 2. New model learns from both:
#    - New task data (hard labels)
#    - Teacher predictions (soft labels)
# 3. Preserves old knowledge while learning new

# Example usage:
from models.continual_learning.knowledge_distillation import KnowledgeDistillation

distillation = KnowledgeDistillation(
    temperature=2.0,
    alpha=0.5  # 50% teacher, 50% task
)

total_loss, loss_dict = distillation(
    student_model=current_model,
    teacher_model=previous_model,
    inputs=images,
    targets=labels,
    task_loss=detection_loss
)
```

## 🔧 Configuration System

### Understanding Configs

**Location**: `configs/defaults.py`

The system uses hierarchical YAML configs:

```yaml
# configs/owod_cl/task1_clip_lora.yaml

FOUNDATION_MODEL:
  TYPE: "clip"
  CLIP:
    MODEL_NAME: "ViT-B/16"
    FREEZE_VISION: True

PEFT:
  ENABLED: True
  METHOD: "lora"
  LORA:
    R: 8         # Rank (lower = fewer params)
    ALPHA: 16    # Scaling
    DROPOUT: 0.1

OWOD:
  PREV_INTRODUCED_CLS: 0   # No previous classes
  CUR_INTRODUCED_CLS: 20   # Learn 20 classes

SOLVER:
  BASE_LR: 0.0001
  MAX_ITER: 20000
```

### Key Parameters

| Parameter | What it does | Typical Values |
|-----------|--------------|----------------|
| `PEFT.LORA.R` | LoRA rank (lower = more efficient) | 4, 8, 16 |
| `PEFT.LORA.ALPHA` | LoRA scaling factor | 8, 16, 32 |
| `CONTINUAL_LEARNING.REPLAY.MEMORY_SIZE` | Exemplars to store | 1000-5000 |
| `CONTINUAL_LEARNING.DISTILLATION.ALPHA` | Teacher weight | 0.3-0.7 |
| `SOLVER.BASE_LR` | Learning rate | 1e-4 to 1e-5 |

## 📊 Training Workflow

### Task 1: Initial Training (Classes 1-20)

```powershell
# Train on first 20 classes
python train.py `
    --config-file configs/owod_cl/task1_clip_lora.yaml `
    --num-gpus 1 `
    --output-dir output/task1

# What happens:
# - Loads CLIP backbone
# - Adds LoRA adapters
# - Trains detection head
# - Stores exemplars in memory buffer
# - Saves: model_final.pth, memory_buffer_final.pth
```

**Expected Time**: 2-4 hours on RTX 3090

### Task 2: Continual Learning (Classes 21-40)

```powershell
# Continue learning without forgetting
python train.py `
    --config-file configs/owod_cl/task2_continual.yaml `
    --prev-model output/task1/model_final.pth `
    --num-gpus 1 `
    --output-dir output/task2

# What happens:
# - Loads Task 1 model as teacher
# - Trains on new classes (21-40)
# - Replays exemplars from Task 1
# - Applies knowledge distillation
# - Prevents forgetting of classes 1-20
```

**Expected Time**: 1.5-3 hours on RTX 3090

## 📈 Monitoring Training

### Log Files

```powershell
# View training logs
Get-Content output/task1/train.log -Tail 50

# Expected output:
# [2025-10-17 10:30:15] INFO - Epoch [1] Iter [100] Loss: 0.8234 LR: 0.000100
# [2025-10-17 10:31:22] INFO - Epoch [1] Iter [200] Loss: 0.7156 LR: 0.000100
```

### Checkpoints

```
output/task1/
├── config.yaml              # Configuration used
├── train.log                # Training logs
├── model_epoch_5.pth       # Intermediate checkpoint
├── model_final.pth         # Final model
└── memory_buffer_final.pth # Exemplar buffer
```

## 🔍 Understanding Results

### Metrics to Track

1. **Known Class mAP**: Detection accuracy on learned classes
   - Task 1: ~67%
   - Task 2: ~65% (small drop expected)

2. **Unknown Recall**: Ability to detect novel objects
   - Should stay above 75-80%

3. **Forgetting**: How much old knowledge is lost
   - Target: <2% per task

4. **Trainable Parameters**: Efficiency metric
   - With LoRA: 0.8% of total parameters

## 🎨 Visualizing Results

```python
# TODO: When visualization is implemented
from utils.visualization import visualize_detections

visualize_detections(
    image_path="demo/images/sample.jpg",
    model_path="output/task2/model_final.pth",
    config_path="configs/owod_cl/task2_continual.yaml",
    output_path="output/vis/"
)

# Expected output:
# - Bounding boxes for known classes
# - "Unknown" labels for novel objects
# - Confidence scores
```

## 🐛 Troubleshooting

### GPU Out of Memory

```yaml
# Reduce batch size in config
SOLVER:
  IMS_PER_BATCH: 4  # Instead of 8

# Enable gradient checkpointing
OPTIMIZATION:
  GRADIENT_CHECKPOINTING:
    ENABLED: True
```

### Import Errors

```powershell
# Add project to Python path
$env:PYTHONPATH = "$pwd;$env:PYTHONPATH"
```

### Slow Training

```yaml
# Use mixed precision
OPTIMIZATION:
  MIXED_PRECISION:
    ENABLED: True
    DTYPE: "fp16"

# Reduce image size
INPUT:
  MIN_SIZE_TRAIN: 600  # Instead of 800
```

## 📚 Next Steps

### 1. Run Baseline Experiments

```powershell
# Compare different LoRA ranks
python train.py --config-file configs/owod_cl/task1_clip_lora.yaml `
    PEFT.LORA.R 4 --output-dir output/task1_r4

python train.py --config-file configs/owod_cl/task1_clip_lora.yaml `
    PEFT.LORA.R 16 --output-dir output/task1_r16
```

### 2. Try Different Memory Sizes

```powershell
# Small memory
python train.py --config-file configs/owod_cl/task2_continual.yaml `
    CONTINUAL_LEARNING.REPLAY.MEMORY_SIZE 1000 `
    --output-dir output/task2_mem1k

# Large memory
python train.py --config-file configs/owod_cl/task2_continual.yaml `
    CONTINUAL_LEARNING.REPLAY.MEMORY_SIZE 5000 `
    --output-dir output/task2_mem5k
```

### 3. Ablation Studies

```powershell
# Without distillation
python train.py --config-file configs/owod_cl/task2_continual.yaml `
    CONTINUAL_LEARNING.DISTILLATION.ENABLED False `
    --output-dir output/task2_no_distill

# Without replay
python train.py --config-file configs/owod_cl/task2_continual.yaml `
    CONTINUAL_LEARNING.REPLAY.ENABLED False `
    --output-dir output/task2_no_replay
```

## 🎓 Research Ideas

1. **PEFT Comparison**
   - Implement adapters and prompt tuning
   - Compare efficiency vs accuracy

2. **Memory Strategies**
   - Test herding vs random vs entropy
   - Optimal buffer sizes

3. **Multi-Modal Fusion**
   - Add text descriptions for classes
   - Text-guided unknown detection

4. **Foundation Model Comparison**
   - CLIP vs DINOv2 vs Hybrid
   - Feature quality analysis

## 💡 Pro Tips

- Start with small experiments to understand the system
- Use smaller batch sizes and lower resolution for faster iteration
- Monitor GPU memory usage with `nvidia-smi`
- Save intermediate checkpoints frequently
- Keep configs organized in separate folders per experiment
- Document parameter changes and results

## 📖 Additional Resources

- **Full Documentation**: `README.md`
- **Installation Guide**: `INSTALL.md`
- **Project Overview**: `PROJECT_OVERVIEW.md`
- **Original OWOD**: `OWOD-master/README.md`

## 🆘 Getting Help

1. Check `PROJECT_OVERVIEW.md` for architecture details
2. Review configuration options in `configs/defaults.py`
3. Look at example configs in `configs/owod_cl/`
4. Examine model code in `models/` for implementation details

---

**You're ready to start experimenting!** 🎉

Begin with Task 1, understand the workflow, then move to continual learning scenarios. The system is modular, so you can easily add new components for your research needs.
