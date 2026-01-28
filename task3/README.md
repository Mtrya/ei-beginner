# Task 3: Diffusion Policy for Push-T

Implementation of Diffusion Policy for visuomotor control on the Push-T task, built from scratch with clean, modern code.

## 📁 Project Structure

```
task3/
├── src/
│   ├── envs/
│   │   ├── __init__.py
│   │   └── pusht.py           # Push-T environment (gymnasium API)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vision_encoder.py   # ResNet18 + Spatial Softmax
│   │   ├── unet1d.py           # Conditional UNet for denoising
│   │   └── diffusion_policy.py # Main policy (DDPM/DDIM)
│   └── dataset.py              # Zarr dataset loader
├── data/
│   └── pusht/                  # Push-T dataset (downloaded)
├── demo_pusht.py               # Interactive demo
├── record_pusht_episode.py     # Record episodes for visualization
└── diffusion_policy.md         # Paper notes

references/                     # Original implementations
├── diffusion_policy/           # Official repo
└── lerobot/                    # HuggingFace LeRobot
```

## 🎯 Implementation Highlights

### **Clean, Modern Codebase**
- ✅ **Gymnasium API** instead of deprecated gym
- ✅ **No Hydra complexity** - simple Python configs
- ✅ **Type hints** throughout
- ✅ **Minimal dependencies**

### **Core Components**
1. **Push-T Environment** (`src/envs/pusht.py`)
   - 2D planar pushing task with PyMunk physics
   - Agent (blue circle) pushes T-block (red) to goal (green)
   - Supports both state and image observations

2. **Vision Encoder** (`src/models/vision_encoder.py`)
   - ResNet18 backbone with GroupNorm
   - Spatial Softmax pooling for spatial features
   - ~11M parameters (11% of total)

3. **Conditional UNet1D** (`src/models/unet1d.py`)
   - 1D U-Net for temporal action sequence denoising
   - FiLM conditioning on observations
   - ~89M parameters (89% of total)

4. **Diffusion Policy** (`src/models/diffusion_policy.py`)
   - DDPM for training (100 steps)
   - DDIM for fast inference (10 steps)
   - Predicts action sequences with receding horizon

### **Model Size**
- **Total**: ~100M parameters (~381 MB)
- **Observation encoder**: 11.2%
- **UNet denoiser**: 88.8%

## 🚀 Quick Start

### **1. Try the Environment**
```bash
# Interactive demo (mouse control)
uv run python task3/demo_pusht.py

# Record episode and create visualization
uv run python task3/record_pusht_episode.py --mode both
```

### **2. Download Dataset**
```bash
# Option 1: Use the download script
cd task3
./download_data.sh

# Option 2: Manual download
cd task3/data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip && rm pusht.zip
```

**Dataset Info:**
- **Location**: `task3/data/pusht/pusht_cchi_v7_replay.zarr`
- **Size**: 31 MB (compressed)
- **Episodes**: ~200
- **Total frames**: ~25,000

### **3. Training** (Coming next)
```bash
# Train diffusion policy
uv run python task3/train.py --config config.yaml

# Evaluate
uv run python task3/eval.py --checkpoint checkpoints/latest.pt
```

## 📊 Key Hyperparameters

From the paper and official implementation:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Observation horizon (`T_o`) | 2 | Number of observation frames |
| Action horizon (`T_a`) | 8 | Actions to execute before replanning |
| Prediction horizon (`T_p`) | 16 | Total predicted action sequence |
| Diffusion steps (train) | 100 | DDPM training timesteps |
| Diffusion steps (inference) | 10 | DDIM fast sampling |
| Batch size | 64 | Training batch size |
| Learning rate | 1e-4 | Adam optimizer |
| Noise schedule | Square Cosine | Beta schedule |

## 🎓 Key Insights

### **Why Diffusion for Robot Policies?**
1. **Multimodal distributions**: Naturally handles multiple valid actions
2. **High-dimensional outputs**: Can predict action sequences (not just single actions)
3. **Stable training**: No adversarial training or negative sampling needed

### **Design Decisions**
1. **Position control > Velocity control**: Surprisingly, position commands work better!
2. **Observation as condition**: Don't model joint distribution p(obs, action), use conditional p(action|obs)
3. **Receding horizon**: Balance between planning (long sequences) and reactivity (replanning)
4. **FiLM conditioning**: Efficient way to inject observations into U-Net

### **Architecture Trade-offs**
- **CNN**: Faster, more stable, but over-smooths high-frequency signals
- **Transformer**: Better for complex tasks, but needs more tuning
- **Our choice**: CNN (simpler to start)

## 📚 References

- **Paper**: [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://diffusion-policy.cs.columbia.edu/)
- **Official Code**: [diffusion-policy](https://github.com/columbia-ai-robotics/diffusion_policy)
- **LeRobot**: [Hugging Face LeRobot](https://github.com/huggingface/lerobot)

## ✅ Progress

- [x] Environment implementation (gymnasium API)
- [x] Dataset loader (Zarr format)
- [x] Vision encoder (ResNet18 + Spatial Softmax)
- [x] Conditional UNet1D (temporal denoising)
- [x] Diffusion Policy (DDPM/DDIM)
- [x] All components tested
- [ ] Training script
- [ ] Evaluation script
- [ ] Experiments & ablations
- [ ] Comprehensive notes

## 🔬 Next Steps

1. **Training**: Implement training loop with logging and checkpointing
2. **Evaluation**: Test on environment and compute success rate
3. **Experiments**:
   - CNN vs Transformer architectures
   - DDPM vs DDIM vs Flow Matching
   - Position vs velocity control
   - Different horizon configurations
4. **Notes**: Document findings and insights

---

**Author**: Built from scratch following the Diffusion Policy paper  
**Date**: 2026-01-28
