# StarGAN with CUT Integration

This repository extends the original StarGAN by integrating **Contrastive Unpaired Translation (CUT)**, enhancing the semantic consistency and quality of domain-translated images.

## Environment Setup

Create a Conda environment:

```bash
conda create -n stargan_cut python=3.8
conda activate stargan_cut

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install numpy packaging pillow tensorboard
```

## Project Structure

```
.
├── main.py                  # Main script for training/testing
├── solver.py                # Solver class handling training/testing logic
├── StarGAN.py               # Generator, Discriminator, and CUT losses
├── data_loader.py           # Data loading utilities (custom implementation required)
├── logger.py                # TensorBoard logging
└── stargan/
    ├── logs/                # TensorBoard logs
    ├── models/              # Model checkpoints
    ├── samples/             # Samples generated during training
    └── results/             # Translated images during testing
```

## Training

To train on CelebA dataset:

```bash
python main.py --mode train \
               --dataset CelebA \
               --image_size 128 \
               --batch_size 16 \
               --g_repeat_num 6 \
               --d_repeat_num 6 \
               --lambda_NCE 1.0 \
               --nce_layers 0,4,8,12,16
```

Resume from checkpoint:

```bash
python main.py --mode train --resume_iters 100000
```

## Testing

Generate images from the trained model:

```bash
python main.py --mode test --dataset CelebA --test_iters 200000
```

Generated images are saved to `stargan/results/`.

## Model Overview

| Component         | Description |
|-------------------|-------------|
| Generator         | Translates images between domains |
| Discriminator     | Classifies real/fake and domain labels |
| Feature Network   | Provides features for CUT |
| Losses            | Adversarial, classification, reconstruction, and CUT (PatchNCE) |

## Key Enhancements

- ✅ Integrated CUT's **PatchNCE loss** for better semantic consistency
- ✅ Multi-domain translation capability
- ✅ Modular and customizable

## References

- [StarGAN (Choi et al., CVPR 2018)](https://arxiv.org/abs/1711.09020)
- [CUT (Park et al., ECCV 2020)](https://arxiv.org/abs/2007.15651)
- Li, Y., Pang, A. W., Kim, T., & Chong, J. Augmenting Bronchoscopy Image Data Using an Enhanced StarGAN for Inhalation Injury Classification.
