# StarGAN with CUT Integration

This project extends [StarGAN](https://arxiv.org/abs/1711.09020) by incorporating [CUT (Contrastive Unpaired Translation)](https://arxiv.org/abs/2007.15651) to enhance the quality and semantic consistency of multi-domain image translations. The combined model is effective for attribute-guided image generation tasks like facial expression, hair color, and identity transformation.

## Environment

We recommend setting up your environment with Conda:

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install numpy packaging pillow

## Project Structure

├── main.py                # Entry point for training and testing
├── solver.py              # Contains the training/testing logic
├── StarGAN.py             # Generator, Discriminator, and CUT feature loss modules
├── data_loader.py         # [REQUIRED] Custom dataset loader
├── logger.py              # [OPTIONAL] TensorBoard logger
└── stargan/
    ├── logs/              # TensorBoard logs (if enabled)
    ├── models/            # Model checkpoints
    ├── samples/           # Sample images during training
    └── results/           # Final test results

## Training  
To train the model on the CelebA dataset with CUT loss:

python main.py \
  --mode train \
  --dataset CelebA \
  --image_size 128 \
  --batch_size 16 \
  --g_repeat_num 6 \
  --d_repeat_num 6 \
  --lambda_NCE 1.0 \
  --nce_layers 0,4,8,12,16

python main.py --mode test --dataset CelebA --test_iters 200000
