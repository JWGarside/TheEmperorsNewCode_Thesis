# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-checkerboard
**Analysis Date:** 2025-05-25

## Analysis Results

1. Paper summary and core claims  
The paper “SliceGAN: Generating 3D Structures from a 2D Slice with GAN-Based Dimensionality Expansion” introduces a generative adversarial network architecture that:
• Trains a 3D conv-transpose “generator” on only 2D slices by slicing each generated 3D volume along x, y and z and feeding those 2D slices to a 2D “discriminator.”  
• Enforces “uniform information density” in the generator by choosing kernel size (k), stride (s) and padding (p) to avoid edge artifacts (e.g. k=4, s=2, p=2).  
• Extends trivially to anisotropic materials by using separate discriminators for different slice orientations.  
• Demonstrates fast generation (10^8 voxels in seconds) and produces statistically accurate 3D microstructures from a single 2D micrograph.

2. Implementation assessment  
The repository contains:  
• run_slicegan.py: CLI for training (argument ‘1’) or sampling (‘0’). Sets project paths, data type (‘tif3D’ for an isotropic 3D stack), image type (‘nphase’), network hyperparameters and calls networks.slicegan_rc_nets by default.  
• slicegan/networks.py: Two net-builders:  
  – slicegan_nets implements the pure conv-transpose architecture matching the paper’s Table 1 with k=s/p choices.  
  – slicegan_rc_nets implements a “resize-convolution” variant (upsample + 3×3 conv) plus conv-transpose layers.  
• slicegan/model.py: WGAN-GP training loop with three discriminators (x,y,z) and one generator. Uses slice permutation/reshape to feed 2D slices to discriminators. Critic iterations=5, gradient penalty λ=10, Adam LR=1e-4 (β₁=0.9,β₂=0.99), 100 epochs.  
• slicegan/preprocessing.py: Builds datasets of one-hot encoded 2D patches (32×900 samples per orientation for 3D stacks).  
• slicegan/util.py: Utilities for weight init, gradient penalty, plotting, model saving/loading, and test-time volume sampling with optional periodic seeding.

The core algorithm—3D generator + slicing + 2D discriminators + WGAN-GP—is implemented end to end, and the code includes the recommended initial latent spatial size of 4×4×4.

3. Categorized discrepancies  

Critical  
• run_slicegan.py calls slicegan_rc_nets (the “resize-conv” variant) by default rather than slicegan_nets, so the paper’s pure conv-transpose architecture (k=4, s=2, p=2, ….) is never used unless manually edited.  
• The resize-conv generator in slicegan_rc_nets diverges substantially from the paper’s design: conv-transpose layers use inconsistent k/s/p values (k=4, s=3, p=1), followed by a trilinear upsample and a 3×3 conv. This variant is not described in the paper and is likely to produce wrong volume sizes or checkerboard artifacts.  

Minor  
• The paper suggests balancing batch sizes m_G=2 m_D to compensate for the many slices per generated volume; the code instead uses typical WGAN-GP critic_iters=5 (D updates per G update) and equal batch_size for G and D.  
• In run_slicegan parameters, the padding list dp for the discriminator has length 5 vs. layers=6, leading to one fewer conv layer than intended.  
• README refers to “train.py” as the training driver, but the code actually uses model.py.  

Cosmetic  
• util.mkdr prompts interactively if the project directory exists, which may interrupt automated runs.  
• Preprocessing draws 32×900 patches per orientation—this crop count is not discussed in the paper but does not alter the method.  
• Learning-rate, epochs, Adam β’s etc. are not specified in the paper but are reasonable defaults.

4. Overall reproducibility conclusion  
The fundamental SliceGAN algorithm (3D conv-transpose generator, slicing to 2D discriminators, WGAN-GP) is present in the code, and key design choices (latent spatial size, one-hot encoding, multi-D setup) are faithfully implemented. However, the default code path uses a different “resize-conv” network that is neither described in the paper nor matches the critical uniform‐density transpose‐conv architecture. Moreover, that variant contains mismatched hyperparameters and upsampling logic. As provided, the repository will not reproduce the paper’s results without manually switching to slicegan_nets, correcting kernel/stride/padding parameters, and ensuring network definitions match Table 1. With those corrections, the paper’s core claims should be reproducible; in its current form, the code diverges critically from the described architecture.