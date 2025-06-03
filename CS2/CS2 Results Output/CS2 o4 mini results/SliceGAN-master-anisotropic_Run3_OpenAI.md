# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-anisotropic
**Analysis Date:** 2025-05-25

## Analysis Results

1. Paper summary and core claims  
The paper introduces SliceGAN, a generative adversarial network architecture that learns to synthesise isotropic 3D microstructural volumes from a single representative 2D slice.  Key methodological elements:  
•  A 3D generator built entirely from transpose-convolutional layers (kernel size k=4, stride s=2, padding p chosen to ensure uniform information density) mapping a latent tensor of size 64×4×4×4 to a 3-channel 64×64×64 volume, with softmax output for n-phase data.  
•  A 2D discriminator (one per axis) receiving all l slices extracted along x, y, and z axes of each generated volume.  
•  Training via the Wasserstein GAN with gradient penalty (WGAN-GP), alternating  nD≈5 discriminator updates per generator update.  
•  Demonstrated isotropic reconstruction from one 2D image; simple extension to anisotropic microstructures using independent discriminators per orientation.  

2. Implementation assessment  
•  Entry point: run_slicegan.py  
  – Parses a single “training” flag; builds project path; sets image_type='nphase', nc=3, data_type='tif3D', data_path=['Examples/NMC.tif'], img_size=64, nz=32, latent spatial size lz=4, 5 generator layers & 6 discriminator layers, kernel/stride/padding sequences matching the paper’s k=4, s=2, p={2,2,2,2,3}.  
  – By default calls networks.slicegan_rc_nets(…) → returns Generator and Discriminator classes.  
  – On Training=1, invokes model.train(…) which:  
    · Loads the single 3D TIFF and builds three PyTorch Datasets by randomly slicing 32×900 patches along each axis (tif3D preprocessing).  
    · Constructs one Generator (netG) and three Discriminators (netDs[0..2]).  
    · Trains in WGAN-GP style:  
       – Sampling noise ∈ℝ⁸ˣ³²ˣ⁴ˣ⁴ˣ⁴ → fake 3D batch → slices and permutes into 2D batches.  
       – Discriminator updates: L_D = E[D(fake)] – E[D(real)] + λ‖∇D(interp)‖².  
       – Generator updates every 5 discriminator steps: L_G = –∑_axes E[D(fake_slices)].  
    · Saves model state and example slice plots intermittently.  
  – On Training=0, instantiates netG, loads weights and calls util.test_img to generate a 64³ TIFF.  

•  Core pieces implemented:  
  – 3D convolutional generator and 2D convolutional discriminators matching the paper’s layer counts and hyperparameters.  
  – Slicing protocol that feeds all three orientations into the discriminator(s).  
  – WGAN-GP loss with gradient penalty.  
  – Data preprocessing for 3D TIFF into one-hot encoded slices.  

3. Categorized discrepancies  

Critical  
1. Default network factory mismatch  
   – run_slicegan.py uses slicegan_rc_nets (a hybrid upsample+conv architecture), not the pure transpose-convolution generator described in the paper.  The rc variant:  
     • Applies only four ConvTranspose3d layers then a trilinear Upsample + a small Conv3d, instead of five specified ConvTranspose3d layers.  
     • This hybrid design (and its final softmax) is not mentioned in the paper and deviates from the stated architecture.  
   →  Reproducing the published results requires switching to slicegan_nets rather than slicegan_rc_nets.  

Minor  
2. Batch size choices  
   – The paper suggests using a larger generator batch size (mG≈2 mD) to rebalance training; code uses mG = mD = 8.  
3. Anisotropic training logic bug  
   – Although the paper claims support for three independent discriminators, model.train overrides netD to netDs[0] in every orientation, so only one discriminator is ever trained.  
4. Unused module in rc generator  
   – slicegan_rc_nets builds a fifth ConvTranspose3d (convs[-1]) that is never applied in forward().  (Harmless but confusing.)  

Cosmetic  
5. Normalisation  
   – Code uses BatchNorm3d between generator layers; the paper does not explicitly discuss normalisation layers.  
6. Parameter logging & plotting  
   – Code saves loss graphs and slice plots which are not described in the paper’s evaluation section.  

4. Overall reproducibility conclusion  
The core methodology—3D GAN generator + 2D slice-based discriminator with WGAN-GP training—is faithfully implemented.  However, the default code variant (slicegan_rc_nets) does not match the transpose-convolution-only architecture reported in the paper.  By selecting the correct network factory (slicegan_nets) and optionally adjusting the batch-size ratio, one can reproduce the paper’s experiments.  Minor bugs in the anisotropic extension and some cosmetic differences do not impede the main isotropic use case.  Overall, the repository contains the necessary components for reproducing the paper’s primary claims, but users must be aware of and correct the factory mismatch to align with the published architecture.