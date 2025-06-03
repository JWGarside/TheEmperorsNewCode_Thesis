# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-softmax
**Analysis Date:** 2025-05-25

## Analysis Results

1. Paper Summary and Core Claims  
SliceGAN is a GAN architecture that learns to generate arbitrarily large, statistically representative 3D image volumes from just 2D micrograph slices. Its core contributions are:  
• Training a 3D transposed-convolutional generator whose outputs are sliced along x, y and z and judged by a 2D discriminator (Section 3).  
• Careful choice of transpose-conv kernel size, stride and padding to ensure uniform “information density” and avoid edge artifacts (Section 4; Table 1).  
• Extension to anisotropic materials via separate discriminators per slice orientation.  
Empirically, SliceGAN reproduces 3D microstructural statistics (e.g. two-point correlations, diffusivity) of diverse materials using only 2D training data, producing 10⁸-voxel volumes in seconds.

2. Implementation Assessment  
• Entry point (run_slicegan.py) sets up project paths, data paths, image type (‘nphase’ for segmented micrographs), image size 64, latent spatial size 4, number of layers, kernel sizes [4,…], strides [2,…], padding for G [2,2,2,2,3] and for D [1,1,1,1,0], filter widths matching Table 1.  
• Preprocessing (preprocessing.py) for ‘tif3D’ reads a single 3D volume, subsamples it, and builds three PyTorch TensorDatasets containing random 2D slices in x, y and z, one-hot encoded across phases.  
• Architectures (networks.py):  
  – slicegan_nets: pure transpose-conv generator + batchnorm, with sigmoid/tanh outputs.  
  – slicegan_rc_nets (“resize-conv”): applies all but last transpose-conv layers, then a trilinear upsample + 3×3 conv, sigmoid.  
• Training loop (model.py) implements the multi-discriminator WGAN-GP procedure:  
  – For each minibatch, generate a 3D fake volume from noise of shape (batch, z_channels=32, 4,4,4).  
  – Slice it along x,y,z (via permute+reshape) into 2D batches of size 64×batch, feed each to its discriminator, apply gradient penalty.  
  – Every 5 D-steps, update G using the sum of discriminator outputs on the slices.  
  – Save model checkpoints and example slice-plots periodically.  
• Test script (util.test_img) reloads a saved G, generates a smaller periodic or non-periodic volume, and writes it to a .tif via an argmax across channels.

3. Categorized Discrepancies  
Critical  
• run_slicegan.py always invokes slicegan_rc_nets, not the pure transpose-conv slicegan_nets described in the paper. The “resize-conv” implementation has a bug in computing the upsample size (uses the same dimension twice) and thus does not match the uniform transpose-conv architecture or output sizes in the paper. This will prevent correct reproduction of the core generator architecture.  
Minor  
• Discriminator padding list dp is length 5 while dk/ds are length 6, so only 5 conv layers get built instead of the intended 6 (Table 1).  
• The paper recommends m_G = 2·m_D for generator batch size, but code uses equal batch sizes for G and D (both 8).  
• No explicit weight initialization (weights_init is defined but never applied), whereas GAN performance can depend on it.  
• The paper specifies a softmax across phase channels; code uses sigmoid per channel + argmax at post-processing.  
• No fixed random seed for reproducibility of results.  
Cosmetic  
• The README refers to a non-existent train.py (actual training code is in model.py).  
• Raytrace.py example requires plotoptix and external .tif files; it’s not part of core GAN pipeline.  
• No code is provided for the TauFactor statistical analysis used in the paper’s validation (Figure 4).

4. Overall Reproducibility Conclusion  
The repository captures the high-level structure of SliceGAN—slicing a 3D generator’s output and training with a 2D discriminator in a WGAN-GP framework—but contains implementation divergences that will prevent exactly reproducing the published results. In particular, the resize-conv generator branch used by default is not the transpose-conv architecture described in the paper and appears to contain a sizing bug; the discriminator also has one fewer layer than specified. Addressing these critical mismatches (or switching to the slicegan_nets implementation and correctly matching the paper’s layer counts and transpose-conv parameters) would be necessary to faithfully reproduce the core claims. Once those are remedied, the remaining minor issues (batch sizes, initialization, softmax vs sigmoid) could be tuned to match the paper’s performance.