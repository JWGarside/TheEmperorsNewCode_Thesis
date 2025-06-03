# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-anisotropic
**Analysis Date:** 2025-05-25

## Analysis Results

1. Paper summary and core claims  
- Introduces SliceGAN, a generative adversarial network that synthesises 3D volumetric microstructures from 2D slices alone.  
- Key ideas:  
  • A 3D generator G produces volumes f; 2D discriminator D is trained on slices of f along x, y, z axes, paired with real 2D images.  
  • Uses WGAN-GP loss for stable training.  
  • Enforces “uniform information density” in transpose-convolution layers (kernel size k=4, stride s=2, padding p=2) to avoid edge artifacts.  
  • Latent input z has spatial dimensions 4×4×4, allowing post-training generation of arbitrarily large volumes by varying z’s spatial extent.  
  • Supports isotropic materials (single 2D image) and anisotropic materials (three orthogonal 2D images, separate discriminators per orientation).  
- Demonstrated high-fidelity microstructures for a diversity of materials and validated statistical metrics (two-point correlations, tortuosity, triple-phase boundaries).

2. Implementation assessment  
- Configuration and entry point:  
  • run_slicegan.py sets project paths, data type (‘nphase’), image size (64³), latent channels (32), network hyperparameters (5 generator layers, 5 discriminator layers, kernel/stride lists, filter sizes).  
  • By default calls slicegan_rc_nets (a “resize-convolution” generator) rather than the pure transpose-convolution generator described in the paper.  
- Data preprocessing (preprocessing.batch):  
  • Reads a 3D TIFF, applies subsampling, extracts 32×900 random 2D patches per orientation, one-hot encodes n phases.  
- Model training (model.train):  
  • If a single data path is given, duplicates it for x, y, z and sets isotropic=True; else anisotropic=False.  
  • Builds one 3D generator and three 2D discriminators, but for isotropic only the first discriminator is used.  
  • Implements WGAN-GP: Adam optimisers, gradient penalty λ=10, critic_iters=5, learning rates 1e-4, Adam betas (0.9,0.99), 100 epochs, batch size 8.  
  • Noise tensor of shape (batch, nz, 4, 4, 4) is fed to G; generated volumes are permuted and reshaped into slices (batch×64 slices) for D.  
  • During training D sees real/fake batches of size 8×64 slices; G update accumulates −D(fake) losses.  
- Networks (networks.py):  
  • slicegan_nets: pure transpose-convolution generator matching paper’s layer specs.  
  • slicegan_rc_nets: conv-transpose for all but last layer, followed by trilinear upsampling and a final 3D conv (“resize-conv”) to avoid checkerboard artifacts.  
- Utilities (util.py):  
  • Gradient penalty calculation, ETA logging, plotting of losses and example slices, test_img for post-training generation (supports larger latent spatial size and periodic tiling).

3. Categorized discrepancies  
- Critical  
  • Anisotropic training bug: In model.train, inside the loop over three orientations, `netD` and `optimizer` are overridden to always use the first discriminator (`netDs[0]`), so the second and third discriminators are never trained. This prevents correct anisotropic reconstruction.  
  • Syntax error in slicegan_rc_nets: the upsample size is computed as `int(x.shape[2]-1,)*2` (trailing comma) which will raise a TypeError at runtime.  
- Minor  
  • Default use of slicegan_rc_nets (resize-conv) rather than the pure transpose-conv generator emphasised in the paper. This may alter performance or artifacts but preserves the slice-based GAN concept.  
  • The paper’s recommendation mG = 2·mD (generator sees twice as many slices per update) is not implemented; code uses equal batch sizes.  
  • Weight initialization (`util.weights_init`) is defined but never applied to the networks (PyTorch uses its own defaults).  
  • dp list length mismatched vs dk/ds in run_slicegan (zip truncates dp); not fatal but obscures layer definitions.  
- Cosmetic  
  • Adam β-parameters (0.9, 0.99) differ from the common WGAN-GP defaults (0.5, 0.9); the paper did not prescribe specific values.  
  • Code comments occasionally refer to “train.py” whereas the file is `model.py`.

4. Overall reproducibility conclusion  
The core isotropic SliceGAN method—training a single 2D discriminator on slices of a 3D generator under WGAN-GP— is implemented and should reproduce the main isotropic results with minor parameter tuning and a few code fixes (syntax in upsampling, applying weight init, clarifying layer parameters). However, the anisotropic extension as delivered in this code contains a critical bug in the discriminator training loop and will not yield the intended 3-discriminator behaviour. Beyond that, the code defaults to a resize-convolution generator variant rather than the paper’s uniform-density transpose-convolution network, which may affect image quality. With corrections to the anisotropic loop and the upsampling syntax, the implementation can fully reproduce both isotropic and anisotropic claims.