# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-training
**Analysis Date:** 2025-05-25

## Analysis Results

1. Paper summary and core claims  
The paper introduces SliceGAN, a generative‐adversarial network that synthesises statistically realistic 3D microstructural volumes from one (or three, for anisotropic samples) 2D training slice(s).  Its key innovations are:  
• A “slicing” procedure that feeds 2D cross‐sections of the 3D generator output to a 2D discriminator, enabling training on only 2D data.  
• Careful choice of transpose‐convolution parameters (kernel size k, stride s, padding p) to ensure a uniform information density and avoid edge artefacts.  
• An architecture that can generate arbitrarily large volumes by training on a small latent grid (4×4×4) and using convolutional layers with k = 4, s = 2, p = 2 (and a final layer with p = 3) to expand to 64³.  
• Demonstrated fidelity on diverse materials (two‐phase, three‐phase, colour, anisotropic), and statistical validation on electrochemical metrics.  

2. Implementation assessment  
– run_slicegan.py: parses arguments, sets up project directory, defines hyperparameters (img_size=64, z_channels=32, latent grid lz=4, 5 G‐layers / 6 D‐layers, k=s=4/2, p from [2,2,2,2,3] for G, [1,1,1,1,0] for D), instantiates networks via networks.slicegan_rc_nets, and calls model.train or util.test_img.  
– networks.py: two factory functions  
   • slicegan_nets(…): builds the transpose‐conv architecture matching the paper (5 ConvTranspose3d layers, BatchNorm, ReLU, softmax/tanh output).  
   • slicegan_rc_nets(…): an alternate “resize‐conv” variant (4 ConvTranspose3d layers + trilinear upsample + final 3³ Conv3d) that diverges from the paper and, as written, contains dimension‐calculation bugs.  
– model.py: implements a 3D generator G and three (or one, if isotropic) 2D discriminators D₁, D₂, D₃, WGAN‐GP loss, gradient‐penalty, and the training loop; uses noise of shape (batch, nz, 4,4,4).  
– preprocessing.py: reads 2D or 3D TIFF/PNG/JPG, sub‐samples into 64×64 patches, one‐hot encodes n‐phase data, and returns three PyTorch datasets (one per axis).  
– util.py: helper routines for directory creation, weight init, gradient penalty, ETA, plotting losses, slicing test outputs, and saving a .tif via util.test_img.  

3. Discrepancies (with classification)  
Critical  
1. Discriminators only see the single central slice per sample in D‐training  
   – In the paper, D is trained on all l=64 slices along x, y, and z (3 × l per volume). The code in train(…) does  
      fake_data = G(noise).detach()  
      fake_slice = fake_data[:, :, l//2, :, :]  
      …  
   so for every axis it only ever uses the middle slice, never permuting axes or iterating over d=1…l. This fundamentally breaks the slicing‐GAN principle.  
2. Default network factory is slicegan_rc_nets, not the conv‐transpose network described in the paper  
   – run_slicegan.py calls networks.slicegan_rc_nets by default. This implements a hybrid transpose‐conv + upsample + small 3³ Conv3d (“resize‐conv”), not the 5‐layer transpose‐conv chain with {k=4,s=2,p=2} (and p=3) that the paper says is crucial.  
3. Bug in slicegan_rc_nets upsample‐size calculation  
   – The line  
      size = (int(x.shape[2]-1,)*2, int(x.shape[3]-1,)*2, int(x.shape[3]-1,)*2)  
      uses a trailing comma inside int(…) and repeats shape[3] twice (instead of shape[4]), so it will raise a TypeError or produce incorrect volume dimensions.  

Minor  
4. The paper recommends m_G = 2·m_D (batch sizes) and slicing all planes, but code uses batch_size = D_batch_size = 8 and no separate m_G, m_D logic.  
5. No explicit random seed is set, so experiments are not bit‐wise reproducible.  
6. Hyperparameters (100 epochs, lr=1e-4, β₁=0.9, β₂=0.99) are hard-coded; paper does not fully specify them, but users must match to reproduce.  

Cosmetic  
7. README suggests using slicegan_nets (adjust networks.py) but does not mention slicegan_rc_nets; that can confuse users but does not prevent running (once the above bugs are fixed).  

4. Overall reproducibility conclusion  
While the code structure follows the paper’s high-level design—loading 2D data, one-hot encoding, latent grid of 4³, 3D generator + 2D discriminators, WGAN-GP loss—the implementation contains critical mismatches and outright bugs that prevent faithful reproduction of the SliceGAN method as described. In its current form the code will not train the discriminators on all slices, uses an unintended architecture by default, and will likely error on the faulty resize‐conv code path. Substantial debugging and correction (especially in the training loop’s slicing logic and network selection) are required before the core claims of the paper can be reproduced.