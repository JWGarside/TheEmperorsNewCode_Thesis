# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-softmax
**Analysis Date:** 2025-05-25

## Analysis Results

1. Paper Summary and Core Claims  
The paper introduces SliceGAN, a generative adversarial network architecture that learns to synthesize arbitrarily large, high-fidelity 3D volumes of isotropic (and by extension, anisotropic) microstructures from one (or three) representative 2D slices.  Its core claims are:  
•  A novel 3D generator + 2D discriminator training scheme (“slice-and-discriminate”), enabling 3D synthesis from only 2D training data.  
•  A set of design rules (kernel size k, stride s, padding p) for transpose convolution layers that guarantees uniform information density (mitigating edge artifacts).  
•  Demonstration on diverse materials, with quantitative validation (two-point correlations, tortuosity, triple-phase boundary density) matching real 3D datasets.  
•  Fast generation (10^8 voxels in seconds) once trained.  

2. Implementation Assessment  
The repository is structured around:  
•  run_slicegan.py – user script: specifies project name, data paths, image type, network hyperparameters (number of layers, filter maps, kernel/stride/padding arrays), and then either trains or samples.  
•  slicegan/networks.py – two factory functions:  
   – slicegan_nets: pure ConvTranspose3d generator per paper  
   – slicegan_rc_nets: an alternative “resize-convolution” variant  
•  slicegan/model.py – WGAN-GP training loop:  
   – Data slicing: builds three PyTorch DataLoaders (x, y, z slices) via slicegan/preprocessing.py  
   – Generator noise of shape (batch, nz, lz, lz, lz), with lz=4  
   – 3 discriminators (one per axis) or shared for isotropic  
   – Critic (D) updates: D(real) – D(fake) + λ·GP, with lr=1e-4, β1=0.9, β2=0.99, λ=10, 5 D-steps per G-step  
   – Generator updates: –D(fake)  
   – Saves generator and (last) discriminator state every 25 iterations  
•  slicegan/preprocessing.py – loads 2D or 3D TIFF/PNG/JPG, one-hot encodes n-phase data, extracts random l×l patches (32 × 900 per epoch per orientation), wraps as TensorDataset  
•  slicegan/util.py – utilities: folder creation, weight init (unused), gradient-penalty calc, plotting slices/loss curves, test_img for sampling volumes with optional periodicity  
•  raytrace.py – optional 3D visualization (not needed to replicate core results)  

Hyperparameters largely mirror the paper:  kernel size = 4, stride = 2, padding = [2,2,2,2,3], WGAN-GP λ=10, critic_iters=5, Adam(lr=1e-4, β1=0.9, β2=0.99), latent spatial size lz=4.  The slicing‐and‐permutation logic corresponds exactly to the 3l 2D slices described.  

3. Discrepancies  

Critical  
•  By default run_slicegan.py calls slicegan_rc_nets (resize-convolution variant) instead of the pure transpose-convolution architecture described in the paper. The paper explicitly warns that resize-convolution has different memory/quality trade-offs.  
•  The provided `slicegan_rc_nets` code contains a syntactic/logic error in the `size = (int(x.shape[2]-1,)*2, …)` line, which will raise an exception and prevent the generator from running without manual correction.  

Minor  
•  The code uses independent sigmoid activations on each output channel for n-phase data, rather than a softmax across channels as stated in the paper.  In practice they then choose arg-max for visualization, but the probability normalization differs.  
•  The default generator filter counts (gf = [32,1024,512,128,32,3]) and latent channel size (nz = 32) in run_slicegan differ from the paper’s Table 1 (which uses 64 latent channels and [512,256,128,64,3] map counts).  (These are user-tunable but the defaults do not match.)  
•  The gradient penalty is computed only on the first `batch_size` of the `l*batch_size` fake slices instead of over all fake slices, deviating from the usual WGAN-GP formulation.  
•  The paper suggests showing `mG = 2 mD` fake volumes per update to rebalance the higher number of fake slices; the code uses `mG = mD = 8` instead.  
•  The training script saves only the last discriminator (D_z) state, not all three discriminators; resuming or analyzing D_x, D_y states is impossible.  
•  `util.mkdr` prompts for console input if a project directory exists, blocking noninteractive runs.  
•  No automatic download/preparation of the example NMC tomography data; users must manually place the training `.tif`.  

Cosmetic  
•  Minor naming mismatches (e.g. parameter comments refer to `train.py`).  
•  The README does not mention the two network variants or the default “rc” net path.  

4. Overall Reproducibility Conclusion  
The code repository captures the fundamental SliceGAN idea—training a 3D generator from 2D slices with a WGAN-GP framework and three discriminators—but in its out-of-the-box form it does not faithfully implement the exact network architecture of the paper (uses an alternate “rc” variant), contains a bug that prevents the generator from running, and diverges in several hyperparameter defaults (latent size, filter maps, gating function).  With the following minimal fixes and clarifications, one could reproduce the core results:

•  Fix the `size = …` syntax in `slicegan_rc_nets` or switch to `slicegan_nets` and ensure run_slicegan imports the pure transpose-conv variant.  
•  Align `gf`/`gk`/`gp` defaults in run_slicegan with Table 1 values from the paper (e.g. z_channels=64, gf=[64,512,256,128,64, nphase]).  
•  Optionally change final activation to softmax for n-phase data.  
•  Provide or script sample data download (e.g. NMC tomography) and a noninteractive folder-creation flag.  

Once these adjustments are made, the training loop, slicing, loss functions, and sampling utilities should reproduce the quantitative metrics and visual fidelity reported.  As delivered, the code is close to a working implementation but requires small bug fixes and parameter alignment to fully reproduce the core claims.