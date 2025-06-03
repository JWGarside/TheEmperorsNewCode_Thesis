# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-softmax
**Analysis Date:** 2025-05-25

## Analysis Results

1. Paper summary and core claims  
• “Generating 3-D structures from a 2-D slice with GAN-based dimensionality expansion” introduces SliceGAN, a GAN that learns from one (isotropic) or three (anisotropic) 2-D micrographs and synthesises statistically equivalent 3-D volumes.  
• Key methodological points given in the paper  
 – 3-D generator + 2-D discriminator(s); fake volumes are cut into l slices per axis and each slice is judged by D.  
 – Wasserstein-GAN with gradient penalty; discriminator trained nD times per generator update, mG = 2 mD.  
 – Generator uses exclusively transpose-convolutions that respect “uniform information-density” rules (k=4, s=2, p=2 on every layer except last which uses p≥k–s).  
 – Latent tensor has spatial size 4×4×4 in order to allow arbitrarily large outputs at inference.  
 – Output activation: soft-max for multi-phase (one-hot) data; tanh for grey/colour.  
 – Table 1 gives a reference architecture (Input 64×4×4×4 → 512 → 256 → 128 → 64 → 3).  
 – For anisotropic data, three independent discriminators are employed.  
Results in the paper show visually realistic reconstructions and matching statistical metrics for several materials.

2. Implementation assessment  
Execution flow (run_slicegan.py → networks.py → model.py):  
a) run_slicegan builds networks with networks.slicegan_rc_nets, then calls model.train.  
b) model.train  
 • Pre-processing reads either 2-D or 3-D tif files, cuts random l×l slices and packs them in three TensorDataset objects (one per axis).  
 • Three discriminators are instantiated; for isotropic data only the first is actually used in the backward pass.  
 • Training loop:  
  – Generate 3-D batch (shape = [D_batch_size, nz, 4,4,4]).  
  – For each axis reshape to 2-D slices and compute Wasserstein loss + gradient penalty.  
  – Generator updated every critic_iters (=5) steps.  
 • Utility functions provide gradient-penalty, ETA display, plotting, test-image saving, etc.  
c) Network definitions (networks.slicegan_rc_nets)  
 • Generator: five ConvTranspose3d layers followed by BatchNorm (on all but last), then a 3-D Conv (rcconv) after a trilinear up-sampling; final activation is torch.sigmoid.  
 • Discriminator: five 2-D Conv layers (stride 2, kernel 4, pad supplied by user).  

3. Discrepancies between paper description and code  

Critical  
C1. Generator forward pass bug – size calculation uses “int(x.shape[2]-1,)” (note the comma) which creates a tuple; calling int() on a tuple raises TypeError. As written, the provided code cannot run without manual correction.  
C2. Output activation for n-phase data is sigmoid, not soft-max as claimed. Channel probabilities are therefore not normalised and do not sum to 1, contradicting the methodology used in the paper for one-hot microstructures.  
C3. The paper’s table-1 architecture (512-256-128-64-3 filters) is not reproduced. The default hyper-parameter list in run_slicegan ([32, 1024, 512, 128, 32, 3]) yields a very different network.  

Minor  
M1. Generator/update ratio: code uses batch_size = D_batch_size = 8, critic_iters = 5; mG ≠ 2 mD as suggested in the algorithm description.  
M2. Three discriminator objects are always created; for isotropic data only one is used (extra memory, not method-breaking).  
M3. Final generator padding is p = 3 when k = 4, s = 2 (condition p ≥ k–s holds, but the paper states p = 2 everywhere except that last layer uses 3D periodicity rule).  
M4. Deprecated F.sigmoid is used (works up to PyTorch 1.9, but generates warnings).  

Cosmetic  
– Variable names differ (e.g. lz vs “spatial size 4”).  
– README still points to “train.py” (file is actually model.py).  

4. Overall reproducibility conclusion  
Although the training loop, slicing strategy and discriminator logic closely follow the SliceGAN algorithm described in the paper, two critical mismatches—(i) a syntactic error that prevents the generator from executing and (ii) the absence of a soft-max output for multi-phase data—mean that the published code cannot be run successfully to reproduce the results without modification. After correcting these points and adjusting the filter sizes to match Table 1, the remainder of the implementation appears faithful and should allow the paper’s claims to be reproduced.