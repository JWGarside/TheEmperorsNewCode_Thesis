# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-training
**Analysis Date:** 2025-05-25

## Analysis Results

1. Paper summary and core claims  
• The paper “Generating 3-D structures from a 2-D slice with GAN-based dimensionality expansion (SliceGAN)” introduces a WGAN–GP architecture that learns from one (isotropic case) or three (anisotropic case) 2-D micrographs and synthesises arbitrarily large 3-D voxel volumes.  
• Key methodological points described in the text and Tables 1/Alg. 1:  
  – 3-D generator, 2-D discriminator; fake volumes are sliced into all l planes along x,y,z before being shown to D.  
  – Generator built only from transpose-convolutions with parameter set {k=4, s=2, p=2} (last layer p = 3) to guarantee uniform information density; input z has spatial size 4 to make the first layer overlap.  
  – WGAN-GP, λ = 10, critic_iters = 5, batch relation mG = 2 mD.  
  – During each discriminator step every slice in any 32-plane block must be seen once so the whole kernel tree is trained.  
  – Generator output uses channel-wise soft-max for one-hot n-phase data, or tanh for grey/colour.  

2. Implementation assessment  
• Execution flow (run_slicegan → networks → model):  
  – `run_slicegan.py` constructs nets with `slicegan_rc_nets`, then calls `model.train`.  
  – `preprocessing.batch` builds three 2-D datasets (x,y,z) exactly as in the paper.  
  – `networks.slicegan_rc_nets` creates a generator consisting of 5 transpose-conv layers followed by an **upsample + ordinary 3-D conv (`rcconv`)** instead of the sixth transpose-conv stated in the paper.  
    * Filter schedule hard-coded in run_slicegan (`gf=[32,1024,512,128,32,C]`) – differs from Table 1 (512,256,128,64).  
  – Training loop (`model.train`) instantiates three discriminators and performs a WGAN-GP update.  
    * Learning-rates, λ=10, critic_iters=5 match the paper.  
    * `D_batch_size` == `batch_size` == 8 (mG ≠ 2 mD).  
    * **Only the centre slice of each fake volume is fed to the discriminator** (`fake_data[:, :, l//2, :, :]`) while the paper states that *all* l slices should be used; for the generator step all l slices are indeed used (`reshape(l*batch,…)`).  
  – Uniform-density rule is respected for the first five layers (k=4,s=2,p=2) and last layer (p=3 ≥ k-s).  

3. Discrepancies  

| # | Description | Classification | Comment |
|---|-------------|----------------|---------|
|1 | Training code gives D only the middle slice, not the full 3 l slice set described in Algorithm 1 | **Critical** | Violates the paper’s core training strategy; may impair generator learning and edge-quality claims. |
|2 | Generator in code ends with Upsample + 3-D conv (`rcconv`); paper specifies a sixth transpose-conv | **Minor** | Still produces the stated output size; architectural change may affect output quality but not the concept. |
|3 | Filter sizes (gf) larger (1024/512/128/32) than Table 1 (512/256/128/64) | Minor | Capacity difference, not conceptual. |
|4 | mG = mD in code; paper recommends mG = 2 mD | Minor | Affects training balance. |
|5 | `slicegan_rc_nets` contains an apparent syntax error: `int(x.shape[2]-1,)` | Minor–Critical (runtime) | Would raise TypeError unless silently fixed; easy single-character fix. |
|6 | Paper emphasises “kernel overlap” by using z of spatial size 4; code uses `lz = 4` so behaviour matches | – | – |
|7 | Documentation points to `train.py`, but file is actually `model.py` | Cosmetic | Naming only. |

4. Overall reproducibility conclusion  
The repository contains all major components needed to retrain SliceGAN; preprocessing, generator/discriminator definitions and a WGAN-GP training loop are present and parameter values are close to those reported.  However, the way the discriminator is trained (single-slice instead of every slice) departs from the algorithm that underpins the main claim of uniform-quality volumes.  Together with a small architectural change and a potential syntax bug, this constitutes a **moderate barrier to exact reproduction** of the quantitative results shown in the paper, although the qualitative behaviour (3-D generation from 2-D data) can likely still be demonstrated after minor fixes.