# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-inputvector
**Analysis Date:** 2025-05-25

## Analysis Results

1. Brief paper summary and core claims  
• The paper introduces SliceGAN, a GAN architecture that “expands” a single 2-D micrograph (or three orthogonal 2-D micrographs for anisotropic cases) into statistically consistent 3-D volumes.  
• Key ingredients described in the manuscript  
  – 3-D generator, 2-D discriminator(s); fake volumes are sliced into 2-D planes before being passed to D.  
  – Wasserstein–GP loss with λ = 10; critic-to-generator update ratio nD = 5.  
  – Generator built only from transpose-convolutions with the parameter triplet {k=4, s=2, p=2} (except a last layer with p ≥ k-s) to guarantee uniform “information density” throughout the volume.  
  – Latent seed has a spatial size of 4×4×4 during training to avoid edge artefacts and to keep the generator fully convolutional (hence able to output arbitrary-sized volumes at test time).  
  – Soft-max output for multi-phase (one-hot) data, tanh for gray/colour.  
  – mG = 2 mD so that the generator sees twice as many gradient steps as the discriminator because each generated sample is sliced into l planes.  
  – Typical network (Table 1) - channels: 512-256-128-64-3 for G and the mirrored 2-D D.

2. Implementation assessment  
The provided code follows the overall SliceGAN workflow:

• Data handling (preprocessing.py) converts a 3-D tif file to large batches of random 2-D one-hot slices.  
• During training (model.py) the generator output (B×C×L×L×L) is permuted to B·L 2-D slices before being sent to each of the three discriminators. Gradient-penalty and Wasserstein loss are implemented exactly as in the paper.  
• run_slicegan.py builds hyper-parameter lists that match the paper defaults (k=4, s=2, p mostly = 2).  
• networks.py contains two generator/discriminator factories; run_slicegan.py uses the *slicegan_rc_nets* version which still consists of transpose-convolutions followed by a 3×3×3 “rcconv” and soft-max – conceptually compatible with the manuscript.

3. Discrepancies

| # | Item | Paper | Code | Severity |
|---|------|-------|------|----------|
|1| Spatial size of latent seed | 4×4×4 during **training** | lz = 1 (model.py) → noise shape (B, nz, 1,1,1) | **Critical** (re-introduces information-density problem the paper claims to solve) |
|2| Generator class used | Plain transpose-conv net (Table 1) | slicegan_rc_nets adds Upsample + 3×3×3 conv; moreover the *size* line contains a syntax error: `int(x.shape[2]-1,)` which raises `TypeError`. | **Critical** (code cannot execute as released) |
|3| mG = 2 mD | Explicitly required | Both batch sizes are hard-coded to 8; generator therefore receives the same number of gradient steps as D (while each fake sample still produces 64 slices). | Minor |
|4| Channel sizes | 512-256-128-64-3 | 1024-512-128-32-nc | Minor |
|5| Learning rate & β | Not specified in text; default WGAN-GP often uses 1e-4/1e-4 | Code uses 1e-4 (ok) but β1 = 0.9, β2 = 0.99 (paper not explicit) | Cosmetic |
|6| Anisotropic training loop | Described in Supp.-Alg. A (3 separate D’s with different real images) | Code always builds three discriminators but, when `isotropic=True`, forcibly re-uses D[0] for all axes (fine). However the convenience wrapper to load three *different* images is not implemented in run_slicegan.py | Minor |

4. Overall reproducibility conclusion  
The repo reproduces many high-level elements of SliceGAN (one-hot preprocessing, 3-D→2-D slicing, WGAN-GP training). However two critical mismatches prevent straightforward replication of the paper’s results:

1. The latent vector is still 1×1×1 during training, contradicting the paper’s solution for edge artefacts.  
2. The generator class used by the example script contains a syntactic error that stops execution and, even if fixed, differs architecturally from the configuration reported in the manuscript.

Until these two issues are corrected, the code as supplied is not directly runnable and therefore the paper’s main claims cannot be reproduced “out-of-the-box”. With minor edits (correct the `size=` line and change `lz` from 1 to 4) the implementation should match the manuscript closely and is likely to reproduce the reported behaviour.