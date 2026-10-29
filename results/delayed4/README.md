# Idea 2: Delayed Memory Injection Training Log (w2_mem_delayed4)

- **Config**: `configs/w2_mem_delayed4.yaml` ($\Delta = 4$, Layer 1 query $\to$ Layer 5 residual injection before attention)
- **Parameters**: 369,777,920 non-embedding parameters (strictly matched to `w2_mem_super1`)
- **Cluster**: Ada Plafnet (`plafnet2` partition, `gnode118`, 1× RTX 3090)
- **Job ID**: 40052
- **Training Trajectory (Steps 1 → 90 / 47.19M Tokens)**:
  - Step 1: loss = 10.60, grad_norm = 14.35
  - Step 10: loss = 8.926, grad_norm = 1.889
  - Step 20: loss = 8.133, grad_norm = 1.838
  - Step 30: loss = 7.429, grad_norm = 0.890
  - Step 40: loss = 7.141, grad_norm = 2.681
  - Step 50: loss = 6.855, grad_norm = 1.168
  - Step 60: loss = 6.598, grad_norm = 1.545
  - Step 70: loss = 6.538, grad_norm = 4.005
  - Step 80: loss = 6.380, grad_norm = 1.249
  - Step 90: loss = 6.212, grad_norm = 1.599
- **Status**: Stable convergence; terminated at step 90.
