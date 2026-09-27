import numpy as np
P = "/ssd_scratch/sparsh.bhartia/morbius/probe"
l1 = np.load(f"{P}/dump_run1/zero-none_token_loss.npy").astype(np.float64).ravel()
l2 = np.load(f"{P}/dump_run2/zero-none_token_loss.npy").astype(np.float64).ravel()
n = min(len(l1), len(l2)); l1, l2 = l1[:n], l2[:n]
print(f"tokens {n:,}  mean loss run1 {l1.mean():.4f} run2 {l2.mean():.4f} gap {np.mean(l2-l1):+.4f}")
for layer in (1, 11):
    for who in ("run1", "run2"):
        s = np.load(f"{P}/dump_{who}/zero-none_sink_mem{layer}.npy").astype(np.float64).ravel()[:n]
        q = np.quantile(s, np.linspace(0, 1, 11))
        idx = np.clip(np.searchsorted(q, s, side="right") - 1, 0, 9)
        print(f"\n== bucket by {who} Sink at memory layer {layer} (decile: sink range | run1 loss | run2 loss | gap 2-1)")
        for b in range(10):
            m = idx == b
            print(f"  d{b}: {q[b]:.3f}-{q[b+1]:.3f} | {l1[m].mean():.3f} | {l2[m].mean():.3f} | {np.mean(l2[m]-l1[m]):+.4f}")
# difficulty control: bucket by run1 loss itself
q = np.quantile(l1, np.linspace(0, 1, 11)); idx = np.clip(np.searchsorted(q, l1, side="right") - 1, 0, 9)
print("\n== bucket by run1 token loss (control for difficulty): range | gap 2-1")
for b in range(10):
    m = idx == b
    print(f"  d{b}: {q[b]:.2f}-{q[b+1]:.2f} | {np.mean(l2[m]-l1[m]):+.4f}")
s1 = np.load(f"{P}/dump_run1/zero-none_sink_mem1.npy").astype(np.float64).ravel()[:n]
print("\ncorr(run1 sink L1, run1 loss) = %.3f ; corr(run1 sink L1, gap) = %.3f" % (np.corrcoef(s1, l1)[0,1], np.corrcoef(s1, l2-l1)[0,1]))
