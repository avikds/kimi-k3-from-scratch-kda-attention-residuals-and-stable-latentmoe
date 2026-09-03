"""
Kimi K3 from Scratch: KDA, Attention Residuals, and Stable LatentMoE scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Mini Kimi K3: KDA + Gated MLA + Attention Residuals + Stable LatentMoE."""

import numpy as np


def main():
    rng = np.random.default_rng(0)

    # --- KDA: chunkwise form must equal the recurrence ---
    T, dk, dv = 12, 4, 3
    q = rng.normal(size=(T, dk))
    k = rng.normal(size=(T, dk))
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    v = rng.normal(size=(T, dv))
    alpha = lower_bounded_decay(rng.normal(size=(T, dk)), 0.0)
    beta = 1.0 / (1.0 + np.exp(-rng.normal(size=T)))
    O_rec, S_rec = kda_recurrence(q, k, v, alpha, beta)
    O_ch, S_ch = kda_chunkwise(q, k, v, alpha, beta, chunk_size=4)
    print("chunkwise == recurrence:", bool(np.allclose(O_ch, O_rec, atol=1e-9)))
    print("alpha bounds ok:", bool((alpha >= np.exp(-5.0) - 1e-12).all() and (alpha < 1.0).all()))

    # --- Attention Residuals: full vs block source counts ---
    srcs = [rng.normal(size=(5, 8)) for _ in range(6)]
    pq = rng.normal(size=8)
    h_full = attnres_full(pq, srcs)
    partials = block_partial_sums(srcs[3:])
    h_block = attnres_block(pq, [srcs[0], srcs[1] + srcs[2]], partials[-1])
    print("attnres full shape:", h_full.shape, "block shape:", h_block.shape)

    # --- SiTU-GLU boundedness ---
    big = rng.normal(size=(10, 6)) * 1000
    out = situ_glu(big, rng.normal(size=(6, 5)), rng.normal(size=(6, 5)))
    print("SiTU-GLU max |out| (bound 100):", round(float(np.abs(out).max()), 2))

    # --- Quantile Balancing on the demo batch ---
    s = 1.0 / (1.0 + np.exp(-np.random.default_rng(12).normal(size=(16, 4)) * 2))
    before = np.bincount(np.argsort(-s, axis=1)[:, :1].ravel(), minlength=4)
    b1 = quantile_balance_update(s, np.zeros(4), 1)
    after = np.bincount(np.argsort(-(s + b1), axis=1)[:, :1].ravel(), minlength=4)
    print("QB loads before:", before.tolist(), "after:", after.tolist())
    margins = rng.normal(size=4000)
    est = histogram_quantile(margins, 200, -5.0, 5.0, 0.75)
    print("histogram quantile err <= binwidth:",
          bool(abs(est - np.quantile(margins, 0.75)) <= 10.0 / 200 + 1e-12))

    # --- Per-head Muon ---
    G = rng.normal(size=(16, 8))
    M = G.copy(); M[:, :4] *= 100.0
    ph = per_head_muon(M, 2, n_iters=3)
    fu = newton_schulz(M, n_iters=3)
    print("per-head norm ratio:",
          round(float(np.linalg.norm(ph[:, 4:]) / np.linalg.norm(ph[:, :4])), 3),
          "| full-matrix:",
          round(float(np.linalg.norm(fu[:, 4:]) / np.linalg.norm(fu[:, :4])), 3))

    # --- The mini model, end to end ---
    print("schedule:", hybrid_schedule(1))
    tokens = [1, 5, 3, 7, 2, 0]
    logits = mini_k3_forward(tokens, seed=0)
    print("logits shape:", logits.shape, "finite:", bool(np.isfinite(logits).all()))
    edited = mini_k3_forward([1, 5, 3, 7, 2, 9], seed=0)
    print("causal end-to-end:", bool(np.allclose(logits[:5], edited[:5])))
    print("next-token argmax per position:", np.argmax(logits, axis=1).tolist())


if __name__ == "__main__":
    main()

