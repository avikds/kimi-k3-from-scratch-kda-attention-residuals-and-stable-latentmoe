# Kimi K3 from Scratch: KDA, Attention Residuals, and Stable LatentMoE

Build every architectural innovation from the Kimi K3 technical report in numpy at toy scale: Kimi Delta Attention with lower-bounded decay and an exact chunkwise-parallel form, Gated MLA with NoPE, Attention Residuals over depth, Stable LatentMoE with SiTU-GLU and Quantile Balancing, and per-head Muon orthogonalization - then assemble a working mini K3 block stack.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** short_conv
- [x] **2.** kda_qkv
- [x] **3.** kda_gates
- [x] **4.** lower_bounded_decay
- [x] **5.** kda_state_update
- [x] **6.** kda_recurrence
- [x] **7.** cumulative_decay
- [x] **8.** chunk_pseudo_values
- [x] **9.** kda_chunkwise
- [x] **10.** kda_output_gate
- [x] **11.** mla_compress_reconstruct
- [x] **12.** nope_attention
- [x] **13.** mla_output_gate
- [x] **14.** hybrid_schedule
- [x] **15.** attnres_weights
- [x] **16.** attnres_full
- [x] **17.** block_partial_sums
- [x] **18.** attnres_block
- [x] **19.** situ_glu
- [x] **20.** route_topk
- [x] **21.** routed_experts
- [x] **22.** stable_latent_moe
- [x] **23.** topk_cutoffs
- [x] **24.** quantile_balance_update
- [x] **25.** histogram_quantile
- [x] **26.** newton_schulz
- [x] **27.** per_head_muon
- [x] **28.** mini_k3_forward

## Results

```
chunkwise == recurrence: True
alpha bounds ok: True
attnres full shape: (5, 8) block shape: (5, 8)
SiTU-GLU max |out| (bound 100): 100.0
QB loads before: [2, 5, 5, 4] after: [4, 4, 4, 4]
histogram quantile err <= binwidth: True
per-head norm ratio: 0.998 | full-matrix: 0.186
schedule: ['KDA', 'KDA', 'KDA', 'MLA', 'MLA']
logits shape: (6, 17) finite: True
causal end-to-end: True
next-token argmax per position: [7, 3, 14, 6, 14, 13]
```
