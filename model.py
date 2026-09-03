"""
Kimi K3 from Scratch: KDA, Attention Residuals, and Stable LatentMoE

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - short_conv
def short_conv(x, w):
    """Causal depthwise conv: y[t,c] = sum_j w[j,c] * x[t-(K-1)+j, c].

    x: (T, d) sequence.  w: (K, d) per-channel kernel, w[K-1] = current token.
    Positions before the sequence start count as zeros.
    """
    T, d = x.shape
    K, d_w = w.shape

    if d != d_w:
        raise ValueError("x and w must have the same number of channels")

    y = np.zeros_like(x, dtype=np.result_type(x, w))

    for j in range(K):
        shift = K - 1 - j

        if shift == 0:
            y += w[j] * x
        elif shift < T:
            y[shift:] += w[j] * x[:T - shift]

    return y

# Step 2 - kda_qkv
def kda_qkv(x, params):
    """KDA projections: q,k = L2Norm(Swish(ShortConv(W x))), v = Swish(ShortConv(Wv x)).

    params: dict with Wq (d,dk), Wk (d,dk), Wv (d,dv), cq (K,dk), ck (K,dk), cv (K,dv).
    Returns (q, k, v).  L2Norm divides each row by sqrt(sum(row**2) + 1e-6).
    """
    def swish(u):
        return u * (1.0 / (1.0 + np.exp(-u)))

    def project(x, W, kernel):
        return swish(short_conv(x @ W, kernel))

    q = project(x, params["Wq"], params["cq"])
    k = project(x, params["Wk"], params["ck"])
    v = project(x, params["Wv"], params["cv"])

    q = q / np.sqrt(np.sum(q**2, axis=1, keepdims=True) + 1e-6)
    k = k / np.sqrt(np.sum(k**2, axis=1, keepdims=True) + 1e-6)

    return q, k, v

# Step 3 - kda_gates
def kda_gates(x, params):
    """Return (beta, z): write strength sigmoid(x@wb+bb), decay logits x@Wd1@Wd2+ba.

    params: wb (d,), bb scalar, Wd1 (d,r), Wd2 (r,dk), ba (dk,).
    beta: (T,) in (0,1).  z: (T, dk), unbounded.
    """
    logits = x @ params["wb"] + params["bb"]
    beta = 1.0 / (1.0 + np.exp(-logits))

    z = x @ params["Wd1"] @ params["Wd2"] + params["ba"]

    return beta, z

# Step 4 - lower_bounded_decay
def lower_bounded_decay(z, A, g_min=-5.0):
    """alpha = exp(g_min * sigmoid(exp(A) * z)), each entry in [exp(g_min), 1).

    z: (T, dk) decay logits.  A: scalar per-head log-scale.
    """
    scaled_z = np.exp(A) * z
    sigmoid = 1.0 / (1.0 + np.exp(-scaled_z))
    g = g_min * sigmoid
    alpha = np.exp(g)

    return alpha

# Step 5 - kda_state_update
def kda_state_update(S, k, v, alpha, beta):
    """One KDA step: (I - beta k k^T) @ diag(alpha) @ S + beta * outer(k, v).

    S: (dk, dv).  k: (dk,).  v: (dv,).  alpha: (dk,).  beta: scalar.
    """
    # Apply channel-wise decay first.
    S_decay = alpha[:, None] * S

    # Delta-rule erase:
    # (I - beta * k k^T) @ S_decay
    S_erase = S_decay - beta * np.outer(k, k) @ S_decay

    # Write the new value.
    S_new = S_erase + beta * np.outer(k, v)

    return S_new

# Step 6 - kda_recurrence
def kda_recurrence(q, k, v, alpha, beta, S0=None):
    """Run KDA token by token: update state, then read O[t] = S_t^T q[t].

    Returns (O, S_final) with O of shape (T, dv). S0 defaults to zeros; never
    mutate the caller's S0.
    """
    T, dk = q.shape
    dv = v.shape[1]

    if S0 is None:
        S = np.zeros((dk, dv), dtype=np.result_type(q, k, v, alpha, beta))
    else:
        S = np.array(S0, copy=True)

    O = np.zeros((T, dv), dtype=np.result_type(S, q))

    for t in range(T):
        S = kda_state_update(
            S,
            k[t],
            v[t],
            alpha[t],
            beta[t],
        )

        # Read only after the state update.
        O[t] = S.T @ q[t]

    return O, S

# Step 7 - cumulative_decay
def cumulative_decay(alpha):
    """Inclusive channel-wise cumulative product of alpha down the time axis.

    alpha: (C, dk) per-step retention factors -> Gamma: (C, dk).
    """
    return np.cumprod(alpha, axis=0)

# Step 8 - chunk_pseudo_values
def chunk_pseudo_values(k, v, alpha, beta, S0):
    """Solve (I + diag(beta) strict_tril(Khat Kcheck^T)) U = diag(beta)(V - Khat S0).

    Khat = k * Gamma, Kcheck = k / Gamma, Gamma = cumulative_decay(alpha).
    Returns U of shape (C, dv).
    """
    Gamma = cumulative_decay(alpha)

    # Scale keys by the cumulative decay factors.
    Khat = k * Gamma
    Kcheck = k / Gamma

    # Strictly lower-triangular intra-chunk interaction matrix.
    M = np.tril(Khat @ Kcheck.T, k=-1)

    C = k.shape[0]

    # (I + diag(beta) @ M) U = diag(beta) @ (V - Khat @ S0)
    lhs = np.eye(C, dtype=np.result_type(k, v, alpha, beta, S0)) + beta[:, None] * M
    rhs = beta[:, None] * (v - Khat @ S0)

    U = np.linalg.solve(lhs, rhs)

    return U

# Step 9 - kda_chunkwise
def kda_chunkwise(q, k, v, alpha, beta, chunk_size, S0=None):
    """Chunkwise-parallel KDA (Eq. 4): O_c = Qhat @ S + tril(Qhat Kcheck^T) @ U.

    State hand-off: S <- Gamma[-1][:,None] * (S + Kcheck^T U). Must equal
    kda_recurrence for every chunk size. Returns (O, S_final).
    """
    T, dk = q.shape
    dv = v.shape[1]

    if S0 is None:
        S = np.zeros(
            (dk, dv),
            dtype=np.result_type(q, k, v, alpha, beta),
        )
    else:
        S = np.array(S0, copy=True)

    O = np.zeros(
        (T, dv),
        dtype=np.result_type(q, k, v, alpha, beta, S),
    )

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)

        q_c = q[start:end]
        k_c = k[start:end]
        v_c = v[start:end]
        alpha_c = alpha[start:end]
        beta_c = beta[start:end]

        # Inclusive cumulative decay within the current chunk.
        Gamma = cumulative_decay(alpha_c)

        # Chunkwise transformed queries and keys.
        Qhat = q_c * Gamma
        Kcheck = k_c / Gamma

        # Corrected intra-chunk write values.
        U = chunk_pseudo_values(
            k_c,
            v_c,
            alpha_c,
            beta_c,
            S,
        )

        # Inter-chunk contribution plus causal intra-chunk contribution.
        intra = np.tril(Qhat @ Kcheck.T)
        O[start:end] = Qhat @ S + intra @ U

        # Carry the updated state to the next chunk.
        S = Gamma[-1][:, None] * (S + Kcheck.T @ U)

    return O, S

# Step 10 - kda_output_gate
def kda_output_gate(o, x, Wg, Wo):
    """y = (sigmoid(x @ Wg) * RMSNorm(o)) @ Wo, RMSNorm = o / sqrt(mean(o^2)+1e-6).

    o: (T, dv) recurrent outputs.  x: (T, d) layer input.  Returns (T, d).
    """
    gate_logits = x @ Wg
    gate = 1.0 / (1.0 + np.exp(-gate_logits))

    rms = np.sqrt(np.mean(o**2, axis=1, keepdims=True) + 1e-6)
    o_norm = o / rms

    return (gate * o_norm) @ Wo

# Step 11 - mla_compress_reconstruct
def mla_compress_reconstruct(x, Wc, Wk_up, Wv_up, n_heads):
    """c = x @ Wc; K = (c @ Wk_up).reshape(T, H, dh); V likewise.

    Returns (c, K, V) with shapes (T, r), (T, H, dh), (T, H, dh).
    """
    c = x @ Wc

    T = x.shape[0]
    total_dim = Wk_up.shape[1]

    if total_dim % n_heads != 0:
        raise ValueError("Wk_up output dimension must be divisible by n_heads")

    dh = total_dim // n_heads

    K = (c @ Wk_up).reshape(T, n_heads, dh)
    V = (c @ Wv_up).reshape(T, n_heads, dh)

    return c, K, V

# Step 12 - nope_attention
def nope_attention(x, Wq, Wc, Wk_up, Wv_up, n_heads):
    """Causal multi-head attention over MLA-reconstructed K,V - no positions.

    Q = (x @ Wq).reshape(T, H, dh); per head softmax(QK^T/sqrt(dh)) V with a
    causal mask; concatenate heads -> (T, H*dh).
    """
    T = x.shape[0]
    total_dim = Wq.shape[1]

    if total_dim % n_heads != 0:
        raise ValueError("Wq output dimension must be divisible by n_heads")

    dh = total_dim // n_heads

    # Query projection.
    Q = (x @ Wq).reshape(T, n_heads, dh)

    # Reconstruct MLA keys and values.
    _, K, V = mla_compress_reconstruct(
        x, Wc, Wk_up, Wv_up, n_heads
    )

    outputs = np.empty((T, n_heads, dh), dtype=np.result_type(Q, K, V))

    scale = np.sqrt(dh)
    causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)

    for h in range(n_heads):
        scores = (Q[:, h] @ K[:, h].T) / scale

        # Mask strictly future positions.
        scores = scores.copy()
        scores[causal_mask] = -np.inf

        # Numerically stable softmax.
        row_max = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - row_max)
        attention = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        outputs[:, h] = attention @ V[:, h]

    return outputs.reshape(T, n_heads * dh)

# Step 13 - mla_output_gate
def mla_output_gate(o, x, Wg, Wo):
    """y = (sigmoid(x @ Wg) * o) @ Wo - note: no RMSNorm here, unlike KDA's gate.

    o: (T, H*dh) attention output.  x: (T, d) layer input.  Returns (T, d).
    """
    gate_logits = x @ Wg
    gate = 1.0 / (1.0 + np.exp(-gate_logits))

    return (gate * o) @ Wo

# Step 14 - hybrid_schedule
def hybrid_schedule(n_repeats):
    """['KDA','KDA','KDA','MLA'] repeated n_repeats times, plus a final 'MLA'."""
    return ["KDA", "KDA", "KDA", "MLA"] * n_repeats + ["MLA"]

# Step 15 - attnres_weights
def attnres_weights(pseudo_q, sources):
    """Softmax over depth: w[i, t] prop. to exp(pseudo_q . RMSNorm(sources[i][t])).

    sources: list of n (T, d) arrays.  Returns (n, T); columns sum to 1.
    """
    # Stack sources along the depth dimension: (n, T, d).
    stacked = np.stack(sources, axis=0)

    # RMS-normalize each source row independently.
    rms = np.sqrt(np.mean(stacked**2, axis=2, keepdims=True) + 1e-6)
    normalized = stacked / rms

    # Compute one logit per source and token: (n, T).
    logits = np.einsum("d,ntd->nt", pseudo_q, normalized)

    # Stable softmax across sources for each token.
    logits = logits - np.max(logits, axis=0, keepdims=True)
    weights = np.exp(logits)
    weights /= np.sum(weights, axis=0, keepdims=True)

    return weights

# Step 16 - attnres_full
def attnres_full(pseudo_q, sources):
    """h[t] = sum_i attnres_weights(...)[i, t] * sources[i][t] (raw values).

    Returns (T, d).
    """
    weights = attnres_weights(pseudo_q, sources)

    # Stack raw source values: (n, T, d).
    values = np.stack(sources, axis=0)

    # Weight each source independently per token, then sum over depth.
    h = np.sum(weights[:, :, None] * values, axis=0)

    return h

# Step 17 - block_partial_sums
def block_partial_sums(layer_outputs):
    """Running sums of a block's layer outputs; entry i sums outputs 0..i.

    Returns a list of independent (T, d) arrays; last entry = block sum b_n.
    """
    partial_sums = []
    running = None

    for output in layer_outputs:
        if running is None:
            running = np.array(output, copy=True)
        else:
            running = running + output

        # Store an independent copy.
        partial_sums.append(np.array(running, copy=True))

    return partial_sums

# Step 18 - attnres_block
def attnres_block(pseudo_q, block_reps, partial):
    """Full AttnRes over [b_0..b_{n-1}] plus the current block's partial (if any).

    partial is None for the first layer of a block. Returns (T, d).
    """
    sources = list(block_reps)

    if partial is not None:
        sources.append(partial)

    return attnres_full(pseudo_q, sources)

# Step 19 - situ_glu
def situ_glu(x, Wg, Wu, beta1=4.0, beta2=25.0):
    """(softcap(x@Wg, b1) * sigmoid(x@Wg)) * softcap(x@Wu, b2), softcap = b*tanh(u/b).

    Use sigmoid(g) = 0.5*(1 + tanh(0.5*g)) to stay overflow-safe. |out| <= b1*b2.
    """
    g = x @ Wg
    u = x @ Wu

    # Smooth soft-cap.
    g_cap = beta1 * np.tanh(g / beta1)
    u_cap = beta2 * np.tanh(u / beta2)

    # Overflow-safe sigmoid.
    sigmoid_g = 0.5 * (1.0 + np.tanh(0.5 * g))

    return (g_cap * sigmoid_g) * u_cap

# Step 20 - route_topk
def route_topk(x, Wr, bias, k):
    """s = sigmoid(x @ Wr); top-k by s + bias (stable, descending);
    p = raw selected scores normalized per token. Returns (s, idx, p).
    """
    logits = x @ Wr
    s = 0.5 * (1.0 + np.tanh(0.5 * logits))

    # Select experts using the biased scores. Stable descending sort makes
    # ties resolve toward lower expert indices.
    biased_scores = s + bias
    order = np.argsort(-biased_scores, axis=1, kind="stable")
    idx = order[:, :k]

    # Mixture weights use the raw, unbiased scores.
    selected_scores = np.take_along_axis(s, idx, axis=1)
    p = selected_scores / np.sum(selected_scores, axis=1, keepdims=True)

    return s, idx, p

# Step 21 - routed_experts
def routed_experts(z, idx, p, experts):
    """u[t] = sum_j p[t,j] * situ_glu(z[t], *experts[idx[t,j]]).

    z: (T, l).  experts: list of (Wg, Wu) latent-width SiTU-GLUs. Returns (T, l).
    """
    T, l = z.shape
    k = idx.shape[1]

    u = np.zeros_like(z, dtype=np.result_type(z, p))

    for t in range(T):
        for j in range(k):
            expert_idx = idx[t, j]
            Wg, Wu = experts[expert_idx]

            # Process the individual latent token while preserving its
            # (1, l) shape for situ_glu.
            expert_out = situ_glu(z[t:t + 1], Wg, Wu)

            u[t] += p[t, j] * expert_out[0]

    return u

# Step 22 - stable_latent_moe
def stable_latent_moe(x, params):
    """y = sum_shared SiTU(x) + RMSNorm(routed_aggregate) @ Wup (Eq. 11).

    Route on full-width x; compute in latent width; RMSNorm(u) before Wup.
    """
    # Project tokens into the latent space.
    z = x @ params["Wdown"]

    # Route using the original full-width token representation.
    _, idx, p = route_topk(
        x,
        params["Wr"],
        params["bias"],
        params["k"],
    )

    # Run the selected latent experts and aggregate their outputs.
    u = routed_experts(
        z,
        idx,
        p,
        params["experts"],
    )

    # RMS-normalize the routed latent representation before up-projection.
    rms = np.sqrt(np.mean(u**2, axis=1, keepdims=True) + 1e-6)
    u_norm = u / rms

    # Up-project the normalized routed representation.
    y = u_norm @ params["Wup"]

    # Add the two shared full-width SiTU-GLU branches.
    for Wg_s, Wu_s in params["shared"]:
        y += situ_glu(x, Wg_s, Wu_s)

    return y

# Step 23 - topk_cutoffs
def topk_cutoffs(s, bias, k):
    """Top-(k+1) on s + bias: first k -> routes (m, k); (k+1)-th biased score
    -> cutoffs (m,). Returns (routes, cutoffs).
    """
    m, n = s.shape

    if k >= n:
        raise ValueError("k must be smaller than the number of experts")

    biased_scores = s + bias

    # Stable descending sort so ties are resolved toward lower expert indices.
    order = np.argsort(-biased_scores, axis=1, kind="stable")

    top_order = order[:, :k + 1]
    routes = top_order[:, :k]

    # The (k+1)-th biased score is each token's admission cutoff.
    cutoffs = np.take_along_axis(
        biased_scores,
        top_order[:, k:k + 1],
        axis=1,
    )[:, 0]

    return routes, cutoffs

# Step 24 - quantile_balance_update
def quantile_balance_update(s, bias, k):
    """QB (Eq. 14): bhat_j = -(the (q+1)-th largest of s[:, j] - cutoffs),
    q = m*k // n; return bhat - mean(bhat).
    """
    m, n = s.shape

    if (m * k) % n != 0:
        raise ValueError("m*k must be divisible by the number of experts")

    q = (m * k) // n

    # Get each token's admission cutoff from the current biased routing.
    _, cutoffs = topk_cutoffs(s, bias, k)

    # Raw score minus the biased cutoff.
    margins = s - cutoffs[:, None]

    # q-th index in the descending ordering corresponds to the
    # (q+1)-th largest margin. Partitioning -margins gives its
    # negative directly.
    bhat = np.partition(-margins, q, axis=0)[q]

    # Mean-centering applies the same shift to every expert and
    # therefore preserves all routing comparisons.
    return bhat - np.mean(bhat)

# Step 25 - histogram_quantile
def histogram_quantile(x, n_bins, lo, hi, q_frac):
    """Quantile from pooled bin counts; error <= (hi - lo) / n_bins.

    Return the right edge of the first bin whose cumulative count reaches
    q_frac * len(x).
    """
    counts, edges = np.histogram(x, bins=n_bins, range=(lo, hi))
    cumulative = np.cumsum(counts)

    target = q_frac * len(x)

    # Find the first bin whose cumulative count reaches the target.
    bin_idx = np.searchsorted(cumulative, target, side="left")

    # Keep the index within the available bins.
    bin_idx = min(bin_idx, n_bins - 1)

    return edges[bin_idx + 1]

# Step 26 - newton_schulz
def newton_schulz(G, n_iters=5):
    """Muon's Newton-Schulz orthogonalization (a,b,c = 3.4445, -4.7750, 2.0315).

    Normalize by the Frobenius norm (+1e-7), iterate the quintic, transpose
    handling for tall matrices. Singular values -> 1.
    """
    a, b, c = 3.4445, -4.7750, 2.0315

    # Work with a matrix having no more rows than columns.
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        G = G.T

    # Normalize by the Frobenius norm.
    X = G / (np.linalg.norm(G, ord="fro") + 1e-7)

    for _ in range(n_iters):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X

# Step 27 - per_head_muon
def per_head_muon(M, n_heads, n_iters=5):
    """Split M (d, H*dh) into H column blocks, newton_schulz each, re-concatenate."""
    d, total_dim = M.shape

    if total_dim % n_heads != 0:
        raise ValueError("M's column dimension must be divisible by n_heads")

    dh = total_dim // n_heads

    blocks = []
    for h in range(n_heads):
        start = h * dh
        end = (h + 1) * dh

        block = M[:, start:end]
        blocks.append(newton_schulz(block, n_iters=n_iters))

    return np.concatenate(blocks, axis=1)

# Step 28 - mini_k3_forward
def mini_k3_forward(tokens, seed):
    """Miniature Kimi K3 forward pass. Returns (T, 17) logits.

    Parameter creation order (all rng.normal(0.0, 0.3, size=...)):
      emb (17, 16)
      then for each layer of hybrid_schedule(1):
        wq_res (16,)
        KDA: Wq (16,8), Wk (16,8), Wv (16,8), cq (3,8), ck (3,8), cv (3,8),
             wb (16,), Wd1 (16,4), Wd2 (4,8), ba (8,), Wg (16,8), Wo (8,16)
             (bb = 0.0 and A = 0.0 are constants, not drawn)
        MLA: Wq (16,16), Wc (16,8), Wk_up (8,16), Wv_up (8,16),
             Wg (16,16), Wo (16,16)
        MoE: Wdown (16,8), Wup (8,16), Wr (16,8), bias = zeros(8), k=2,
             experts: 8 x (Wg (8,8), Wu (8,8)) drawn gate-then-up per expert,
             shared: 2 x (Wg (16,16), Wu (16,16)) drawn gate-then-up
      w_final (16,)
    """
    rng = np.random.default_rng(seed)

    def W(*shape):
        return rng.normal(0.0, 0.3, size=shape)

    # Fixed toy dimensions.
    vocab = 17
    d = 16
    dk = 8
    dv = 8
    H = 2
    latent_r = 8
    n_experts = 8
    top_k = 2
    conv_k = 3
    decay_rank = 4

    # Embedding is drawn first.
    emb = W(vocab, d)

    schedule = hybrid_schedule(1)
    layers = []

    # Draw all layer parameters in the exact required order.
    for layer_type in schedule:
        wq_res = W(d)

        if layer_type == "KDA":
            kda = {
                "Wq": W(d, dk),
                "Wk": W(d, dk),
                "Wv": W(d, dv),
                "cq": W(conv_k, dk),
                "ck": W(conv_k, dk),
                "cv": W(conv_k, dv),
                "wb": W(d),
                "bb": 0.0,
                "Wd1": W(d, decay_rank),
                "Wd2": W(decay_rank, dk),
                "ba": W(dk),
                "Wg": W(d, dv),
                "Wo": W(dv, d),
            }
            attention_params = kda

        else:  # MLA
            mla = {
                "Wq": W(d, H * dk),
                "Wc": W(d, latent_r),
                "Wk_up": W(latent_r, H * dk),
                "Wv_up": W(latent_r, H * dk),
                "Wg": W(d, H * dk),
                "Wo": W(H * dk, d),
            }
            attention_params = mla

        # MoE parameters.
        moe = {
            "Wdown": W(d, latent_r),
            "Wup": W(latent_r, d),
            "Wr": W(d, n_experts),
            "bias": np.zeros(n_experts),
            "k": top_k,
            "experts": [],
            "shared": [],
        }

        # Experts: gate projection first, then up projection.
        for _ in range(n_experts):
            Wg = W(latent_r, latent_r)
            Wu = W(latent_r, latent_r)
            moe["experts"].append((Wg, Wu))

        # Shared experts: gate projection first, then up projection.
        for _ in range(2):
            Wg = W(d, d)
            Wu = W(d, d)
            moe["shared"].append((Wg, Wu))

        layers.append(
            {
                "type": layer_type,
                "wq_res": wq_res,
                "attention": attention_params,
                "moe": moe,
            }
        )

    # Final Attention Residual pseudo-query is drawn last.
    w_final = W(d)

    tokens = np.asarray(tokens, dtype=int)
    sources = [emb[tokens]]

    for layer in layers:
        # Attention Residual over the token embedding and all prior layer outputs.
        h = attnres_full(layer["wq_res"], sources)

        if layer["type"] == "KDA":
            params = layer["attention"]

            q, k, v = kda_qkv(h, params)
            beta, z = kda_gates(h, params)

            # A = 0.0, so exp(A) = 1.
            alpha = lower_bounded_decay(z, 0.0)

            o, _ = kda_recurrence(q, k, v, alpha, beta)

            a = kda_output_gate(
                o,
                h,
                params["Wg"],
                params["Wo"],
            )

        else:  # MLA
            params = layer["attention"]

            o = nope_attention(
                h,
                params["Wq"],
                params["Wc"],
                params["Wk_up"],
                params["Wv_up"],
                n_heads=H,
            )

            a = mla_output_gate(
                o,
                h,
                params["Wg"],
                params["Wo"],
            )

        # Stable LatentMoE residual branch.
        f = a + stable_latent_moe(a, layer["moe"])
        sources.append(f)

    # Final global Attention Residual read.
    h_out = attnres_full(w_final, sources)

    # Tied output projection using the embedding matrix.
    logits = h_out @ emb.T

    return logits

