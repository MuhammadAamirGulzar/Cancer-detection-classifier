"""
Verify the tiled-ALiBi memory patch in surgen_titan.py is mathematically
equivalent to stock TITAN.

The patch replaces TITAN's dense (1, num_heads, N+1, N+1) attention-bias
tensor with a lazy descriptor plus attention tiled over query rows. Because
each tile computes COMPLETE score rows over all keys, the softmax is over the
full key set and the result should match stock up to floating-point
reassociation - it is a memory-layout change, not an approximation.

This script checks that claim on three levels:

  1. Per-head ALiBi slopes match TITAN's internal get_slopes exactly.
  2. The lazy bias, densified, matches stock get_alibi() elementwise.
  3. End-to-end slide embeddings from stock vs patched agree.

It then optionally benchmarks a large synthetic slide that stock TITAN cannot
process at all, to confirm the patch delivers what it is for.

Run on the server (where the model + GPU live):

    python verify_titan_patch.py
    python verify_titan_patch.py --large 22500     # also benchmark a big slide
"""

import argparse
import sys
import time

import numpy as np
import torch

import surgen_titan as st


# =============================================================================
# Synthetic slide construction
# =============================================================================

def build_synthetic_slide(n_tokens: int, patch_size_lv0: int = 1024, seed: int = 0):
    """
    Build a synthetic slide with `n_tokens` patches on a roughly square grid,
    with a random subset of cells left empty so the background mask is
    exercised (real slides are never a full rectangle of tissue).

    Returns (features (N,768) float32, coords (N,2) int64, patch_size_lv0).
    Every coordinate maps to a distinct grid cell, matching the invariant
    encode_slide asserts before calling TITAN.
    """
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_tokens / 0.7)))
    rows, cols = np.meshgrid(np.arange(side), np.arange(side), indexing='ij')
    cells = np.stack([rows.ravel(), cols.ravel()], axis=1)

    keep = rng.permutation(len(cells))[:n_tokens]
    cells = cells[np.sort(keep)]

    coords = (cells * patch_size_lv0).astype(np.int64)
    feats = rng.standard_normal((len(cells), 768)).astype(np.float32)
    return feats, coords, patch_size_lv0


def encode(model, feats, coords, patch_size_lv0, device):
    f = torch.from_numpy(feats).unsqueeze(0).to(device)
    c = torch.from_numpy(coords).unsqueeze(0).to(device)
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == 'cuda')):
            emb = model.encode_slide_from_patch_features(f, c, patch_size_lv0)
    return emb.float().cpu()


# =============================================================================
# Checks
# =============================================================================

def check_slopes(vision_encoder) -> bool:
    """_alibi_slopes must reproduce TITAN's internal get_slopes closure."""
    print("\n[1/3] ALiBi slopes")
    num_heads = vision_encoder.num_heads

    # Stock slopes are only reachable through get_alibi's dense output: for a
    # 1x2 grid the distance between the two points is exactly 1, so
    # all_bias[0, h, 1, 2] == -slopes[h].
    #
    # Stock stores them in a float32 tensor, so compare in float32: our
    # _alibi_slopes returns Python floats (float64) and the float64-vs-float32
    # gap is ~1e-8 pure rounding, not a real discrepancy.
    stock_bias = vision_encoder._titan_orig_get_alibi(1, 2, None)
    stock = -stock_bias[0, :, 1, 2].double()
    ours = torch.tensor(st._alibi_slopes(num_heads), dtype=torch.float32).double()

    max_diff = (stock - ours).abs().max().item()
    ok = max_diff < 1e-6
    print(f"      num_heads={num_heads}  max|stock - ours| = {max_diff:.3e}  (float32)")
    print(f"      {'PASS' if ok else 'FAIL'}")
    return ok


def check_dense_bias(model, feats, coords, patch_size_lv0, device) -> bool:
    """The densified lazy bias must equal stock get_alibi() elementwise."""
    print("\n[2/3] Dense bias reconstruction")
    ve = model.vision_encoder
    vt_mod = sys.modules[type(ve).__module__]

    f = torch.from_numpy(feats).unsqueeze(0).to(device)
    c = torch.from_numpy(coords).unsqueeze(0).to(device)
    x, _, bg_mask = vt_mod.preprocess_features(f, c, patch_size_lv0)
    _, _, w, h = x.shape

    stock = ve._titan_orig_get_alibi(w, h, bg_mask).float().cpu()
    lazy = st._get_alibi_lazy(ve, w, h, bg_mask)

    P = lazy.points.shape[0]
    dist = torch.cdist(lazy.points, lazy.points,
                       compute_mode='donot_use_mm_for_euclid_dist').cpu()
    ours = torch.zeros(1, ve.num_heads, P + 1, P + 1)
    for head in range(ve.num_heads):
        ours[0, head, 1:, 1:] = -lazy.slopes[head].cpu() * dist

    if stock.shape != ours.shape:
        print(f"      SHAPE MISMATCH stock={tuple(stock.shape)} ours={tuple(ours.shape)}")
        print("      FAIL")
        return False

    max_diff = (stock - ours).abs().max().item()
    ok = max_diff < 1e-3
    print(f"      grid={w}x{h}  tokens={P}  shape={tuple(stock.shape)}")
    print(f"      max|stock - ours| = {max_diff:.3e}")
    print(f"      {'PASS' if ok else 'FAIL'}")
    return ok


def check_embeddings(model, feats, coords, patch_size_lv0, device, chunk) -> bool:
    """End-to-end: stock vs patched slide embeddings must agree."""
    print("\n[3/3] End-to-end slide embedding")

    st._restore_titan_memory_patch(model)
    t0 = time.time()
    emb_stock = encode(model, feats, coords, patch_size_lv0, device)
    t_stock = time.time() - t0

    st._apply_titan_memory_patch(model, chunk)
    t0 = time.time()
    emb_patched = encode(model, feats, coords, patch_size_lv0, device)
    t_patched = time.time() - t0

    cos = torch.nn.functional.cosine_similarity(emb_stock, emb_patched, dim=-1).item()
    max_diff = (emb_stock - emb_patched).abs().max().item()
    rel = max_diff / emb_stock.abs().max().item()

    ok = cos > 0.9999 and rel < 1e-2
    print(f"      shape={tuple(emb_stock.shape)}  tokens={len(feats)}")
    print(f"      cosine similarity = {cos:.8f}")
    print(f"      max|diff| = {max_diff:.3e}   (relative {rel:.3e})")
    print(f"      time: stock {t_stock:.2f}s  ->  patched {t_patched:.2f}s")
    print(f"      {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark_large(model, n_tokens, patch_size_lv0, device, chunk):
    """Encode a slide far beyond what stock TITAN can fit, patched only."""
    print(f"\n[bench] Large slide: {n_tokens} tokens (stock TITAN cannot fit this)")
    dense_gb = model.vision_encoder.num_heads * (n_tokens + 1) ** 2 * 4 / 1024 ** 3
    print(f"        stock dense bias would need {dense_gb:.1f} GB for the bias alone")

    feats, coords, psz = build_synthetic_slide(n_tokens, patch_size_lv0, seed=1)
    st._apply_titan_memory_patch(model, chunk)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    emb = encode(model, feats, coords, psz, device)
    elapsed = time.time() - t0

    print(f"        OK: embedding {tuple(emb.shape)} in {elapsed:.1f}s")
    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        print(f"        peak GPU memory: {peak:.2f} GB  (chunk={chunk})")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', type=int, default=1200,
                    help='token count for the equivalence tests (must fit stock TITAN)')
    ap.add_argument('--large', type=int, default=0,
                    help='also benchmark a slide of this many tokens (patched only)')
    ap.add_argument('--chunk', type=int, default=st.ALIBI_CHUNK)
    ap.add_argument('--device', default=st.DEVICE)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Device: {device}   chunk: {args.chunk}")

    # Load WITHOUT the patch so the stock path is available as a reference.
    extractor = st.TITANExtractor(device=str(device), hf_token=st.HF_TOKEN,
                                  apply_memory_patch=False)
    model = extractor.titan

    # Install once to capture the originals as _titan_orig_*, then immediately
    # restore, so both paths are reachable for comparison below.
    st._apply_titan_memory_patch(model, args.chunk)
    st._restore_titan_memory_patch(model)

    # Authoritative architecture facts, read off the loaded model (not the
    # config defaults) - these are the numbers to quote in a methods section.
    ve = model.vision_encoder
    print(f"\nSlide encoder: embed_dim={ve.embed_dim}  "
          f"depth={len(ve.blocks.modules_list)}  num_heads={ve.num_heads}  "
          f"pos_encode={ve.pos_encode_type}")

    feats, coords, psz = build_synthetic_slide(args.tokens)

    results = [
        check_slopes(model.vision_encoder),
        check_dense_bias(model, feats, coords, psz, device),
        check_embeddings(model, feats, coords, psz, device, args.chunk),
    ]

    if args.large:
        benchmark_large(model, args.large, psz, device, args.chunk)

    print("\n" + "=" * 70)
    if all(results):
        print("ALL CHECKS PASSED - the patch is equivalent to stock TITAN.")
        print("Safe to run the full pipeline with apply_memory_patch=True.")
    else:
        print("SOME CHECKS FAILED - do NOT use the patch for production runs.")
        sys.exit(1)
    print("=" * 70)


if __name__ == '__main__':
    main()
