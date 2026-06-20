"""Extract bloc97 Anime4K v4.1 Upscale_GAN family weights from upstream
GLSL into .kaizenw blobs.

Source (MIT, bloc97 2019-2021):
  bloc97/Anime4K upstream GLSL / glsl\\Upscale\\Anime4K_Upscale_GAN_x2_S.glsl
  bloc97/Anime4K upstream GLSL / glsl\\Upscale\\Anime4K_Upscale_GAN_x2_M.glsl
  (x3 / x4 tiers exist but not yet covered by this extractor.)

Architecture summary (verified via topology audit 2026-06-01):
  - DenseNet-style: 1x1 fan-in 'aggregate' convs read all prior layer outputs
    expanded via CReLU (2x width per source).
  - 3x3 body convs read a SINGLE prior aggregate, CReLU-expanded (2x width).
  - Final tail Conv-3x3x3x16 reads MAIN (bilinear) + 2 conv0ups siblings
    via L-style 2-source CReLU layout, and adds the MAIN bilinear at the end.
  - No denoise variants in this family (single-objective GAN-trained).

x2_S pass table (17 weighted convs):
  L1  Conv-4x3x3x3       head      MAIN                          (no CReLU)
  L2  Conv-4x3x3x8       body      conv2d_tf                     (CReLU)
  L3  Conv-4x3x3x8       body      conv2d_tf                     (CReLU)
  L4  Conv-4x1x1x24      agg/3-src conv2d_tf + _2_tf + _1_tf     (CReLU, alternating)
  L5  Conv-4x3x3x8       body      conv2d_3_tf                   (CReLU)
  L6  Conv-4x3x3x8       body      conv2d_3_tf                   (CReLU)
  L7  Conv-4x1x1x32      agg/4-src conv2d_3_tf + _5_tf + _1_tf + _4_tf
  L8  Conv-4x3x3x8       body      conv2d_6_tf                   (CReLU)
  L9  Conv-4x3x3x8       body      conv2d_6_tf                   (CReLU)
  L10 Conv-4x1x1x40      agg/5-src conv2d_6_tf + _8_tf + _1_tf + _4_tf + _7_tf
  L11 Conv-4x3x3x8       body      conv2d_9_tf                   (CReLU)
  L12 Conv-4x3x3x8       body      conv2d_9_tf                   (CReLU)
  L13 Conv-4x1x1x48      agg/6-src conv2d_9_tf + _11_tf + _1_tf + _4_tf + _7_tf + _10_tf
  L14 Conv-4x3x3x8       body      conv2d_12_tf                  (CReLU)
  L15 Conv-4x1x1x56      tail/7-src conv2d_12_tf + _11_tf + _1_tf + _4_tf + _7_tf + _10_tf + _13_tf
  L16 Conv-4x1x1x56      tail/7-src same 7 binds (sibling)
  L17 Conv-3x3x3x16      final     MAIN + conv0ups + conv0ups1   (2 conv srcs, L-style CReLU)

Two distinct CReLU layout conventions appear in the same shader file:
  - 3x3 body convs (single source): go_0=pos, go_1=neg -> ic 0..3 / 4..7
  - 1x1 aggregates (multi source):  g_idx// 2 = source, g_idx % 2 = polarity ->
                                    ic_base = g_idx * 4 (alternating per source)
  - 3x3 final tail (2 sources):     g + 2*p layout (L-style)
                                    go_0 src0 pos -> ic 0..3, go_1 src1 pos -> ic 8..11,
                                    go_2 src0 neg -> ic 4..7, go_3 src1 neg -> ic 12..15

Header (6 x uint32 = 24 bytes; NEW KaizenwKind for this arch):
  magic   = 0x4B575A4D ('KWZM')
  version = 1
  kind    = KaizenwKind::Bloc97UpscaleGanS = 21  (NEW)
          | KaizenwKind::Bloc97UpscaleGanM = 22  (NEW)
  num_conv = 17 (S) | 23 (M)
  num_feat = 4
  scale    = 2

Output paths:
  bloc97/Anime4K upstream GLSL / glsl\\Upscale\\ani4k_upscale_gan_s_x2.kaizenw
  bloc97/Anime4K upstream GLSL / glsl\\Upscale\\ani4k_upscale_gan_m_x2.kaizenw

Usage:
  python extract_anime4k_upscale_gan_glsl.py        # both s + m
  python extract_anime4k_upscale_gan_glsl.py s
  python extract_anime4k_upscale_gan_glsl.py m
"""

import argparse
import os
import re
import struct
import sys

import numpy as np


MAGIC                = 0x4B575A4D
VERSION              = 1
KIND_BLOC97_UPGAN_S   = 21
KIND_BLOC97_UPGAN_M   = 22
KIND_BLOC97_UPGAN_L3  = 23
KIND_BLOC97_UPGAN_VL3 = 24
KIND_BLOC97_UPGAN_UL4 = 25
KIND_BLOC97_UPGAN_UUL4= 26
NUM_FEAT             = 4
SCALE                = 2     # default for S/M; x3 tiers use SCALE=3, x4 use SCALE=4 in their headers

SOURCE_DIR = r"the Anime4K GLSL distribution / Upscale"
OUT_DIR    = r"the websr weights / anime4k"

# Match a single `mat4(...) * IDENT [(args)]` term. Args optional so we can
# handle both `g_5` (1x1, no offset) and `go_0(-1.0, -1.0)` (spatial).
MAT4_TERM_RE = re.compile(
    r"mat4\(\s*([\-\d\.\,\seE\+]+?)\s*\)\s*\*\s*([A-Za-z_]\w*)"
    r"(?:\(\s*([\+\-]?[\d\.eE\+\-]+)\s*,\s*([\+\-]?[\d\.eE\+\-]+)\s*\))?"
)
BIAS_RE = re.compile(
    r"result\s*\+=\s*vec4\(\s*([\-\d\.\,\seE\+]+?)\s*\)\s*;"
)
PASS_RE = re.compile(
    r"//!DESC\s+Anime4K-v4\.1-Upscale-GAN-x[234]-\([SMLVU]+\)-(Conv-\dx\dx\dx\d+)"
)


def parse_glsl_passes(glsl_text):
    matches = list(PASS_RE.finditer(glsl_text))
    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(glsl_text)
        sections.append((m.group(1), glsl_text[start:end]))
    return sections


def parse_floats(s, expected):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != expected:
        raise ValueError(f"expected {expected} floats, got {len(parts)}: {s[:80]}")
    return [float(p) for p in parts]


def parse_bias(section, oc):
    """Parse the result += vec4(...) bias.  For oc=3 the 4th component
    in the GLSL is always 0.0 (padding); we still return all 4 and the
    caller decides what to do."""
    bias_match = BIAS_RE.search(section)
    if not bias_match:
        raise ValueError("no bias")
    floats = parse_floats(bias_match.group(1), 4)
    return np.asarray(floats[:oc], dtype=np.float32)


def collect_terms(section):
    """Iterate over (floats, gname, kx, ky) tuples for each mat4 * g[...]
    term in the pass body. kx, ky are None for 1x1 (no offset call)."""
    for m in MAT4_TERM_RE.finditer(section):
        floats = parse_floats(m.group(1), 16)
        gname = m.group(2)
        if m.group(3) is None:
            yield floats, gname, None, None
        else:
            kx = int(float(m.group(3)))
            ky = int(float(m.group(4)))
            yield floats, gname, kx, ky


# Dynamic g_/go_ definition parser. The GLSL `#define g_N (max(...) ...)`
# (1x1) or `#define go_N(x, y) (max(...) ...)` (spatial) lines encode each
# g/go index's source identity and CReLU polarity. We parse these to build
# a {g_idx: (src_idx, polarity)} mapping per layer, where src_idx is the
# 0-based index into the layer's BIND list. This handles BOTH conventions
# the bloc97 GLSL emitter uses:
#   x2_S 1x1 aggregates: g_0=src0+, g_1=src0-, g_2=src1+, g_3=src1- (alternating)
#   x2_M 1x1 aggregates: g_0=src0+, g_1=src1+, g_2=src0-, g_3=src1-, g_4=src2+, ... (mixed L-style + alternating)
#   x2_M 3x3 body 2-src: g_0=src0+, g_1=src1+, g_2=src0-, g_3=src1- (L-style)
#   x2_M 3x3 final tail 3-src: g_0=src0+, g_1=src1+, g_2=src2+, g_3=src0-, g_4=src1-, g_5=src2- (L-style for N=3)
GO_DEFINE_RE = re.compile(
    r"#define\s+(g_\d+|go_\d+)(?:\([^)]+\))?\s+\("
    r"max\(\s*(-)?\(?\s*([A-Za-z_]\w*)_tex(?:Off)?\("
)
BIND_RE = re.compile(r"//!BIND\s+([A-Za-z_]\w*)")


def parse_g_mapping(section):
    """Parse #define g_N / go_N lines in `section`. Returns dict mapping
    g_idx -> (src_idx, polarity) where src_idx is 0-based index into the
    *actually-read-via-CReLU* sources (assigned in the order each unique
    source name first appears across the g_/go_ defines), and polarity
    is 0 (positive) or 1 (negative).

    Why first-appearance and NOT //!BIND order: passes like the final
    tail BIND `MAIN + conv0ups + conv0ups1`, where MAIN is bound for the
    bilinear residual `+ MAIN_tex(MAIN_pos)` at end-of-hook but is NOT
    referenced via any g_/go_ define. Indexing by BIND order would shift
    the CReLU sources by 1 and overflow the conv weight buffer.
    """
    src_order = []
    src_idx = {}
    mapping = {}
    for m in GO_DEFINE_RE.finditer(section):
        gname     = m.group(1)
        negate    = m.group(2) == "-"
        src_name  = m.group(3)
        g_idx     = int(gname.split("_")[1])
        if src_name not in src_idx:
            src_idx[src_name] = len(src_order)
            src_order.append(src_name)
        polarity = 1 if negate else 0
        mapping[g_idx] = (src_idx[src_name], polarity)
    return mapping


def extract_conv_head_3x3x3(section):
    """Head conv (Conv-4x3x3x3, no CReLU, single source go_0)."""
    by_offset = {}
    for floats, gname, kx, ky in collect_terms(section):
        if gname != "go_0":
            raise ValueError(f"head: unexpected reader {gname}")
        if kx is None:
            raise ValueError("head: missing offset")
        by_offset[(kx, ky)] = floats
    if len(by_offset) != 9:
        raise ValueError(f"head: expected 9 offsets, got {len(by_offset)}")

    bias = parse_bias(section, 4)
    w = np.zeros((4, 3, 3, 3), dtype=np.float32)   # OC=4, IC=3, KH=3, KW=3
    for h in range(3):
        for ww in range(3):
            kx, ky = ww - 1, h - 1
            floats = by_offset[(kx, ky)]
            for oc in range(4):
                for ic in range(3):
                    w[oc, ic, h, ww] = floats[ic * 4 + oc]
                # GLSL pads to 16 with 0.0 for the unused 4th IC column
                if floats[3 * 4 + oc] != 0.0:
                    raise ValueError(f"head: non-zero pad col at oc={oc}")
    return w, bias


def extract_conv_body_3x3x8(section):
    """Body conv (Conv-4x3x3x8, single source CReLU).  18 mat4 reads.
    go_0 = source positive -> ic 0..3
    go_1 = source negative -> ic 4..7
    """
    reads = {}
    for floats, gname, kx, ky in collect_terms(section):
        if gname not in ("go_0", "go_1"):
            raise ValueError(f"body: unexpected reader {gname}")
        if kx is None:
            raise ValueError("body: missing offset")
        polarity = int(gname[3:])
        key = (polarity, kx, ky)
        if key in reads:
            raise ValueError(f"body: duplicate {key}")
        reads[key] = floats
    if len(reads) != 18:
        raise ValueError(f"body: expected 18 reads, got {len(reads)}")

    bias = parse_bias(section, 4)
    w = np.zeros((4, 8, 3, 3), dtype=np.float32)
    for (polarity, kx, ky), floats in reads.items():
        ic_base = polarity * 4
        h, ww = ky + 1, kx + 1
        for oc in range(4):
            for ic in range(4):
                w[oc, ic_base + ic, h, ww] = floats[ic * 4 + oc]
    return w, bias


def extract_conv_aggregate_1x1(section, n_sources):
    """1x1 aggregate conv (Conv-4x1x1xN, multi-source CReLU).
    Output ic layout: 'alternating per source' (matches the C++ side which
    issues one CReLU call per source, each writing pos lobe into the
    first 4 channels and neg lobe into the next 4 channels of an 8-ch slot).
    Per-source ic_base = src_idx * 8 + polarity * 4. Source/polarity per
    g_idx parsed dynamically from `#define g_N` lines.
    """
    g_map = parse_g_mapping(section)
    expected = 2 * n_sources
    if len(g_map) != expected:
        raise ValueError(f"aggregate({n_sources}-src): expected {expected} g_ defines, "
                         f"got {len(g_map)}: {sorted(g_map.keys())}")

    reads = {}
    for floats, gname, kx, ky in collect_terms(section):
        if not gname.startswith("g_"):
            raise ValueError(f"aggregate: unexpected reader {gname} (want g_N)")
        if kx is not None:
            raise ValueError(f"aggregate: 1x1 conv has offset {gname}({kx},{ky})")
        g_idx = int(gname[2:])
        if g_idx in reads:
            raise ValueError(f"aggregate: duplicate g_{g_idx}")
        reads[g_idx] = floats
    if len(reads) != expected:
        raise ValueError(f"aggregate({n_sources}-src): expected {expected} reads, "
                         f"got {len(reads)}")

    bias = parse_bias(section, 4)
    in_c = 4 * expected   # 8 ch per source (CReLU pos+neg)
    w = np.zeros((4, in_c, 1, 1), dtype=np.float32)
    for g_idx, floats in reads.items():
        src_idx, polarity = g_map[g_idx]
        ic_base = src_idx * 8 + polarity * 4
        for oc in range(4):
            for ic in range(4):
                w[oc, ic_base + ic, 0, 0] = floats[ic * 4 + oc]
    return w, bias


def extract_conv_body_3x3_multi(section, n_sources, oc=4):
    """Multi-source 3x3 body conv (Conv-OCx3x3xIC, IC = 8*n_sources).
    For n_sources >= 2 the GLSL uses L-style or mixed g_idx layouts; we
    parse #define lines to get the exact (src_idx, polarity) per g_idx.
    Output layout same as aggregate: ic_base = src_idx*8 + polarity*4.
    Total mat4 reads = 9 * 2 * n_sources.
    """
    g_map = parse_g_mapping(section)
    expected_g = 2 * n_sources
    if len(g_map) != expected_g:
        raise ValueError(f"body({n_sources}-src 3x3): expected {expected_g} g_ defines, "
                         f"got {len(g_map)}: {sorted(g_map.keys())}")

    reads = {}
    for floats, gname, kx, ky in collect_terms(section):
        if not gname.startswith("go_"):
            raise ValueError(f"body({n_sources}-src): unexpected reader {gname}")
        if kx is None:
            raise ValueError(f"body({n_sources}-src): missing spatial offset")
        g_idx = int(gname[3:])
        key = (g_idx, kx, ky)
        if key in reads:
            raise ValueError(f"body({n_sources}-src): duplicate {key}")
        reads[key] = floats
    expected_reads = 9 * expected_g
    if len(reads) != expected_reads:
        raise ValueError(f"body({n_sources}-src): expected {expected_reads} reads, "
                         f"got {len(reads)}")

    bias = parse_bias(section, oc)
    in_c = 4 * expected_g
    w = np.zeros((oc, in_c, 3, 3), dtype=np.float32)
    for (g_idx, kx, ky), floats in reads.items():
        src_idx, polarity = g_map[g_idx]
        ic_base = src_idx * 8 + polarity * 4
        h, ww = ky + 1, kx + 1
        for o in range(oc):
            for ic in range(4):
                w[o, ic_base + ic, h, ww] = floats[ic * 4 + o]
        # For OC=3 (final tail), the 4th OC row across all IC columns is
        # zero-padded.
        if oc == 3:
            for ic in range(4):
                if floats[ic * 4 + 3] != 0.0:
                    raise ValueError(f"body({n_sources}-src 3x3 oc=3): non-zero pad "
                                     f"at go_{g_idx} ({kx},{ky}) ic={ic}: {floats[ic*4+3]}")
    return w, bias


# NOTE: final-tail extractor for x2_S (Conv-3x3x3x16 with 2 sources, OC=3)
# is now handled uniformly by extract_conv_body_3x3_multi(section, n_sources=2,
# oc=3). The legacy `extract_conv_final_tail_3x3x16` function was retired in
# the 2026-06-01 refactor that added dynamic #define parsing.


def reorder_oihw_to_ohwi_fp16(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.transpose(arr, (0, 2, 3, 1))
    return np.ascontiguousarray(arr.astype(np.float16))


def to_fp16(arr):
    return np.ascontiguousarray(np.asarray(arr, dtype=np.float32).astype(np.float16))


# Pass schedule per tier. Each entry is (parse_fn_name, fn_arg).
# fn_arg is the n_sources for multi-source variants, or None for fixed-shape.
SCHEDULE_S = [
    ("head_3x3x3",         None),     # L1   head
    ("body_3x3x8",         None),     # L2   body (single src CReLU)
    ("body_3x3x8",         None),     # L3   body
    ("aggregate_1x1",      3),        # L4   agg 3-src
    ("body_3x3x8",         None),     # L5   body
    ("body_3x3x8",         None),     # L6   body
    ("aggregate_1x1",      4),        # L7   agg 4-src
    ("body_3x3x8",         None),     # L8   body
    ("body_3x3x8",         None),     # L9   body
    ("aggregate_1x1",      5),        # L10  agg 5-src
    ("body_3x3x8",         None),     # L11  body
    ("body_3x3x8",         None),     # L12  body
    ("aggregate_1x1",      6),        # L13  agg 6-src
    ("body_3x3x8",         None),     # L14  body
    ("aggregate_1x1",      7),        # L15  conv0ups (56-in)
    ("aggregate_1x1",      7),        # L16  conv0ups1
    ("body_3x3_2src_oc3",  None),     # L17  final tail 2-src CReLU OC=3
]

# x2_M is wider than x2_S: 2 heads, 2 body siblings per dense block,
# 2 aggregate siblings per block, 3 conv0ups siblings, 3-source final tail.
SCHEDULE_M = [
    ("head_3x3x3",         None),     # L1   head_a
    ("head_3x3x3",         None),     # L2   head_b
    ("body_3x3_2src_oc4",  None),     # L3   body 2-src CReLU (16-in)
    ("body_3x3_2src_oc4",  None),     # L4   body sibling
    ("aggregate_1x1",      4),        # L5   agg 4-src (head pair + 2 body siblings)
    ("aggregate_1x1",      4),        # L6   agg sibling
    ("body_3x3_2src_oc4",  None),     # L7   body
    ("body_3x3_2src_oc4",  None),     # L8   body sibling
    ("aggregate_1x1",      5),        # L9   agg 5-src
    ("aggregate_1x1",      5),        # L10  agg sibling
    ("body_3x3_2src_oc4",  None),     # L11  body
    ("body_3x3_2src_oc4",  None),     # L12  body sibling
    ("aggregate_1x1",      6),        # L13  agg 6-src
    ("aggregate_1x1",      6),        # L14  agg sibling
    ("body_3x3_2src_oc4",  None),     # L15  body
    ("body_3x3_2src_oc4",  None),     # L16  body sibling
    ("aggregate_1x1",      7),        # L17  agg 7-src
    ("aggregate_1x1",      7),        # L18  agg sibling
    ("body_3x3_2src_oc4",  None),     # L19  lone body
    ("aggregate_1x1",      8),        # L20  conv0ups (64-in)
    ("aggregate_1x1",      8),        # L21  conv0ups1
    ("aggregate_1x1",      8),        # L22  conv0ups2
    ("body_3x3_3src_oc3",  None),     # L23  final tail 3-src CReLU OC=3 (24-in)
]

DISPATCH = {
    "head_3x3x3":         extract_conv_head_3x3x3,
    "body_3x3x8":         extract_conv_body_3x3x8,
    "aggregate_1x1":      extract_conv_aggregate_1x1,
    # Multi-source 3x3 body wrappers (n_sources baked into fn name to
    # keep the schedule table compact).
    "body_3x3_2src_oc4":  lambda section: extract_conv_body_3x3_multi(section, n_sources=2, oc=4),
    "body_3x3_2src_oc3":  lambda section: extract_conv_body_3x3_multi(section, n_sources=2, oc=3),
    "body_3x3_3src_oc4":  lambda section: extract_conv_body_3x3_multi(section, n_sources=3, oc=4),
    "body_3x3_3src_oc3":  lambda section: extract_conv_body_3x3_multi(section, n_sources=3, oc=3),
    "body_3x3_4src_oc4":  lambda section: extract_conv_body_3x3_multi(section, n_sources=4, oc=4),
    "body_3x3_4src_oc3":  lambda section: extract_conv_body_3x3_multi(section, n_sources=4, oc=3),
    "body_3x3_6src_oc4":  lambda section: extract_conv_body_3x3_multi(section, n_sources=6, oc=4),
    "body_3x3_6src_oc3":  lambda section: extract_conv_body_3x3_multi(section, n_sources=6, oc=3),
}


# x4_UL: 67 weighted convs at 4x scale.
#   4 head + 9 dense blocks (each: 2 body 3x3 paired + 4 aggregate 1x1 siblings)
#   + 1 lone body (singleton 3x3 intermediate before conv0ups) + 4 conv0ups
#   + 3 conv1ups (at 4x dims; CReLU(4 conv0ups) = 32 IC)
#   + 1 final 3x3 OC=3 (at 4x dims; CReLU(3 conv1ups) = 24 IC).
SCHEDULE_UL4 = (
    [("head_3x3x3", None)] * 4
    + sum(
        [
            [("body_3x3_4src_oc4", None)] * 2  # body pair (4-src = prev agg quartet OR heads)
            + [("aggregate_1x1", n_src)] * 4    # aggregate quartet (n_src srcs)
            for n_src in (6, 7, 8, 9, 10, 11, 12, 13, 14)  # blocks 0..8
        ], []
    )
    + [("body_3x3_4src_oc4", None)]  # singleton intermediate body (4-src = block 8 agg quartet)
    + [("aggregate_1x1", 15)] * 4    # conv0ups quartet (15 srcs)
    + [("body_3x3_4src_oc4", None)] * 3  # conv1ups trio (4-src = CReLU(conv0ups quartet))
    + [("body_3x3_3src_oc3", None)]      # final tail (3-src = CReLU(conv1ups trio), OC=3)
)
# Sanity check (build-time): 4 + 9*(2+4) + 1 + 4 + 3 + 1 = 67
assert len(SCHEDULE_UL4) == 67, f"SCHEDULE_UL4 length {len(SCHEDULE_UL4)} != 67"


# x4_UUL: 84 weighted convs at 4x scale.
#   6 head + 8 dense blocks (each: 2 body 3x3 paired + 6 aggregate 1x1 siblings)
#   + 1 lone body (singleton 3x3 intermediate)
#   + 6 conv0ups + 6 conv1ups (at 4x dims; CReLU(6 conv0ups) = 48 IC)
#   + 1 final 3x3 OC=3 (at 4x dims; CReLU(6 conv1ups) = 48 IC).
SCHEDULE_UUL4 = (
    [("head_3x3x3", None)] * 6
    + sum(
        [
            [("body_3x3_6src_oc4", None)] * 2  # body pair (6-src = prev agg hex OR 6 heads)
            + [("aggregate_1x1", n_src)] * 6    # aggregate hex (n_src srcs)
            for n_src in (8, 9, 10, 11, 12, 13, 14, 15)  # blocks 0..7
        ], []
    )
    + [("body_3x3_6src_oc4", None)]  # singleton intermediate (6-src = block 7 agg hex)
    + [("aggregate_1x1", 16)] * 6    # conv0ups hex (16 srcs)
    + [("body_3x3_6src_oc4", None)] * 6  # conv1ups hex (6-src = CReLU(conv0ups hex))
    + [("body_3x3_6src_oc3", None)]      # final tail (6-src = CReLU(conv1ups hex), OC=3)
)
assert len(SCHEDULE_UUL4) == 84, f"SCHEDULE_UUL4 length {len(SCHEDULE_UUL4)} != 84"


# x3_L: 30 weighted convs at 3x scale. New tail layout: conv0ups stage at
# src dims, then a NEW conv1ups intermediate 3x3 at 3x dims, then the
# final 3x3 OC=3 tail also at 3x dims. CReLU(conv0ups trio) feeds conv1ups
# (3-src CReLU = 24 IC body conv), CReLU(conv1ups pair) feeds final tail
# (2-src CReLU = 16 IC OC=3 conv).
SCHEDULE_L3 = [
    ("head_3x3x3",         None),     # L1  head_a
    ("head_3x3x3",         None),     # L2  head_b
    ("head_3x3x3",         None),     # L3  head_c
    ("body_3x3_3src_oc4",  None),     # L4  body (CReLU(head trio) = 24 IC)
    ("body_3x3_3src_oc4",  None),     # L5  body sibling
    ("aggregate_1x1",      5),        # L6  agg 5-src (40 IC)
    ("aggregate_1x1",      5),        # L7  agg sibling
    ("aggregate_1x1",      5),        # L8  agg sibling
    ("body_3x3_3src_oc4",  None),     # L9  body (CReLU(agg trio) = 24 IC)
    ("body_3x3_3src_oc4",  None),     # L10 body sibling
    ("aggregate_1x1",      6),        # L11 agg 6-src (48 IC)
    ("aggregate_1x1",      6),        # L12 agg sibling
    ("aggregate_1x1",      6),        # L13 agg sibling
    ("body_3x3_3src_oc4",  None),     # L14 body
    ("body_3x3_3src_oc4",  None),     # L15 body sibling
    ("aggregate_1x1",      7),        # L16 agg 7-src (56 IC)
    ("aggregate_1x1",      7),        # L17 agg sibling
    ("aggregate_1x1",      7),        # L18 agg sibling
    ("body_3x3_3src_oc4",  None),     # L19 body
    ("body_3x3_3src_oc4",  None),     # L20 body sibling
    ("aggregate_1x1",      8),        # L21 agg 8-src (64 IC)
    ("aggregate_1x1",      8),        # L22 agg sibling
    ("aggregate_1x1",      8),        # L23 agg sibling
    ("body_3x3_3src_oc4",  None),     # L24 lone body (CReLU(agg trio))
    ("aggregate_1x1",      9),        # L25 conv0ups (72 IC, 9-src concat)
    ("aggregate_1x1",      9),        # L26 conv0ups sibling
    ("aggregate_1x1",      9),        # L27 conv0ups sibling
    ("body_3x3_3src_oc4",  None),     # L28 conv1ups (at 3x dims; CReLU(conv0ups trio) = 24 IC)
    ("body_3x3_3src_oc4",  None),     # L29 conv1ups sibling
    ("body_3x3_2src_oc3",  None),     # L30 final tail (at 3x dims; CReLU(conv1ups pair) = 16 IC, OC=3)
]

# x3_VL: 47 weighted convs at 3x scale. Same skeleton as x3_L but with
# 2 extra dense blocks (7 total vs 5), 1 extra conv0ups sibling (4 vs 3),
# and 1 extra conv1ups sibling (3 vs 2). Aggregate channel ramp goes
# wider: 40/48/56/64/72/80/88 (vs L's 40/48/56/64), conv0ups @ 96 (vs 72),
# conv1ups @ 32 IC (vs 24), final tail @ 24 IC (vs 16).
SCHEDULE_VL3 = [
    ("head_3x3x3",         None),     # L1  head_a
    ("head_3x3x3",         None),     # L2  head_b
    ("head_3x3x3",         None),     # L3  head_c
    ("body_3x3_3src_oc4",  None),     # L4  body (CReLU(head trio))
    ("body_3x3_3src_oc4",  None),     # L5  body sibling
    ("aggregate_1x1",      5),        # L6  agg 5-src (40 IC)
    ("aggregate_1x1",      5),        # L7  agg sibling
    ("aggregate_1x1",      5),        # L8  agg sibling
    ("body_3x3_3src_oc4",  None),     # L9  body
    ("body_3x3_3src_oc4",  None),     # L10 body sibling
    ("aggregate_1x1",      6),        # L11 agg 6-src
    ("aggregate_1x1",      6),        # L12 agg sibling
    ("aggregate_1x1",      6),        # L13 agg sibling
    ("body_3x3_3src_oc4",  None),     # L14 body
    ("body_3x3_3src_oc4",  None),     # L15 body sibling
    ("aggregate_1x1",      7),        # L16 agg 7-src
    ("aggregate_1x1",      7),        # L17 agg sibling
    ("aggregate_1x1",      7),        # L18 agg sibling
    ("body_3x3_3src_oc4",  None),     # L19 body
    ("body_3x3_3src_oc4",  None),     # L20 body sibling
    ("aggregate_1x1",      8),        # L21 agg 8-src
    ("aggregate_1x1",      8),        # L22 agg sibling
    ("aggregate_1x1",      8),        # L23 agg sibling
    ("body_3x3_3src_oc4",  None),     # L24 body (extra block vs L)
    ("body_3x3_3src_oc4",  None),     # L25 body sibling
    ("aggregate_1x1",      9),        # L26 agg 9-src (72 IC)
    ("aggregate_1x1",      9),        # L27 agg sibling
    ("aggregate_1x1",      9),        # L28 agg sibling
    ("body_3x3_3src_oc4",  None),     # L29 body (extra block vs L)
    ("body_3x3_3src_oc4",  None),     # L30 body sibling
    ("aggregate_1x1",      10),       # L31 agg 10-src (80 IC)
    ("aggregate_1x1",      10),       # L32 agg sibling
    ("aggregate_1x1",      10),       # L33 agg sibling
    ("body_3x3_3src_oc4",  None),     # L34 body (extra block vs L)
    ("body_3x3_3src_oc4",  None),     # L35 body sibling
    ("aggregate_1x1",      11),       # L36 agg 11-src (88 IC)
    ("aggregate_1x1",      11),       # L37 agg sibling
    ("aggregate_1x1",      11),       # L38 agg sibling
    ("body_3x3_3src_oc4",  None),     # L39 lone body
    ("aggregate_1x1",      12),       # L40 conv0ups (96 IC, 12-src concat)
    ("aggregate_1x1",      12),       # L41 conv0ups sibling
    ("aggregate_1x1",      12),       # L42 conv0ups sibling
    ("aggregate_1x1",      12),       # L43 conv0ups sibling (extra vs L's 3)
    ("body_3x3_4src_oc4",  None),     # L44 conv1ups (at 3x dims; CReLU(conv0ups quartet) = 32 IC)
    ("body_3x3_4src_oc4",  None),     # L45 conv1ups sibling
    ("body_3x3_4src_oc4",  None),     # L46 conv1ups sibling (extra vs L's 2)
    ("body_3x3_3src_oc3",  None),     # L47 final tail (at 3x dims; CReLU(conv1ups trio) = 24 IC, OC=3)
]


def extract_tier(glsl_path, out_path, tier_kind, schedule, tier_scale=SCALE):
    print(f"\n=== {os.path.basename(glsl_path)} (scale={tier_scale}) ===")
    with open(glsl_path, "r", encoding="utf-8") as f:
        text = f.read()
    passes = parse_glsl_passes(text)
    print(f"  {len(passes)} passes parsed")
    if len(passes) != len(schedule):
        raise ValueError(f"pass count mismatch: got {len(passes)}, "
                         f"expected {len(schedule)}")

    layers = []
    total_w = 0
    total_b = 0
    for idx, ((tag, body), (fn_name, fn_arg)) in enumerate(zip(passes, schedule)):
        fn = DISPATCH[fn_name]
        if fn_arg is None:
            w, b = fn(body)
        else:
            w, b = fn(body, fn_arg)
        layers.append((w, b))
        total_w += w.size
        total_b += b.size
        print(f"  L{idx+1:2d} {tag:<20s} {fn_name:<22s} "
              f"w_shape={tuple(w.shape)} ({w.size:>5} floats) "
              f"b_shape={tuple(b.shape)}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as out:
        out.write(struct.pack(
            "<6I",
            MAGIC, VERSION, tier_kind,
            len(schedule), NUM_FEAT, tier_scale,
        ))
        for w, b in layers:
            out.write(reorder_oihw_to_ohwi_fp16(w).tobytes())
            out.write(to_fp16(b).tobytes())

    sz = os.path.getsize(out_path)
    # Expected payload bytes (fp16):
    #   sum_over_layers(OC * IC * KH * KW + OC) * 2
    expected_payload = (total_w + total_b) * 2
    expected_total = 24 + expected_payload
    print(f"  wrote {out_path}  ({sz:,} bytes, expected {expected_total:,})")
    if sz != expected_total:
        raise ValueError(f"file size {sz} != expected {expected_total}")
    print(f"  totals: weights={total_w}, biases={total_b}, header=24")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tier", nargs="?", default="all",
                    choices=["s", "m", "l3", "vl3", "ul4", "uul4", "x2all", "x3all", "x4all", "all"])
    args = ap.parse_args()

    targets = []
    # (tier_name, glsl_file, output_kaizenw, kind, schedule, scale)
    all_targets = [
        ("s",    "Anime4K_Upscale_GAN_x2_S.glsl",   "ani4k_upscale_gan_s_x2.kaizenw",   KIND_BLOC97_UPGAN_S,    SCHEDULE_S,    2),
        ("m",    "Anime4K_Upscale_GAN_x2_M.glsl",   "ani4k_upscale_gan_m_x2.kaizenw",   KIND_BLOC97_UPGAN_M,    SCHEDULE_M,    2),
        ("l3",   "Anime4K_Upscale_GAN_x3_L.glsl",   "ani4k_upscale_gan_l_x3.kaizenw",   KIND_BLOC97_UPGAN_L3,   SCHEDULE_L3,   3),
        ("vl3",  "Anime4K_Upscale_GAN_x3_VL.glsl",  "ani4k_upscale_gan_vl_x3.kaizenw",  KIND_BLOC97_UPGAN_VL3,  SCHEDULE_VL3,  3),
        ("ul4",  "Anime4K_Upscale_GAN_x4_UL.glsl",  "ani4k_upscale_gan_ul_x4.kaizenw",  KIND_BLOC97_UPGAN_UL4,  SCHEDULE_UL4,  4),
        ("uul4", "Anime4K_Upscale_GAN_x4_UUL.glsl", "ani4k_upscale_gan_uul_x4.kaizenw", KIND_BLOC97_UPGAN_UUL4, SCHEDULE_UUL4, 4),
    ]
    for entry in all_targets:
        tier = entry[0]
        if (args.tier == "all"
                or args.tier == tier
                or (args.tier == "x2all" and tier in ("s", "m"))
                or (args.tier == "x3all" and tier in ("l3", "vl3"))
                or (args.tier == "x4all" and tier in ("ul4", "uul4"))):
            targets.append(entry)

    for tier, glsl_name, out_name, kind, sched, scale in targets:
        glsl_path = os.path.join(SOURCE_DIR, glsl_name)
        out_path  = os.path.join(OUT_DIR, out_name)
        extract_tier(glsl_path, out_path, kind, sched, tier_scale=scale)


if __name__ == "__main__":
    main()
