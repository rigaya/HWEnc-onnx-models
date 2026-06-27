#!/usr/bin/env python3
"""Convert NNEDI3 (bjin/mpv-prescalers, LGPL-3.0) to ONNX for --vpp-onnx.
Faithful port of nnedi3.py: per predicted pixel, gather a W x H window of source
luma, normalise (mean/std), run the softmax mixture-of-experts predictor, output
mstd0 + 5*vsum/wsum*std. Doubling is done in y then x (two passes) for a 2x model.
Self-verifies an independent numpy reference vs the exported ONNX.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from onnx_export_common import export_onnx

UP = r"upstream"  # path to a clone of bjin/mpv-prescalers (source branch); has weights/nnedi3_weights.bin
EPS = 1.192092896e-7
# weight_offsets[window.value*5 + neurons.value]; windows: 8x4 -> value 0, 8x6 -> value 1
WEIGHT_OFFSETS = [0, 1088, 3264, 7616, 16320, 33728, 35328, 38528, 44928, 57728]
NNS = {16:0, 32:1, 64:2, 128:3, 256:4}

def load_nnedi3(nns, W, H):
    raw = open(f"{UP}/weights/nnedi3_weights.bin","rb").read()
    assert len(raw) == 83328*4
    wint = np.frombuffer(raw, dtype="<i4")
    wflt = wint.view("<f4")                      # int bits -> float
    win_val = 0 if (W,H)==(8,4) else 1
    off = WEIGHT_OFFSETS[win_val*5 + NNS[nns]]
    ws = W*H
    stride = ws*2 + 4
    W0 = np.zeros((nns, ws), np.float64)         # softmax weights (set 0)
    W1 = np.zeros((nns, ws), np.float64)         # value weights   (set 1)
    b0 = np.zeros(nns, np.float64); b1 = np.zeros(nns, np.float64)
    for n in range(nns):
        base = off + stride*n
        W0[n] = wflt[base        : base+ws]
        W1[n] = wflt[base+ws     : base+ws*2]
        # GLSL: weightWS(n,s,i) = base + ws*2 + i ; the WS() macro uses i=0,1
        # so b0 (softmax bias) = +0 and b1 (value bias) = +1 (NOT +2/+3).
        b0[n] = wflt[base+ws*2 + 0]
        b1[n] = wflt[base+ws*2 + 1]
    return W0, W1, b0, b1

def predict(win, W0, W1, b0, b1):
    # win: [..., ws] window taps (row-major x + y*W). returns predicted value [...].
    # CENTERED formulation for fp16 stability: NNEDI3's raw variance mean(x^2)-mean(x)^2
    # AND the dot sum_i x*W0 (W0 sums to ~0) both lose all precision to fp16
    # catastrophic cancellation. Subtract the window mean first; this is exact
    # (sum1 = (x-m)@W0 + m*sum(W0) == x@W0) but keeps the accumulands small.
    mean = win.mean(-1, keepdims=True)            # [...,1]
    wc = win - mean
    var = (wc*wc).mean(-1)                         # [...]  stable variance
    inv = np.where(var >= EPS, 1.0/np.sqrt(np.maximum(var,EPS)), 0.0)
    std = var*inv
    m = mean[...,0]
    W0s = W0.sum(1); W1s = W1.sum(1)              # per-neuron weight sums
    sum1 = wc @ W0.T + m[...,None]*W0s            # == win @ W0, fp16-stable
    sum2 = wc @ W1.T + m[...,None]*W1s
    logits = sum1*inv[...,None] + b0
    logits = logits - logits.max(-1, keepdims=True)   # fp16-stable softmax (shift-invariant)
    e = np.exp(logits)
    v = sum2*inv[...,None] + b1
    ell = v/(1.0+np.abs(v))
    vsum = (e*ell).sum(-1); wsum = e.sum(-1)
    return np.clip(m + 5.0*vsum/np.maximum(wsum,1e-30)*std, 0.0, 1.0)

def double_y_np(img, W0,W1,b0,b1, W,H):
    # predict the in-between rows; even out-rows = input, odd = predicted
    cy, cx = H//2 - 1, W//2 - 1
    Hh, Ww = img.shape
    pad = np.pad(img, ((H, H), (W, W)), mode='edge')
    out = np.zeros((2*Hh, Ww))
    out[0::2] = img
    # window for predicted pixel below input row r at col c:
    # win[x + y*W] = img[r + (y-cy), c + (x-cx)]  (anchor ny=0 -> row r)
    for y in range(H):
        for x in range(W):
            ry = y - cy; rx = x - cx
            # shifted input block aligned to (r=row index of even pixel above gap)
            sub = pad[H+ry : H+ry+Hh, W+rx : W+rx+Ww]
            if x==0 and y==0:
                win = np.zeros((Hh, Ww, W*H))
            win[:,:,x + y*W] = sub
    pred = predict(win, W0,W1,b0,b1)
    out[1::2] = pred
    return out

def nnedi3_ref(img, nns, W, H):
    W0,W1,b0,b1 = load_nnedi3(nns,W,H)
    vy = double_y_np(img, W0,W1,b0,b1, W,H)          # 2x height
    vyx = double_y_np(vy.T, W0,W1,b0,b1, W,H).T      # transpose -> 2x width
    return vyx

# ---- torch module (vertical doubler applied twice) ----
class Doubler(nn.Module):
    def __init__(self, nns, W, H):
        super().__init__()
        W0,W1,b0,b1 = load_nnedi3(nns,W,H)
        self.W=W; self.H=H; self.cy=H//2-1; self.cx=W//2-1
        self.register_buffer('W0', torch.tensor(W0,dtype=torch.float32))
        self.register_buffer('W1', torch.tensor(W1,dtype=torch.float32))
        self.register_buffer('b0', torch.tensor(b0,dtype=torch.float32))
        self.register_buffer('b1', torch.tensor(b1,dtype=torch.float32))
        self.register_buffer('W0s', torch.tensor(W0.sum(1),dtype=torch.float32))
        self.register_buffer('W1s', torch.tensor(W1.sum(1),dtype=torch.float32))
    def double_y(self, x):                            # x:[1,1,Hh,Ww] -> [1,1,2Hh,Ww]
        W,H,cy,cx = self.W,self.H,self.cy,self.cx
        xp = torch.nn.functional.pad(x,(W,W,H,H),mode='replicate')
        _,_,Hp,Wp = xp.shape; Hh=Hp-2*H; Ww=Wp-2*W
        taps=[]
        for y in range(H):
            for x_ in range(W):
                ry=y-cy; rx=x_-cx
                taps.append(xp[:,:,H+ry:H+ry+Hh, W+rx:W+rx+Ww])
        win = torch.cat(taps,dim=1).permute(0,2,3,1)  # [1,Hh,Ww,ws]
        mean = win.mean(-1, keepdim=True)             # [1,Hh,Ww,1]
        wc = win - mean                               # centered (fp16-stable)
        var = (wc*wc).mean(-1)
        inv = torch.where(var>=EPS, 1.0/torch.sqrt(torch.clamp(var,min=EPS)), torch.zeros_like(var))
        std = var*inv
        m = mean[...,0]                               # [1,Hh,Ww]
        sum1 = torch.matmul(wc, self.W0.t()) + m.unsqueeze(-1)*self.W0s   # == win@W0, stable
        sum2 = torch.matmul(wc, self.W1.t()) + m.unsqueeze(-1)*self.W1s
        logits = sum1*inv.unsqueeze(-1) + self.b0
        logits = logits - logits.max(-1, keepdim=True).values   # fp16-stable softmax
        e = torch.exp(logits)
        v = sum2*inv.unsqueeze(-1) + self.b1
        ell = v/(1.0+torch.abs(v))
        vsum=(e*ell).sum(-1); wsum=e.sum(-1)
        pred = torch.clamp(m + 5.0*vsum/torch.clamp(wsum,min=1e-30)*std, 0,1)  # [1,Hh,Ww]
        out = torch.zeros(1,1,2*Hh,Ww)
        out[0,0,0::2]=x[0,0]; out[0,0,1::2]=pred[0]
        return out
    def forward(self,x):
        vy = self.double_y(x)
        vyx = self.double_y(vy.transpose(2,3)).transpose(2,3)
        return vyx

def convert(nns, W, H, output_dir, opset):
    rng=np.random.default_rng(0); img=rng.random((40,52)).astype(np.float32)
    mod=Doubler(nns,W,H).eval()
    with torch.no_grad(): t=mod(torch.tensor(img)[None,None]).numpy()[0,0]
    ref=nnedi3_ref(img.astype(np.float64),nns,W,H)
    print(f"[nns{nns} {W}x{H}] torch-vs-numpyref max diff: {np.abs(t-ref).max():.2e}")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"nnedi3_nns{nns}_win{W}x{H}.onnx"
    export_onnx(mod, torch.tensor(img)[None,None], str(path),
                      input_names=['input'],output_names=['output'],
                      dynamic_axes={'input':{2:'h',3:'w'},'output':{2:'h2',3:'w2'}}, opset_version=opset)
    s=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    o=s.run(None,{'input':img[None,None]})[0][0,0]
    print(f"[nns{nns} {W}x{H}] ONNX-vs-numpyref max diff: {np.abs(o-ref).max():.2e} | out {s.get_outputs()[0].shape} -> {path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path,
                        help="mpv-prescalers source branch checkout containing weights/nnedi3_weights.bin")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    UP = str(args.repo_root)
    for nns in (16, 32, 64, 128, 256):
        for window in ((8, 4), (8, 6)):
            convert(nns, window[0], window[1], args.output, args.opset)
