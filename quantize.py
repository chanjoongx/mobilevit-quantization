#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) for MobileViT on CIFAR-10.

Applies FP16, INT8, and INT4 quantization to a pre-trained MobileViT
checkpoint and compares accuracy, model size, and inference latency.

Usage:
    python quantize.py --checkpoint checkpoints/mobilevit_xxs/best.pth
    python quantize.py --checkpoint checkpoints/mobilevit_xs/best.pth --calibration-batches 50

Quantization approach:
    FP16  — simple half-precision cast (torch.float16)
    INT8  — static PTQ via FX graph mode (torch.ao.quantization.quantize_fx).
            FX tracing auto-inserts observers and handles ops without
            quantized kernels (e.g. SiLU) by keeping them in float.
    INT4  — simulated 4-bit weight-only quantization (per-channel asymmetric).
            Weights quantized to 16 levels per output channel with
            percentile clipping (1st-99th) to mitigate outliers, then
            dequantized back to FP32.  No native INT4 kernel — this
            measures accuracy impact only.

All benchmarks (accuracy, latency) are run on CPU so that quantized
and non-quantized models are compared under the same compute backend.

References:
    - Jacob et al., "Quantization and Training of Neural Networks for
      Efficient Integer-Arithmetic-Only Inference," CVPR 2018
    - PyTorch quantization docs: pytorch.org/docs/stable/quantization.html
"""

import os
import copy
import time
import json
import argparse
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.quantization as tq
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from mobilevit import MobileViT


# ------------------------------------------------------------------ #
#  Data                                                                #
# ------------------------------------------------------------------ #

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def get_test_loader(data_dir, img_size, batch_size, workers=2):
    tf = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds = torchvision.datasets.CIFAR10(
        data_dir, train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True)


def get_calibration_loader(data_dir, img_size, batch_size, n_batches, workers=2):
    """Subset of training data for calibration (no augmentation)."""
    tf = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    full_ds = torchvision.datasets.CIFAR10(
        data_dir, train=True, download=True, transform=tf)
    n_samples = min(n_batches * batch_size, len(full_ds))
    indices = np.random.RandomState(0).choice(len(full_ds), n_samples, replace=False)
    subset = Subset(full_ds, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=False,
                      num_workers=workers)


# ------------------------------------------------------------------ #
#  Evaluation                                                          #
# ------------------------------------------------------------------ #

@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def measure_latency(model, input_shape, device="cpu", n_warmup=10, n_runs=50):
    """Average inference time over n_runs forward passes."""
    x = torch.randn(*input_shape, device=device)
    model.eval()

    for _ in range(n_warmup):
        with torch.no_grad():
            model(x)
    if device != "cpu":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        if device != "cpu":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = sorted(times)
    # trim top/bottom 10% for stability
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim] if trim < len(times) // 2 else times
    return float(np.mean(trimmed) * 1000)   # ms


def get_model_size_mb(model):
    """Serialized model size in MB (via state_dict snapshot)."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        torch.save(model.state_dict(), tmp_path)
        size = os.path.getsize(tmp_path) / (1024 * 1024)
    finally:
        os.remove(tmp_path)
    return size


# ------------------------------------------------------------------ #
#  Quantization backends                                               #
# ------------------------------------------------------------------ #

def quantize_int8_fx(model, calib_loader, example_inputs):
    """
    Static INT8 quantization via FX graph mode.

    FX mode traces the computation graph and auto-inserts
    quantize / dequantize nodes.  Ops without a QuantizedCPU kernel
    (like SiLU) are left in float automatically — no manual
    QuantStub / DeQuantStub wrapping needed.

    MobileViT has two FX-incompatible control flow sites that use
    tensor shape values in Python if/assert statements:
      1. MobileViT._check_input()  — if H % 32 != 0
      2. MobileViTBlock._unfold()  — assert H % ph == 0
    Both are disabled during tracing since the caller guarantees
    correct input dimensions.
    """
    from torch.ao.quantization import QConfigMapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    from mobilevit import MobileViTBlock

    model_copy = copy.deepcopy(model)
    model_copy.eval()

    # Disable shape-based control flow that FX can't symbolically trace.
    model_copy._check_input = lambda x: None

    # _unfold has `assert H % ph == 0` — same problem.
    # Temporarily replace with an assert-free version on the class,
    # then restore after tracing to avoid side effects.
    _orig_unfold = MobileViTBlock._unfold

    def _unfold_traceable(self, x):
        B, d, H, W = x.shape
        ph, pw = self.ph, self.pw
        nh, nw = H // ph, W // pw
        x = x.reshape(B, d, nh, ph, nw, pw)
        x = x.permute(0, 3, 5, 2, 4, 1)
        x = x.reshape(B * ph * pw, nh * nw, d)
        return x, (B, d, nh, nw)

    MobileViTBlock._unfold = _unfold_traceable

    try:
        qconfig_mapping = QConfigMapping().set_global(
            tq.get_default_qconfig("x86")      # "qnnpack" on ARM
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            prepared = prepare_fx(model_copy, qconfig_mapping, example_inputs)

        # calibration — BN already frozen (eval mode)
        with torch.no_grad():
            for imgs, _ in calib_loader:
                prepared(imgs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            quantized = convert_fx(prepared)
    finally:
        MobileViTBlock._unfold = _orig_unfold

    return quantized


def quantize_int4_weights(model):
    """
    Simulated 4-bit weight-only quantization (per-channel asymmetric).

    Two improvements over naive min-max per-channel:
      1. Percentile clipping (1st-99th) — outlier weights don't blow
         up the scale, so the 16 quantization levels cover the bulk
         of the distribution more evenly.
      2. Min-offset formula — instead of computing a zero-point and
         clamping it to [0, 15] (which saturates channels whose
         weights are all-positive or all-negative), we directly use
         the clipped minimum as the offset:
           q = round((w - lo) / scale),  w_hat = q * scale + lo
         This is mathematically equivalent to asymmetric quantization
         but avoids the zero-point clamping pathology.

    PyTorch doesn't ship native INT4 compute kernels, so we:
        1. Quantize each output channel to 16 levels (4-bit)
        2. Dequantize back to FP32 for inference

    Accuracy impact is real; latency speedup is not.
    """
    model_q = copy.deepcopy(model)

    for name, param in model_q.named_parameters():
        if param.dim() < 2:
            continue

        w = param.data.float()
        shape_orig = w.shape
        flat = w.view(w.shape[0], -1)           # [out_ch, -1]

        # Percentile clipping: ignore top/bottom 1% outliers.
        # For tiny channels (e.g. depthwise 3x3 = 9 elements),
        # the 1st/99th percentile approaches min/max anyway.
        lo = torch.quantile(flat, 0.01, dim=1, keepdim=True)
        hi = torch.quantile(flat, 0.99, dim=1, keepdim=True)
        flat_c = torch.clamp(flat, min=lo, max=hi)

        scale = (hi - lo) / 15.0                # 16 levels per channel
        scale.clamp_(min=1e-10)

        # Min-offset quantization: map [lo, hi] -> [0, 15]
        q = ((flat_c - lo) / scale).round().clamp(0, 15)
        dequantized = q * scale + lo

        param.data.copy_(dequantized.view(shape_orig))

    return model_q


# ------------------------------------------------------------------ #
#  INT4 packed size estimation                                         #
# ------------------------------------------------------------------ #

def estimate_int4_size_mb(model):
    """
    Estimate serialized size assuming true 4-bit weight packing.
    Accounts for per-channel scale (FP32) and zero-point (4-bit)
    overhead — still a lower bound vs. real deployment formats.
    """
    total_bits = 0
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            total_bits += p.numel() * 4               # 4-bit per weight
            total_bits += p.shape[0] * (32 + 4)       # scale(fp32) + zp(int4) per ch
        else:
            total_bits += p.numel() * 32              # keep biases in FP32
    for name, b in model.named_buffers():
        total_bits += b.numel() * 32                  # BN running stats etc.
    return total_bits / 8 / (1024 * 1024)


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main(args):
    device = "cpu"    # quantized models run on CPU
    torch.set_num_threads(1)
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Note: all benchmarks run on CPU for fair comparison.\n")

    # load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", args.model)
    model = MobileViT(config, num_classes=10)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Model: MobileViT-{config.upper()}  "
          f"(trained acc: {ckpt.get('test_acc', '?')}%)")

    # data
    test_loader = get_test_loader(args.data_dir, args.img_size, args.batch_size)
    calib_loader = get_calibration_loader(
        args.data_dir, args.img_size, args.batch_size,
        args.calibration_batches)

    input_shape = (1, 3, args.img_size, args.img_size)
    results = {}

    # ---- FP32 baseline ----------------------------------------------
    print("\n[1/4] FP32 baseline")
    fp32_acc = evaluate(model, test_loader, device)
    fp32_size = get_model_size_mb(model)
    fp32_lat = measure_latency(model, input_shape, device)
    results["FP32"] = {
        "accuracy": round(fp32_acc, 2),
        "size_mb": round(fp32_size, 2),
        "latency_ms": round(fp32_lat, 2),
    }
    print(f"  Accuracy: {fp32_acc:.2f}%  Size: {fp32_size:.2f} MB  "
          f"Latency: {fp32_lat:.1f} ms")

    # ---- FP16 -------------------------------------------------------
    print("\n[2/4] FP16")
    model_fp16 = copy.deepcopy(model).half()
    fp16_size = get_model_size_mb(model_fp16)
    # round-trip half->float to measure accuracy impact of FP16 precision loss.
    # latency is identical to FP32 on CPU (no FP16 compute acceleration).
    model_fp16.float()
    fp16_acc = evaluate(model_fp16, test_loader, device)
    fp16_lat = measure_latency(model_fp16, input_shape, device)
    results["FP16"] = {
        "accuracy": round(fp16_acc, 2),
        "size_mb": round(fp16_size, 2),
        "latency_ms": round(fp16_lat, 2),
        "compression": round(fp32_size / fp16_size, 1),
        "acc_delta": round(fp16_acc - fp32_acc, 2),
        "note": "Latency same as FP32 -- no FP16 compute on CPU.",
    }
    print(f"  Accuracy: {fp16_acc:.2f}%  Size: {fp16_size:.2f} MB  "
          f"({fp32_size/fp16_size:.1f}x)  Latency: {fp16_lat:.1f} ms")

    # ---- INT8 static quantization (FX graph mode) -------------------
    print("\n[3/4] INT8 (static PTQ, FX graph mode)")
    try:
        example_inputs = (torch.randn(*input_shape),)
        model_int8 = quantize_int8_fx(model, calib_loader, example_inputs)
        int8_acc = evaluate(model_int8, test_loader, device)
        int8_size = get_model_size_mb(model_int8)
        int8_lat = measure_latency(model_int8, input_shape, device)
        results["INT8"] = {
            "accuracy": round(int8_acc, 2),
            "size_mb": round(int8_size, 2),
            "latency_ms": round(int8_lat, 2),
            "compression": round(fp32_size / int8_size, 1),
            "acc_delta": round(int8_acc - fp32_acc, 2),
        }
        print(f"  Accuracy: {int8_acc:.2f}% ({int8_acc - fp32_acc:+.2f}%)  "
              f"Size: {int8_size:.2f} MB ({fp32_size/int8_size:.1f}x)  "
              f"Latency: {int8_lat:.1f} ms")
    except Exception as e:
        print(f"  INT8 quantization failed: {e}")
        print("  (FX tracing or quantized op dispatch issue)")
        results["INT8"] = {"error": str(e)}

    # ---- INT4 simulated (per-channel) --------------------------------
    print("\n[4/4] INT4 (simulated weight-only, per-channel + clipping)")
    model_int4 = quantize_int4_weights(model)
    int4_acc = evaluate(model_int4, test_loader, device)
    int4_size_actual = get_model_size_mb(model_int4)   # still FP32 on disk
    int4_size_packed = estimate_int4_size_mb(model)     # theoretical
    int4_lat = measure_latency(model_int4, input_shape, device)
    results["INT4"] = {
        "accuracy": round(int4_acc, 2),
        "size_mb_actual": round(int4_size_actual, 2),
        "size_mb_packed_est": round(int4_size_packed, 2),
        "latency_ms": round(int4_lat, 2),
        "acc_delta": round(int4_acc - fp32_acc, 2),
        "note": ("Per-channel 4-bit weight quantization with percentile clipping "
                 "(1st-99th).  Dequantized back to FP32 for inference.  "
                 "No native INT4 kernel -- latency unchanged. "
                 "Size estimate includes per-channel scale/zp overhead."),
    }
    print(f"  Accuracy: {int4_acc:.2f}% ({int4_acc - fp32_acc:+.2f}%)  "
          f"Size (packed est): {int4_size_packed:.2f} MB  "
          f"Latency: {int4_lat:.1f} ms (FP32 fallback)")

    # ---- summary table ----------------------------------------------
    print(f"\n{'='*65}")
    print(f"{'Method':<10} {'Acc (%)':<12} {'Size (MB)':<14} "
          f"{'Compress':<10} {'Lat (ms)':<10}")
    print(f"{'-'*65}")

    for method in ["FP32", "FP16", "INT8", "INT4"]:
        r = results.get(method, {})
        if "error" in r:
            print(f"{method:<10} {'FAILED':<12} {'-':<14} {'-':<10} {'-':<10}")
            continue
        acc = r.get("accuracy", "-")
        size = r.get("size_mb", r.get("size_mb_packed_est", "-"))
        comp = r.get("compression", "-")
        if method == "FP32":
            comp = "1.0x"
        elif method == "INT4":
            comp = f"{fp32_size / int4_size_packed:.1f}x*"
            size = int4_size_packed
        lat = r.get("latency_ms", "-")
        print(f"{method:<10} {acc:<12} {size:<14} {comp:<10} {lat:<10}")

    print(f"{'='*65}")
    print("* INT4 size is estimated (packed). Latency uses FP32 fallback.")
    print("  All benchmarks on CPU (batch_size=1 for latency).")

    # save
    out_path = Path(args.output) / f"quantization_results_{config}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="PTQ comparison for MobileViT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--checkpoint", required=True,
                   help="path to trained .pth file")
    p.add_argument("--model", default="xxs", choices=["xxs", "xs", "s"],
                   help="model config (used if checkpoint lacks 'config')")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--calibration-batches", type=int, default=32,
                   help="number of batches for INT8 calibration")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output", default="./results")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())