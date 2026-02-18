# MobileViT + Post-Training Quantization

Re-implementation of [MobileViT](https://arxiv.org/abs/2110.02178) (Mehta & Rastegari, ICLR 2022) with post-training quantization experiments on CIFAR-10.

All three variants (XXS / XS / S) are trained from scratch and quantized to FP16, INT8, and INT4 to compare accuracy, model size, and inference latency.

## Architecture

MobileViT combines MobileNetV2 inverted residual blocks with lightweight transformers. Instead of ViT-style global self-attention on flattened patches, it unfolds spatial feature maps so that each pixel attends across all patches — achieving a global receptive field while preserving spatial order.

```
Input (3×256×256)
  │
  ├─ Conv 3×3 ↓2          ─── Stem
  ├─ MV2                   ─── Stage 1 (stride 1)
  ├─ MV2↓2 → MV2 → MV2    ─── Stage 2 (stride 2)
  ├─ MV2↓2 → MobileViT     ─── Stage 3 (L=2 transformer layers)
  ├─ MV2↓2 → MobileViT     ─── Stage 4 (L=4)
  ├─ MV2↓2 → MobileViT     ─── Stage 5 (L=3)
  ├─ Conv 1×1 → AvgPool    ─── Head
  └─ Linear                ─── Classifier
```

Each MobileViT block: Conv-3×3 → Conv-1×1 for local features, unfold to `(B·P, N, d)`, run transformer across patches, fold back, concat with skip connection, Conv-3×3.

### Configurations (Table 4, Appendix A)

| Variant | Channels | Transformer dims | Heads | Params |
|---------|----------|------------------|-------|--------|
| XXS     | 16→16→24→48→64→80→320 | 64, 80, 96 | 1, 1, 1 | 0.95M |
| XS      | 16→32→48→64→80→96→384 | 96, 120, 144 | 1, 2, 3 | 1.94M |
| S       | 16→32→64→96→128→160→640 | 144, 192, 240 | 1, 2, 3 | 4.94M |

FFN hidden dim is 2d (not the usual 4d). XXS uses expansion ratio 2, others use 4. Patch size is 2×2 at all levels.

## Quick Start

```bash
git clone https://github.com/chanjoongx/mobilevit-quantization.git
cd mobilevit-quantization
pip install -r requirements.txt
```

```bash
# Training
python train.py --model xxs --epochs 100 --amp
python train.py --model xs --epochs 100 --amp
python train.py --model s --epochs 100 --amp

# Quantization
python quantize.py --checkpoint checkpoints/mobilevit_xxs/best.pth --model xxs
```

CIFAR-10 is downloaded automatically. Checkpoints go to `checkpoints/mobilevit_{variant}/`, quantization results to `results/`.

## Results

All models trained for 100 epochs on CIFAR-10 (resized to 256×256), AdamW, cosine LR, AMP. Quantization benchmarks on CPU, single-threaded.

### Training

| Variant | Params | Test Accuracy |
|---------|-------:|--------------:|
| XXS     | 0.95M  | **93.99%**    |
| XS      | 1.94M  | **94.83%**    |
| S       | 4.94M  | **95.44%**    |

### Quantization

| Variant | Method | Accuracy | Δ Acc    | Size (MB) | Compression | Latency (ms) |
|---------|--------|:--------:|:--------:|:---------:|:-----------:|:------------:|
| XXS     | FP32   | 93.99%   | —        | 3.76      | 1.0×        | 11.7         |
|         | FP16   | 94.00%   | +0.01    | 1.93      | 2.0×        | 11.5         |
|         | INT8   | 91.08%   | −2.91    | 1.26      | 3.0×        | 21.6         |
|         | INT4   | 23.43%   | −70.56   | 0.55*     | 6.9×*       | FP32 fallback|
| XS      | FP32   | 94.83%   | —        | 7.52      | 1.0×        | 27.0         |
|         | FP16   | 94.82%   | −0.01    | 3.81      | 2.0×        | 27.2         |
|         | INT8   | 91.24%   | −3.59    | 2.29      | 3.3×        | 21.6         |
|         | INT4   | 51.75%   | −43.08   | 1.08*     | 7.0×*       | FP32 fallback|
| S       | FP32   | 95.44%   | —        | 19.01     | 1.0×        | 67.7         |
|         | FP16   | 95.42%   | −0.02    | 9.55      | 2.0×        | 66.9         |
|         | INT8   | 91.02%   | −4.42    | 5.29      | 3.6×        | 64.2         |
|         | INT4   | 11.22%   | −84.22   | 2.60*     | 7.3×*       | FP32 fallback|

\*INT4 size is estimated (ideal 4-bit packing). Inference uses FP32 fallback — no native INT4 kernel.

<img alt="Training curves" src="https://github.com/user-attachments/assets/634afa0e-9b5c-4528-979f-5a414fba41b2" />
<img alt="Quantization comparison" src="https://github.com/user-attachments/assets/25e83b8e-0103-45b8-bcee-16671cc9e9d3" />

### Observations

FP16 just works — half the size, same accuracy, same latency on CPU (no x86 FP16 compute).

INT8 is the interesting middle ground. All three variants land around 91% regardless of model size, probably because the FX graph mode quantizer has the same float fallback behavior (SiLU etc. stay in FP32) no matter what. XXS actually gets *slower* with INT8 due to quant/dequant overhead; XS and S get faster, so the overhead amortizes with scale.

INT4 results are non-monotonic: XS (51.75%) > XXS (23.43%) and XS > S (11.22%). XXS has too few weights per channel for 4-bit to work — a depthwise 3×3 conv is just 9 values mapped to 16 levels. S likely collapses from quantization noise compounding through its deeper transformer blocks (dims 144/192/240, depths [2,4,3]). XS happens to be wide enough to absorb noise but shallow enough to avoid cascading errors.

## Implementation Notes

Trained on CIFAR-10 (32×32 → 256×256) instead of ImageNet, so absolute numbers don't match the paper. XXS uses 1 attention head everywhere; XS and S use [1, 2, 3] (the paper doesn't specify, Apple's ml-cvnets uses `head_dim=32`). Batch norm omitted from the 1×1 conv in the local representation path. No dropout.

INT8 uses PyTorch FX graph mode (`prepare_fx` / `convert_fx`, `x86` backend). MobileViT's shape-based assertions (`_check_input`, `_unfold`) had to be monkey-patched out during FX tracing — they use data-dependent control flow that symbolic tracing can't handle. Calibration: 32 batches (~4K images), no augmentation, fixed seed.

INT4 is simulated weight-only quantization: per-channel asymmetric, percentile clipping (1st–99th), 16 levels, dequantized back to FP32. Accuracy impact only — no real INT4 acceleration.

## File Structure

```
├── mobilevit.py      Model definition (Table 4, Figure 1b)
├── train.py          Training (CIFAR-10, cosine LR, label smoothing, AMP)
├── quantize.py       PTQ experiments (FP16 / INT8-FX / INT4-simulated)
├── requirements.txt
└── LICENSE
```

## Requirements

```
torch >= 2.0
torchvision
numpy
matplotlib
```

## References

- Mehta & Rastegari, "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer," ICLR 2022. [[paper]](https://arxiv.org/abs/2110.02178)
- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR 2018. [[paper]](https://arxiv.org/abs/1712.05877)
- Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR 2018. [[paper]](https://arxiv.org/abs/1801.04381)

## License

MIT
