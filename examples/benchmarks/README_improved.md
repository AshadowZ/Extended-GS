# Improved Strategy Benchmark

This directory contains scripts for benchmarking the ImprovedStrategy against DefaultStrategy on the Mip-NeRF360 dataset.

## Files

- `improved.sh` - Main benchmark script that compares DefaultStrategy vs ImprovedStrategy with budgets of 1M and 2M

## Usage

### Prerequisites

1. Download the Mip-NeRF360 dataset:
   ```bash
   cd examples
   python datasets/download_dataset.py
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Benchmark

To run the full benchmark comparing all strategies:

```bash
./examples/benchmarks/improved.sh
```

This will:
- Run training and evaluation for 3 strategies:
  - DefaultStrategy (original 3DGS approach)
  - ImprovedStrategy with 1M budget
  - ImprovedStrategy with 2M budget
- Generate comparison tables for each scene
- Create a summary markdown file with overall averages

## Output Format

The benchmark generates a markdown file with the following structure:

```markdown
# Strategy Comparison Results on Mip-NeRF360 Dataset

## Scene: garden

| Strategy | PSNR | SSIM | LPIPS | Num GS | Training Time |
|----------|------|------|-------|--------|---------------|
| default | 25.5 | 0.85 | 0.15  | 1000000 | 3600.0 |
| improved_1M | 25.5 | 0.85 | 0.15 | 1000000 | 3600.0 |
| improved_2M | 25.5 | 0.85 | 0.15 | 1000000 | 3600.0 |

## Scene: bicycle

...

## Overall Summary

| Strategy | Avg PSNR | Avg SSIM | Avg LPIPS | Avg Num GS | Avg Training Time |
|----------|----------|----------|-----------|------------|------------------|
| default | 25.500 | 0.850 | 0.150 | 1000000.000 | 3600.000 |
| improved_1M | 25.500 | 0.850 | 0.150 | 1000000.000 | 3600.000 |
| improved_2M | 25.500 | 0.850 | 0.150 | 1000000.000 | 3600.000 |
```

## Scenes

The benchmark runs on the following Mip-NeRF360 scenes:
- garden
- bicycle
- stump
- bonsai
- counter
- kitchen
- room

## Metrics

For each strategy and scene, the following metrics are collected:
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (higher is better)
- **LPIPS**: Learned Perceptual Image Patch Similarity (lower is better)
- **Num GS**: Number of Gaussians in the final model
- **Training Time**: Total training time in seconds

## Results Directory

Results are saved to:
- `results/benchmark_improved_comparison/` - Main results
- `results/test_benchmark_improved_comparison/` - Test results

The final summary is saved as `comparison_summary.md` in each directory.