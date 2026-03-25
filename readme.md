
## Run the following command to do a speed check

```bash
python scripts/profile_speed.py --model_dir weights/23-36-37/model_best_bp2_serialize.pth --warmup 20 --total 80 --valid_iters 8 --max_disp 192
```

Before speed up: Vanilla Pytorch speed average (after warmup): 338.4[ms] over 60 iters

### Baseline speed runs (before optimization)

| valid_iters | max_disp | Avg latency (ms) | Approx FPS | Measured iters |
|---|---:|---:|---:|---:|
| 2 | 128 | 187.1 | 5.34 | 60 |
| 4 | 160 | 251.3 | 3.98 | 60 |
| 8 | 192 | 338.4 | 2.95 | 60 |

> Notes:
> - Command template: `python scripts/profile_speed.py --model_dir weights/23-36-37/model_best_bp2_serialize.pth --warmup 20 --total 80 --valid_iters <N> --max_disp <M>`
> - Approx FPS is computed as `1000 / latency_ms`.



## Run this command to run a trial with the provided demo images:

```bash
python scripts/run_demo.py --model_dir weights/23-36-37/model_best_bp2_serialize.pth --left_file demo_data/left.png --right_file demo_data/right.png --intrinsic_file demo_data/K.txt --out_dir output/ --remove_invisible 0 --denoise_cloud 1  --scale 1 --get_pc 1 --valid_iters 8 --max_disp 192 --zfar 100
```

You should get a consolidated image with left right and heatmap, as well as a pointcloud output

## Silence Dynamo/Triton Warnings:

```bash
$env:TORCH_COMPILE_DISABLE="1"; $env:TORCHDYNAMO_DISABLE="1"
```