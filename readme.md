
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


## DOCKER!

Build image from repo root: docker build -t ffs -f docker/dockerfile .
Start GPU container (PowerShell): docker run --gpus all -it --rm -v "${PWD}:/workspace" -w /workspace ffs bash
Inside container:
python make_onnx.py --model_dir model_best_bp2_serialize.pth --save_path output/ --height 448 --width 640 --valid_iters 2 --max_disp 160
trtexec --onnx=output/feature_runner.onnx --saveEngine=output/feature_runner.engine --fp16 --useCudaGraph
trtexec --onnx=output/post_runner.onnx --saveEngine=output/post_runner.engine --fp16 --useCudaGraph


## Windows TensorRT (local)

Use the same TensorRT version for engine build and runtime.

Install Triton for Windows (PyTorch 2.6):

```powershell
python -m pip install --force-reinstall triton-windows==3.2.0.post21
```

```powershell
# 1) Export ONNX
python scripts/make_onnx.py --model_dir weights/20-26-39/model_best_bp2_serialize.pth --save_path output2/ --height 448 --width 640 --valid_iters 2 --max_disp 160

# 2) Build BOTH engines with your local trtexec.exe
& "C:/path/to/TensorRT-10.16.0.72/bin/trtexec.exe" --onnx=output2/feature_runner.onnx --saveEngine=output2/feature_runner.engine --fp16 --useCudaGraph
& "C:/path/to/TensorRT-10.16.0.72/bin/trtexec.exe" --onnx=output2/post_runner.onnx --saveEngine=output2/post_runner.engine --fp16 --useCudaGraph

# 3) Run TensorRT demo (headless-safe path)
python scripts/run_demo_tensorrt.py --onnx_dir output2/ --left_file demo_data/left.png --right_file demo_data/right.png --intrinsic_file demo_data/K.txt --out_dir output2/ --remove_invisible 0 --denoise_cloud 1 --get_pc 0 --zfar 100
```

If you hit Torch Dynamo/Triton compile errors on Windows:

```powershell
$env:TORCH_COMPILE_DISABLE="1"; $env:TORCHDYNAMO_DISABLE="1"
```


## Profile_Speed_TRT!

```bash
python scripts/profile_speed_trt.py --onnx_dir output2/ --warmup 20 --total 80
```

Latest local Windows result (output2): ~54.5 ms average after warmup.