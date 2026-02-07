<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:8B0000,100:006400&height=180&section=header&text=viva_tensor&fontSize=60&fontColor=fff&animation=twinkling&fontAlignY=35&desc=High-Performance%20Tensors%20for%20Gleam&descSize=20&descAlignY=55" width="100%"/>

[![Gleam](https://img.shields.io/badge/Gleam-FFAFF3?style=for-the-badge&logo=gleam&logoColor=000)](https://gleam.run/)
[![Tests](https://img.shields.io/badge/Tests-187%20passed-2E8B57?style=for-the-badge)](./test)
[![License](https://img.shields.io/badge/MIT-2E8B57?style=for-the-badge)](./LICENSE)

**The fastest tensor library on the BEAM**

</div>

---

## Performance

```mermaid
xychart-beta
    title "GEMM Performance (GFLOPS) - Higher is Better"
    x-axis [2000, 3000, 4000, 5000]
    y-axis "GFLOPS" 0 --> 700
    bar [513, 512, 592, 649]
    bar [442, 486, 516, 535]
    bar [453, 511, 567, 576]
```

> **viva_tensor** (green) vs **PyTorch** (red) vs **NumPy** (blue)

### Verified Benchmarks

| Size | viva_tensor | PyTorch | NumPy | vs PyTorch |
|:----:|:-----------:|:-------:|:-----:|:----------:|
| 2000×2000 | **513 ±13** | 442 ±33 | 453 ±115 | **+16%** |
| 3000×3000 | **512 ±80** | 486 ±16 | 511 ±17 | **+5%** |
| 4000×4000 | **592 ±8** | 516 ±35 | 567 ±5 | **+15%** |
| 5000×5000 | **649 ±22** | 535 ±11 | 576 ±28 | **+21%** |

<details>
<summary>📊 Methodology & Environment</summary>

```mermaid
flowchart LR
    subgraph Methodology
        W[3 Warmup Runs] --> T[10 Timed Runs]
        T --> O[IQR Outlier Removal]
        O --> C[95% Confidence Interval]
    end
```

**Environment:**
- Hardware: Intel i7-13700K (24 threads)
- OS: WSL2 Ubuntu 22.04
- BLAS: Intel MKL 2020.4
- Reference: Kalibera & Jones (2013)

**Reproduce:**
```bash
python3 bench/benchmark.py
Rscript bench/analysis.R  # Statistical analysis
```

</details>

---

## Install

```bash
gleam add viva_tensor
```

## Architecture

```mermaid
graph TB
    subgraph "Gleam Layer"
        A[viva_tensor API]
    end

    subgraph "Erlang Layer"
        B[NIF Resources]
        C[Zero-Copy Memory]
    end

    subgraph "Native Layer"
        D[Zig SIMD]
        E[Intel MKL]
        F[CUDA cuBLAS]
    end

    A --> B
    B --> C
    C --> D & E & F

    style A fill:#FFAFF3
    style D fill:#F7A41D
    style E fill:#0071C5
    style F fill:#76B900
```

## Backend Selection

```mermaid
flowchart LR
    Start([Tensor Op]) --> Check{Platform?}

    Check -->|Windows| MKL[Intel MKL<br/>818 GFLOPS]
    Check -->|Linux + NVIDIA| CUDA[cuBLAS<br/>702 GFLOPS]
    Check -->|Linux| OpenBLAS[OpenBLAS<br/>528 GFLOPS]
    Check -->|Any| Zig[Zig SIMD<br/>134 GFLOPS]

    MKL --> Result([Result])
    CUDA --> Result
    OpenBLAS --> Result
    Zig --> Result

    style MKL fill:#0071C5,color:#fff
    style CUDA fill:#76B900,color:#fff
    style OpenBLAS fill:#FF6B6B,color:#fff
    style Zig fill:#F7A41D,color:#fff
```

## Quick Start

```gleam
import viva_tensor as t

// Create tensors
let a = t.zeros([1000, 1000])
let b = t.random_uniform([1000, 1000])

// Matrix multiplication @ 649 GFLOPS
let c = t.matmul(a, b)

// Activations (SIMD vectorized)
let activated = t.relu(c) |> t.sigmoid()
```

## Features

```mermaid
mindmap
  root((viva_tensor))
    Core
      add/sub/mul/div
      sum/mean/max/min
      matmul/transpose
      dot/outer
    Activations
      relu
      sigmoid
      exp/log
      tanh
    CNN
      conv2d
      max_pool2d
      avg_pool2d
      global_avg_pool2d
    Quantization
      INT8 4x
      NF4 7.5x
      AWQ 7.7x
```

### Quantization

```mermaid
flowchart LR
    A[FP32 Tensor<br/>24 GB] -->|quantize| B{Method}
    B -->|INT8| C[6 GB<br/>4× smaller]
    B -->|NF4| D[3.2 GB<br/>7.5× smaller]
    B -->|AWQ| E[3.1 GB<br/>7.7× smaller]

    style A fill:#FF6B6B
    style C fill:#4ECDC4
    style D fill:#2E8B57
    style E fill:#9B59B6
```

| Method | Compression | Quality | Use Case |
|:------:|:-----------:|:-------:|:--------:|
| INT8 | 4× | 96% | Inference |
| NF4 | 7.5× | 99% | QLoRA Fine-tuning |
| AWQ | 7.7× | 97% | Edge Deployment |

## Build

```bash
# Pure Gleam (no native deps)
gleam build && gleam test

# With native acceleration
cd zig_src && zig build
cp zig-out/lib/libviva_tensor_zig.so ../priv/
```

## Documentation

| Language | Link |
|:--------:|:----:|
| English | [docs/en/](docs/en/) |
| Português | [docs/pt-br/](docs/pt-br/) |
| 中文 | [docs/zh-cn/](docs/zh-cn/) |

---

<div align="center">

```mermaid
flowchart LR
    G[Gleam] --> Z[Zig] --> M[Intel MKL]
    style G fill:#FFAFF3,color:#000
    style Z fill:#F7A41D,color:#000
    style M fill:#0071C5,color:#fff
```

**Built with love for the BEAM**

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:006400,100:8B0000&height=80&section=footer" width="100%"/>

</div>
