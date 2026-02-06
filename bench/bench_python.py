"""
NumPy vs PyTorch vs viva_tensor benchmark (Windows native)
"""
import time
import numpy as np

def bench_numpy(sizes):
    print("\n=== NumPy (using: {}) ===".format(np.__config__.show()))

    for size in sizes:
        # Create matrices
        a = np.random.rand(size, size).astype(np.float64)
        b = np.random.rand(size, size).astype(np.float64)

        # Warmup
        _ = a @ b

        # Benchmark
        iters = 3 if size >= 2000 else 5
        start = time.perf_counter()
        for _ in range(iters):
            c = a @ b
        end = time.perf_counter()

        elapsed_us = (end - start) * 1e6 / iters
        flops = 2.0 * size * size * size
        gflops = flops / elapsed_us / 1000

        print(f"  {size}x{size}: {int(elapsed_us)} us ({int(gflops)} GFLOPS)")

def bench_pytorch(sizes):
    try:
        import torch
        print(f"\n=== PyTorch {torch.__version__} (MKL: {torch.backends.mkl.is_available()}) ===")

        # Use CPU
        device = torch.device('cpu')
        torch.set_num_threads(16)

        for size in sizes:
            a = torch.rand(size, size, dtype=torch.float64, device=device)
            b = torch.rand(size, size, dtype=torch.float64, device=device)

            # Warmup
            _ = torch.mm(a, b)

            # Benchmark
            iters = 3 if size >= 2000 else 5
            start = time.perf_counter()
            for _ in range(iters):
                c = torch.mm(a, b)
            end = time.perf_counter()

            elapsed_us = (end - start) * 1e6 / iters
            flops = 2.0 * size * size * size
            gflops = flops / elapsed_us / 1000

            print(f"  {size}x{size}: {int(elapsed_us)} us ({int(gflops)} GFLOPS)")

    except ImportError:
        print("\n=== PyTorch not installed ===")

if __name__ == "__main__":
    print("=" * 60)
    print("Windows Native GEMM Benchmark")
    print("=" * 60)

    # Show NumPy config
    print("\nNumPy version:", np.__version__)
    try:
        print("BLAS info:", np.show_config())
    except:
        pass

    sizes = [500, 1000, 2000, 4000, 5000]

    bench_numpy(sizes)
    bench_pytorch(sizes)

    print("\n" + "=" * 60)
    print("Done!")
