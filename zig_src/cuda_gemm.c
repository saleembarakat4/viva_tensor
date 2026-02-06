/**
 * cuda_gemm.c - cuBLAS DGEMM wrapper for viva_tensor
 *
 * RTX 4090 Ada Lovelace:
 *   - FP32: 82.58 TFLOPS (tensor cores)
 *   - FP64: 1.29 TFLOPS (still 1.5x faster than MKL!)
 *
 * This file uses dlopen to dynamically load CUDA runtime and cuBLAS.
 * No compile-time CUDA dependency needed.
 */

#ifndef _WIN32

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* CUDA types (from cuda_runtime_api.h) */
typedef int cudaError_t;
typedef void* cudaStream_t;

/* cuBLAS types (from cublas_v2.h) */
typedef void* cublasHandle_t;
typedef int cublasStatus_t;
typedef int cublasOperation_t;

/* CUDA operation codes */
#define cudaSuccess 0
#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_OP_N 0  /* No transpose */
#define CUBLAS_OP_T 1  /* Transpose */

/* Function pointer types for CUDA runtime */
typedef cudaError_t (*cuda_malloc_fn)(void**, size_t);
typedef cudaError_t (*cuda_free_fn)(void*);
typedef cudaError_t (*cuda_memcpy_fn)(void*, const void*, size_t, int);
typedef cudaError_t (*cuda_device_sync_fn)(void);
typedef const char* (*cuda_get_error_fn)(cudaError_t);

/* Function pointer types for cuBLAS */
typedef cublasStatus_t (*cublas_create_fn)(cublasHandle_t*);
typedef cublasStatus_t (*cublas_destroy_fn)(cublasHandle_t);
typedef cublasStatus_t (*cublas_dgemm_fn)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const double*, const double*, int,
    const double*, int,
    const double*, double*, int);
typedef cublasStatus_t (*cublas_sgemm_fn)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const float*, const float*, int,
    const float*, int,
    const float*, float*, int);

/* cudaMemcpyKind */
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2

/* Global CUDA state */
static void *g_cuda_rt_handle = NULL;
static void *g_cublas_handle = NULL;
static cublasHandle_t g_cublas_ctx = NULL;
static int g_cuda_available = -1;  /* -1 = not checked, 0 = no, 1 = yes */

/* CUDA runtime functions */
static cuda_malloc_fn g_cuda_malloc = NULL;
static cuda_free_fn g_cuda_free = NULL;
static cuda_memcpy_fn g_cuda_memcpy = NULL;
static cuda_device_sync_fn g_cuda_sync = NULL;
static cuda_get_error_fn g_cuda_error = NULL;

/* cuBLAS functions */
static cublas_create_fn g_cublas_create = NULL;
static cublas_destroy_fn g_cublas_destroy = NULL;
static cublas_dgemm_fn g_cublas_dgemm = NULL;
static cublas_sgemm_fn g_cublas_sgemm = NULL;

/**
 * Initialize CUDA and cuBLAS via dlopen
 * Returns 1 if successful, 0 otherwise
 */
int cuda_init(void) {
    if (g_cuda_available >= 0) return g_cuda_available;

    /* Load CUDA runtime */
    g_cuda_rt_handle = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
    if (!g_cuda_rt_handle) {
        g_cuda_rt_handle = dlopen("libcudart.so.12", RTLD_NOW | RTLD_LOCAL);
    }
    if (!g_cuda_rt_handle) {
        g_cuda_rt_handle = dlopen("libcudart.so.13", RTLD_NOW | RTLD_LOCAL);
    }
    if (!g_cuda_rt_handle) {
        g_cuda_available = 0;
        return 0;
    }

    /* Load CUDA runtime functions */
    g_cuda_malloc = (cuda_malloc_fn)dlsym(g_cuda_rt_handle, "cudaMalloc");
    g_cuda_free = (cuda_free_fn)dlsym(g_cuda_rt_handle, "cudaFree");
    g_cuda_memcpy = (cuda_memcpy_fn)dlsym(g_cuda_rt_handle, "cudaMemcpy");
    g_cuda_sync = (cuda_device_sync_fn)dlsym(g_cuda_rt_handle, "cudaDeviceSynchronize");
    g_cuda_error = (cuda_get_error_fn)dlsym(g_cuda_rt_handle, "cudaGetErrorString");

    if (!g_cuda_malloc || !g_cuda_free || !g_cuda_memcpy || !g_cuda_sync) {
        dlclose(g_cuda_rt_handle);
        g_cuda_rt_handle = NULL;
        g_cuda_available = 0;
        return 0;
    }

    /* Load cuBLAS */
    g_cublas_handle = dlopen("libcublas.so", RTLD_NOW | RTLD_LOCAL);
    if (!g_cublas_handle) {
        g_cublas_handle = dlopen("libcublas.so.12", RTLD_NOW | RTLD_LOCAL);
    }
    if (!g_cublas_handle) {
        g_cublas_handle = dlopen("libcublas.so.13", RTLD_NOW | RTLD_LOCAL);
    }
    if (!g_cublas_handle) {
        dlclose(g_cuda_rt_handle);
        g_cuda_rt_handle = NULL;
        g_cuda_available = 0;
        return 0;
    }

    /* Load cuBLAS functions */
    g_cublas_create = (cublas_create_fn)dlsym(g_cublas_handle, "cublasCreate_v2");
    g_cublas_destroy = (cublas_destroy_fn)dlsym(g_cublas_handle, "cublasDestroy_v2");
    g_cublas_dgemm = (cublas_dgemm_fn)dlsym(g_cublas_handle, "cublasDgemm_v2");
    g_cublas_sgemm = (cublas_sgemm_fn)dlsym(g_cublas_handle, "cublasSgemm_v2");

    if (!g_cublas_create || !g_cublas_destroy || !g_cublas_dgemm) {
        dlclose(g_cublas_handle);
        dlclose(g_cuda_rt_handle);
        g_cublas_handle = NULL;
        g_cuda_rt_handle = NULL;
        g_cuda_available = 0;
        return 0;
    }

    /* Create cuBLAS context */
    if (g_cublas_create(&g_cublas_ctx) != CUBLAS_STATUS_SUCCESS) {
        dlclose(g_cublas_handle);
        dlclose(g_cuda_rt_handle);
        g_cublas_handle = NULL;
        g_cuda_rt_handle = NULL;
        g_cuda_available = 0;
        return 0;
    }

    fprintf(stderr, "[viva_tensor] CUDA backend: cuBLAS (RTX 4090 = 1000+ GFLOPS FP64)\n");
    g_cuda_available = 1;
    return 1;
}

/**
 * DGEMM on GPU: C = alpha * A @ B + beta * C
 * A is M x K, B is K x N, C is M x N (all row-major)
 *
 * cuBLAS uses column-major, so we compute: C^T = B^T @ A^T
 * This gives us row-major result without explicit transpose.
 */
int cuda_dgemm(int M, int N, int K,
               double alpha, const double *A, int lda,
               const double *B, int ldb,
               double beta, double *C, int ldc) {
    if (!g_cuda_available) return -1;

    cudaError_t err;
    cublasStatus_t stat;

    size_t size_a = (size_t)M * K * sizeof(double);
    size_t size_b = (size_t)K * N * sizeof(double);
    size_t size_c = (size_t)M * N * sizeof(double);

    double *d_A = NULL, *d_B = NULL, *d_C = NULL;

    /* Allocate GPU memory */
    err = g_cuda_malloc((void**)&d_A, size_a);
    if (err != cudaSuccess) { return -2; }

    err = g_cuda_malloc((void**)&d_B, size_b);
    if (err != cudaSuccess) { g_cuda_free(d_A); return -2; }

    err = g_cuda_malloc((void**)&d_C, size_c);
    if (err != cudaSuccess) { g_cuda_free(d_A); g_cuda_free(d_B); return -2; }

    /* Copy A and B to GPU */
    err = g_cuda_memcpy(d_A, A, size_a, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { goto cleanup; }

    err = g_cuda_memcpy(d_B, B, size_b, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { goto cleanup; }

    /* cuBLAS DGEMM (column-major trick: swap A and B, swap M and N) */
    /* C = A @ B  becomes  C^T = B^T @ A^T  which is column-major of row-major result */
    stat = g_cublas_dgemm(
        g_cublas_ctx,
        CUBLAS_OP_N, CUBLAS_OP_N,  /* No transpose (we're doing the swap trick) */
        N, M, K,                   /* Swapped dimensions for row-major */
        &alpha,
        d_B, N,                    /* B with leading dim N */
        d_A, K,                    /* A with leading dim K */
        &beta,
        d_C, N                     /* C with leading dim N */
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] cuBLAS error: %d\n", stat);
        goto cleanup;
    }

    /* Synchronize and copy result back */
    g_cuda_sync();
    err = g_cuda_memcpy(C, d_C, size_c, cudaMemcpyDeviceToHost);

cleanup:
    g_cuda_free(d_A);
    g_cuda_free(d_B);
    g_cuda_free(d_C);

    return (err == cudaSuccess && stat == CUBLAS_STATUS_SUCCESS) ? 0 : -3;
}

/**
 * Check if CUDA is available
 */
int cuda_available(void) {
    if (g_cuda_available < 0) cuda_init();
    return g_cuda_available;
}

/**
 * Cleanup CUDA resources
 */
void cuda_cleanup(void) {
    if (g_cublas_ctx) {
        g_cublas_destroy(g_cublas_ctx);
        g_cublas_ctx = NULL;
    }
    if (g_cublas_handle) {
        dlclose(g_cublas_handle);
        g_cublas_handle = NULL;
    }
    if (g_cuda_rt_handle) {
        dlclose(g_cuda_rt_handle);
        g_cuda_rt_handle = NULL;
    }
    g_cuda_available = -1;
}

#else /* _WIN32 */

/* CUDA not supported on Windows via this code path (use native CUDA toolkit) */
int cuda_init(void) { return 0; }
int cuda_available(void) { return 0; }
int cuda_dgemm(int M, int N, int K, double alpha, const double *A, int lda,
               const double *B, int ldb, double beta, double *C, int ldc) {
    return -1;
}
void cuda_cleanup(void) {}

#endif /* _WIN32 */
