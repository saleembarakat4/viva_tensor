/**
 * cuSPARSELt Wrapper for 2:4 Structured Sparsity
 *
 * RTX 4090 (Ada Lovelace) Tensor Core Performance:
 * - FP16 Dense: 330 TFLOPS
 * - FP16 Sparse (2:4): 660 TFLOPS (2x!)
 * - INT8 Dense: 660 TFLOPS
 * - INT8 Sparse (2:4): 1320 TFLOPS (2x!)
 *
 * 2:4 Sparsity: Keep exactly 2 non-zero values per 4-element block
 * Hardware accelerated on Ada Lovelace Tensor Cores!
 *
 * This file uses dlopen to dynamically load cuSPARSELt at runtime,
 * avoiding compile-time CUDA dependency.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#ifndef _WIN32

#include <dlfcn.h>

/* =========================================================================
 * cuSPARSELt Types and Constants
 * ========================================================================= */

/* Opaque handles */
typedef void* cusparseLtHandle_t;
typedef void* cusparseLtMatDescriptor_t;
typedef void* cusparseLtMatmulDescriptor_t;
typedef void* cusparseLtMatmulAlgSelection_t;
typedef void* cusparseLtMatmulPlan_t;

/* cusparseStatus_t */
typedef enum {
    CUSPARSE_STATUS_SUCCESS = 0,
    CUSPARSE_STATUS_NOT_INITIALIZED = 1,
    CUSPARSE_STATUS_ALLOC_FAILED = 2,
    CUSPARSE_STATUS_INVALID_VALUE = 3,
    CUSPARSE_STATUS_ARCH_MISMATCH = 4,
    CUSPARSE_STATUS_MAPPING_ERROR = 5,
    CUSPARSE_STATUS_EXECUTION_FAILED = 6,
    CUSPARSE_STATUS_INTERNAL_ERROR = 7,
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    CUSPARSE_STATUS_ZERO_PIVOT = 9,
    CUSPARSE_STATUS_NOT_SUPPORTED = 10,
    CUSPARSE_STATUS_INSUFFICIENT_RESOURCES = 11
} cusparseStatus_t;

/* cusparseOrder_t */
typedef enum {
    CUSPARSE_ORDER_COL = 0,
    CUSPARSE_ORDER_ROW = 1
} cusparseOrder_t;

/* cusparseOperation_t */
typedef enum {
    CUSPARSE_OPERATION_NON_TRANSPOSE = 0,
    CUSPARSE_OPERATION_TRANSPOSE = 1,
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
} cusparseOperation_t;

/* cusparseLtSparsity_t */
typedef enum {
    CUSPARSELT_SPARSITY_50_PERCENT = 0  /* 2:4 structured sparsity */
} cusparseLtSparsity_t;

/* cusparseLtMatmulAlg_t */
typedef enum {
    CUSPARSELT_MATMUL_ALG_DEFAULT = 0
} cusparseLtMatmulAlg_t;

/* cusparseLtPruneAlg_t */
typedef enum {
    CUSPARSELT_PRUNE_SPMMA_TILE = 0,   /* L1-norm tile-based pruning */
    CUSPARSELT_PRUNE_SPMMA_STRIP = 1   /* Strip-based pruning */
} cusparseLtPruneAlg_t;

/* cudaDataType (from CUDA) */
typedef enum {
    CUDA_R_16F = 2,   /* FP16 */
    CUDA_R_32F = 0,   /* FP32 */
    CUDA_R_8I = 3,    /* INT8 */
    CUDA_R_32I = 10,  /* INT32 */
    CUDA_R_16BF = 14  /* BF16 */
} cudaDataType;

/* cusparseComputeType - values from cuSPARSELt v0.7.1 */
typedef enum {
    CUSPARSE_COMPUTE_32I = 0,
    CUSPARSE_COMPUTE_16F = 1,
    CUSPARSE_COMPUTE_32F = 2
} cusparseComputeType;

/* cudaStream_t */
typedef void* cudaStream_t;

/* =========================================================================
 * cuSPARSELt Function Pointer Types
 * ========================================================================= */

/* Handle management */
typedef cusparseStatus_t (*cusparseLtInit_fn)(cusparseLtHandle_t* handle);
typedef cusparseStatus_t (*cusparseLtDestroy_fn)(const cusparseLtHandle_t* handle);

/* Matrix descriptor */
typedef cusparseStatus_t (*cusparseLtStructuredDescriptorInit_fn)(
    const cusparseLtHandle_t* handle,
    cusparseLtMatDescriptor_t* matDescr,
    int64_t rows, int64_t cols, int64_t ld,
    uint32_t alignment,
    cudaDataType valueType,
    cusparseOrder_t order,
    cusparseLtSparsity_t sparsity);

typedef cusparseStatus_t (*cusparseLtDenseDescriptorInit_fn)(
    const cusparseLtHandle_t* handle,
    cusparseLtMatDescriptor_t* matDescr,
    int64_t rows, int64_t cols, int64_t ld,
    uint32_t alignment,
    cudaDataType valueType,
    cusparseOrder_t order);

typedef cusparseStatus_t (*cusparseLtMatDescriptorDestroy_fn)(
    const cusparseLtMatDescriptor_t* matDescr);

/* Matmul descriptor */
typedef cusparseStatus_t (*cusparseLtMatmulDescriptorInit_fn)(
    const cusparseLtHandle_t* handle,
    cusparseLtMatmulDescriptor_t* matmulDescr,
    cusparseOperation_t opA,
    cusparseOperation_t opB,
    const cusparseLtMatDescriptor_t* matA,
    const cusparseLtMatDescriptor_t* matB,
    const cusparseLtMatDescriptor_t* matC,
    const cusparseLtMatDescriptor_t* matD,
    cusparseComputeType computeType);

/* Algorithm selection */
typedef cusparseStatus_t (*cusparseLtMatmulAlgSelectionInit_fn)(
    const cusparseLtHandle_t* handle,
    cusparseLtMatmulAlgSelection_t* algSelection,
    const cusparseLtMatmulDescriptor_t* matmulDescr,
    cusparseLtMatmulAlg_t alg);

/* Plan */
typedef cusparseStatus_t (*cusparseLtMatmulPlanInit_fn)(
    const cusparseLtHandle_t* handle,
    cusparseLtMatmulPlan_t* plan,
    const cusparseLtMatmulDescriptor_t* matmulDescr,
    const cusparseLtMatmulAlgSelection_t* algSelection);

typedef cusparseStatus_t (*cusparseLtMatmulPlanDestroy_fn)(
    const cusparseLtMatmulPlan_t* plan);

/* Pruning */
typedef cusparseStatus_t (*cusparseLtSpMMAPrune_fn)(
    const cusparseLtHandle_t* handle,
    const cusparseLtMatmulDescriptor_t* matmulDescr,
    const void* d_in,
    void* d_out,
    cusparseLtPruneAlg_t pruneAlg,
    cudaStream_t stream);

typedef cusparseStatus_t (*cusparseLtSpMMAPruneCheck_fn)(
    const cusparseLtHandle_t* handle,
    const cusparseLtMatmulDescriptor_t* matmulDescr,
    const void* d_in,
    int* d_valid,
    cudaStream_t stream);

/* Compression */
typedef cusparseStatus_t (*cusparseLtSpMMACompressedSize_fn)(
    const cusparseLtHandle_t* handle,
    const cusparseLtMatmulPlan_t* plan,
    size_t* compressedSize,
    size_t* compressBufferSize);

typedef cusparseStatus_t (*cusparseLtSpMMACompress_fn)(
    const cusparseLtHandle_t* handle,
    const cusparseLtMatmulPlan_t* plan,
    const void* d_dense,
    void* d_compressed,
    void* d_compressBuffer,
    cudaStream_t stream);

/* Workspace */
typedef cusparseStatus_t (*cusparseLtMatmulGetWorkspace_fn)(
    const cusparseLtHandle_t* handle,
    const cusparseLtMatmulPlan_t* plan,
    size_t* workspaceSize);

/* Matmul execution */
typedef cusparseStatus_t (*cusparseLtMatmul_fn)(
    const cusparseLtHandle_t* handle,
    const cusparseLtMatmulPlan_t* plan,
    const void* alpha,
    const void* d_A,
    const void* d_B,
    const void* beta,
    const void* d_C,
    void* d_D,
    void* workspace,
    cudaStream_t* streams,
    int32_t numStreams);

/* =========================================================================
 * Global State
 * ========================================================================= */

static void* g_cusparselt_lib = NULL;
static int g_cusparselt_available = -1;

/* Persistent handle for efficiency */
static char g_cusparselt_handle_storage[4096];  /* Large enough for opaque handle */
static cusparseLtHandle_t g_cusparselt_handle = NULL;

/* Function pointers */
static cusparseLtInit_fn g_cusparseLtInit = NULL;
static cusparseLtDestroy_fn g_cusparseLtDestroy = NULL;
static cusparseLtStructuredDescriptorInit_fn g_cusparseLtStructuredDescriptorInit = NULL;
static cusparseLtDenseDescriptorInit_fn g_cusparseLtDenseDescriptorInit = NULL;
static cusparseLtMatDescriptorDestroy_fn g_cusparseLtMatDescriptorDestroy = NULL;
static cusparseLtMatmulDescriptorInit_fn g_cusparseLtMatmulDescriptorInit = NULL;
static cusparseLtMatmulAlgSelectionInit_fn g_cusparseLtMatmulAlgSelectionInit = NULL;
static cusparseLtMatmulPlanInit_fn g_cusparseLtMatmulPlanInit = NULL;
static cusparseLtMatmulPlanDestroy_fn g_cusparseLtMatmulPlanDestroy = NULL;
static cusparseLtSpMMAPrune_fn g_cusparseLtSpMMAPrune = NULL;
static cusparseLtSpMMAPruneCheck_fn g_cusparseLtSpMMAPruneCheck = NULL;
static cusparseLtSpMMACompressedSize_fn g_cusparseLtSpMMACompressedSize = NULL;
static cusparseLtSpMMACompress_fn g_cusparseLtSpMMACompress = NULL;
static cusparseLtMatmulGetWorkspace_fn g_cusparseLtMatmulGetWorkspace = NULL;
static cusparseLtMatmul_fn g_cusparseLtMatmul = NULL;

/* CUDA functions we need (from cuda_gemm.c or loaded directly) */
extern int cuda_available(void);
extern void* cuda_tensor_alloc_fp16(size_t num_elements);
extern void cuda_tensor_free(void* ptr);

/* CUDA function types (from cuda_gemm.c) */
typedef int cudaError_t;
typedef cudaError_t (*cuda_malloc_fn)(void**, size_t);
typedef cudaError_t (*cuda_free_fn)(void*);
typedef cudaError_t (*cuda_memcpy_fn)(void*, const void*, size_t, int);
typedef cudaError_t (*cuda_device_sync_fn)(void);

/* We'll reuse CUDA functions from cuda_gemm.c */
extern cuda_malloc_fn g_cuda_malloc;
extern cuda_free_fn g_cuda_free;
extern cuda_memcpy_fn g_cuda_memcpy;
extern cuda_device_sync_fn g_cuda_sync;

#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
#define cudaMemcpyDeviceToDevice 3

/* =========================================================================
 * cuSPARSELt Initialization
 * ========================================================================= */

/**
 * Initialize cuSPARSELt library via dlopen
 * Returns: 1 if available, 0 if not
 */
int cusparselt_init(void) {
    if (g_cusparselt_available >= 0) return g_cusparselt_available;

    /* Check CUDA is available first */
    if (!cuda_available()) {
        fprintf(stderr, "[viva_tensor] cuSPARSELt: CUDA not available\n");
        g_cusparselt_available = 0;
        return 0;
    }

    /* Try to load cuSPARSELt library */
    g_cusparselt_lib = dlopen("libcusparseLt.so", RTLD_LAZY);
    if (!g_cusparselt_lib) {
        g_cusparselt_lib = dlopen("libcusparseLt.so.0", RTLD_LAZY);
    }
    if (!g_cusparselt_lib) {
        /* Try CUDA toolkit path */
        g_cusparselt_lib = dlopen("/usr/local/cuda/lib64/libcusparseLt.so", RTLD_LAZY);
    }
    if (!g_cusparselt_lib) {
        fprintf(stderr, "[viva_tensor] cuSPARSELt: library not found\n");
        g_cusparselt_available = 0;
        return 0;
    }

    /* Load all function pointers */
    #define LOAD_FN(name) \
        g_##name = (name##_fn)dlsym(g_cusparselt_lib, #name); \
        if (!g_##name) { \
            fprintf(stderr, "[viva_tensor] cuSPARSELt: failed to load %s\n", #name); \
            dlclose(g_cusparselt_lib); \
            g_cusparselt_lib = NULL; \
            g_cusparselt_available = 0; \
            return 0; \
        }

    LOAD_FN(cusparseLtInit);
    LOAD_FN(cusparseLtDestroy);
    LOAD_FN(cusparseLtStructuredDescriptorInit);
    LOAD_FN(cusparseLtDenseDescriptorInit);
    LOAD_FN(cusparseLtMatDescriptorDestroy);
    LOAD_FN(cusparseLtMatmulDescriptorInit);
    LOAD_FN(cusparseLtMatmulAlgSelectionInit);
    LOAD_FN(cusparseLtMatmulPlanInit);
    LOAD_FN(cusparseLtMatmulPlanDestroy);
    LOAD_FN(cusparseLtSpMMAPrune);
    LOAD_FN(cusparseLtSpMMAPruneCheck);
    LOAD_FN(cusparseLtSpMMACompressedSize);
    LOAD_FN(cusparseLtSpMMACompress);
    LOAD_FN(cusparseLtMatmulGetWorkspace);
    LOAD_FN(cusparseLtMatmul);

    #undef LOAD_FN

    /* Initialize persistent handle */
    g_cusparselt_handle = (cusparseLtHandle_t)g_cusparselt_handle_storage;
    cusparseStatus_t status = g_cusparseLtInit(g_cusparselt_handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] cuSPARSELt: init failed with status %d\n", status);
        dlclose(g_cusparselt_lib);
        g_cusparselt_lib = NULL;
        g_cusparselt_available = 0;
        return 0;
    }

    fprintf(stderr, "[viva_tensor] cuSPARSELt loaded - 2:4 Sparsity Tensor Cores ready!\n");
    g_cusparselt_available = 1;
    return 1;
}

/**
 * Check if cuSPARSELt is available
 */
int cusparselt_available(void) {
    if (g_cusparselt_available < 0) cusparselt_init();
    return g_cusparselt_available;
}

/* =========================================================================
 * SparseTensor Data Structure
 *
 * Holds the compressed 2:4 sparse matrix on GPU
 * ========================================================================= */

typedef struct {
    void* d_compressed;         /* Compressed sparse data on GPU */
    void* d_workspace;          /* Workspace for matmul */
    size_t compressed_size;     /* Size of compressed data */
    size_t workspace_size;      /* Size of workspace */
    int64_t rows;               /* Original rows (M) */
    int64_t cols;               /* Original cols (K) */
    cudaDataType dtype;         /* Data type (FP16, INT8) */

    /* cuSPARSELt descriptors - stored inline for efficiency */
    char mat_descr_storage[1024];
    char matmul_descr_storage[1024];
    char alg_sel_storage[1024];
    char plan_storage[1024];
} SparseTensorInternal;

/**
 * Create a 2:4 sparse tensor from dense FP16 data on GPU
 *
 * This function:
 * 1. Prunes the dense matrix to 2:4 pattern (keeps 2 largest per 4)
 * 2. Compresses to ~50% size with metadata
 * 3. Pre-allocates workspace for efficient matmul
 *
 * @param d_dense       Dense FP16 data on GPU (will NOT be modified)
 * @param rows          Number of rows (M)
 * @param cols          Number of columns (K)
 * @param out_sparse    Output sparse tensor structure
 * @return              0 on success, negative on error
 */
int sparse_tensor_create_fp16(
    const uint16_t* d_dense,
    int64_t rows,
    int64_t cols,
    SparseTensorInternal* out_sparse
) {
    if (!cusparselt_available()) return -1;
    if (!d_dense || !out_sparse) return -2;

    /* Dimensions must be multiples of 16 for FP16 */
    if (rows % 16 != 0 || cols % 16 != 0) {
        fprintf(stderr, "[viva_tensor] SparseTensor: FP16 requires dims multiples of 16 (got %ldx%ld)\n",
                rows, cols);
        return -3;
    }

    cusparseStatus_t status;
    int64_t ld = cols;  /* Leading dimension (row-major) */

    /* Initialize output structure */
    memset(out_sparse, 0, sizeof(SparseTensorInternal));
    out_sparse->rows = rows;
    out_sparse->cols = cols;
    out_sparse->dtype = CUDA_R_16F;

    /* Create sparse matrix descriptor (A) */
    cusparseLtMatDescriptor_t* matA = (cusparseLtMatDescriptor_t*)out_sparse->mat_descr_storage;
    status = g_cusparseLtStructuredDescriptorInit(
        g_cusparselt_handle,
        matA,
        rows, cols, ld,
        16,                             /* 16-byte alignment */
        CUDA_R_16F,                     /* FP16 */
        CUSPARSE_ORDER_ROW,             /* Row-major */
        CUSPARSELT_SPARSITY_50_PERCENT  /* 2:4 sparsity */
    );
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] SparseTensor: StructuredDescriptorInit failed: %d\n", status);
        return -4;
    }

    /* Create dummy dense descriptors for B, C, D to initialize matmul descriptor */
    char matB_storage[1024], matC_storage[1024];
    cusparseLtMatDescriptor_t* matB = (cusparseLtMatDescriptor_t*)matB_storage;
    cusparseLtMatDescriptor_t* matC = (cusparseLtMatDescriptor_t*)matC_storage;

    /* B: cols x cols (square for now, will be adjusted in matmul) */
    /* C/D: rows x cols */
    status = g_cusparseLtDenseDescriptorInit(
        g_cusparselt_handle, matB,
        cols, cols, cols, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW
    );
    status = g_cusparseLtDenseDescriptorInit(
        g_cusparselt_handle, matC,
        rows, cols, cols, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW
    );

    /* Create matmul descriptor */
    cusparseLtMatmulDescriptor_t* matmul = (cusparseLtMatmulDescriptor_t*)out_sparse->matmul_descr_storage;
    status = g_cusparseLtMatmulDescriptorInit(
        g_cusparselt_handle,
        matmul,
        CUSPARSE_OPERATION_NON_TRANSPOSE,  /* opA */
        CUSPARSE_OPERATION_NON_TRANSPOSE,  /* opB */
        matA, matB, matC, matC,
        CUSPARSE_COMPUTE_32F               /* Try FP32 compute */
    );
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] SparseTensor: MatmulDescriptorInit failed: %d\n", status);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -5;
    }

    /* Algorithm selection */
    cusparseLtMatmulAlgSelection_t* alg = (cusparseLtMatmulAlgSelection_t*)out_sparse->alg_sel_storage;
    status = g_cusparseLtMatmulAlgSelectionInit(
        g_cusparselt_handle,
        alg,
        matmul,
        CUSPARSELT_MATMUL_ALG_DEFAULT
    );
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] SparseTensor: AlgSelectionInit failed: %d\n", status);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -6;
    }

    /* Create plan */
    cusparseLtMatmulPlan_t* plan = (cusparseLtMatmulPlan_t*)out_sparse->plan_storage;
    status = g_cusparseLtMatmulPlanInit(
        g_cusparselt_handle,
        plan,
        matmul,
        alg
    );
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] SparseTensor: MatmulPlanInit failed: %d\n", status);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -7;
    }

    /* Allocate temporary buffer for pruned data */
    size_t dense_size = rows * cols * sizeof(uint16_t);
    void* d_pruned;
    if (g_cuda_malloc(&d_pruned, dense_size) != 0) {
        g_cusparseLtMatmulPlanDestroy(plan);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -7;
    }

    /* Copy dense data to pruned buffer (prune modifies in-place) */
    g_cuda_memcpy(d_pruned, d_dense, dense_size, cudaMemcpyDeviceToDevice);

    /* Prune to 2:4 pattern! */
    status = g_cusparseLtSpMMAPrune(
        g_cusparselt_handle,
        matmul,
        d_pruned,
        d_pruned,                        /* In-place */
        CUSPARSELT_PRUNE_SPMMA_TILE,     /* Tile-based L1-norm pruning */
        NULL                             /* Default stream */
    );
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] SparseTensor: Prune failed: %d\n", status);
        g_cuda_free(d_pruned);
        g_cusparseLtMatmulPlanDestroy(plan);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -8;
    }

    /* Get compressed size */
    size_t compress_buffer_size;
    status = g_cusparseLtSpMMACompressedSize(
        g_cusparselt_handle,
        plan,
        &out_sparse->compressed_size,
        &compress_buffer_size
    );
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] SparseTensor: CompressedSize failed: %d\n", status);
        g_cuda_free(d_pruned);
        g_cusparseLtMatmulPlanDestroy(plan);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -9;
    }

    /* Allocate compressed buffer and workspace */
    if (g_cuda_malloc(&out_sparse->d_compressed, out_sparse->compressed_size) != 0) {
        g_cuda_free(d_pruned);
        g_cusparseLtMatmulPlanDestroy(plan);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -10;
    }

    void* d_compress_buffer;
    if (g_cuda_malloc(&d_compress_buffer, compress_buffer_size) != 0) {
        g_cuda_free(out_sparse->d_compressed);
        g_cuda_free(d_pruned);
        g_cusparseLtMatmulPlanDestroy(plan);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -10;
    }

    /* Compress! */
    status = g_cusparseLtSpMMACompress(
        g_cusparselt_handle,
        plan,
        d_pruned,
        out_sparse->d_compressed,
        d_compress_buffer,
        NULL  /* Default stream */
    );

    g_cuda_free(d_compress_buffer);
    g_cuda_free(d_pruned);

    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] SparseTensor: Compress failed: %d\n", status);
        g_cuda_free(out_sparse->d_compressed);
        g_cusparseLtMatmulPlanDestroy(plan);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -11;
    }

    /* Get workspace size for matmul */
    status = g_cusparseLtMatmulGetWorkspace(
        g_cusparselt_handle,
        plan,
        &out_sparse->workspace_size
    );

    /* Allocate workspace */
    if (out_sparse->workspace_size > 0) {
        if (g_cuda_malloc(&out_sparse->d_workspace, out_sparse->workspace_size) != 0) {
            g_cuda_free(out_sparse->d_compressed);
            g_cusparseLtMatmulPlanDestroy(plan);
            g_cusparseLtMatDescriptorDestroy(matA);
            return -12;
        }
    }

    g_cuda_sync();

    fprintf(stderr, "[viva_tensor] SparseTensor created: %ldx%ld -> %zu bytes (%.1f%% of dense)\n",
            rows, cols, out_sparse->compressed_size,
            100.0 * out_sparse->compressed_size / dense_size);

    return 0;
}

/**
 * Free a sparse tensor
 */
void sparse_tensor_free(SparseTensorInternal* sparse) {
    if (!sparse) return;

    if (sparse->d_compressed) {
        g_cuda_free(sparse->d_compressed);
        sparse->d_compressed = NULL;
    }
    if (sparse->d_workspace) {
        g_cuda_free(sparse->d_workspace);
        sparse->d_workspace = NULL;
    }

    /* Cleanup cuSPARSELt objects */
    cusparseLtMatmulPlan_t* plan = (cusparseLtMatmulPlan_t*)sparse->plan_storage;
    cusparseLtMatDescriptor_t* matA = (cusparseLtMatDescriptor_t*)sparse->mat_descr_storage;

    if (g_cusparseLtMatmulPlanDestroy && plan) {
        g_cusparseLtMatmulPlanDestroy(plan);
    }
    if (g_cusparseLtMatDescriptorDestroy && matA) {
        g_cusparseLtMatDescriptorDestroy(matA);
    }
}

/**
 * Sparse GEMM: C = alpha * A_sparse @ B_dense + beta * C
 *
 * A is the 2:4 sparse matrix (created with sparse_tensor_create_fp16)
 * B and C are dense FP16 matrices on GPU
 *
 * RTX 4090: 660 TFLOPS FP16 with 2:4 sparsity (2x of dense 330T!)
 *
 * @param sparse    SparseTensor (A)
 * @param d_B       Dense FP16 B [K x N] on GPU
 * @param d_C       Dense FP16 C [M x N] on GPU (input/output)
 * @param N         Number of columns in B and C
 * @param alpha     Scaling factor
 * @param beta      Scaling factor for C
 * @return          0 on success
 */
int sparse_matmul_fp16(
    SparseTensorInternal* sparse,
    const uint16_t* d_B,
    uint16_t* d_C,
    int64_t N,
    float alpha,
    float beta
) {
    if (!cusparselt_available()) return -1;
    if (!sparse || !d_B || !d_C) return -2;

    /* N must be multiple of 16 for FP16 */
    if (N % 16 != 0) {
        fprintf(stderr, "[viva_tensor] sparse_matmul: N must be multiple of 16 (got %ld)\n", N);
        return -3;
    }

    int64_t M = sparse->rows;
    int64_t K = sparse->cols;

    /* We need to create proper B, C descriptors for this specific N */
    cusparseStatus_t status;

    char matB_storage[1024], matC_storage[1024];
    cusparseLtMatDescriptor_t* matB = (cusparseLtMatDescriptor_t*)matB_storage;
    cusparseLtMatDescriptor_t* matC = (cusparseLtMatDescriptor_t*)matC_storage;

    status = g_cusparseLtDenseDescriptorInit(
        g_cusparselt_handle, matB,
        K, N, N, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW
    );
    if (status != CUSPARSE_STATUS_SUCCESS) return -4;

    status = g_cusparseLtDenseDescriptorInit(
        g_cusparselt_handle, matC,
        M, N, N, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW
    );
    if (status != CUSPARSE_STATUS_SUCCESS) {
        g_cusparseLtMatDescriptorDestroy(matB);
        return -5;
    }

    /* Create matmul descriptor with correct dimensions */
    char matmul_storage[1024], alg_storage[1024], plan_storage[1024];
    cusparseLtMatmulDescriptor_t* matmul = (cusparseLtMatmulDescriptor_t*)matmul_storage;
    cusparseLtMatmulAlgSelection_t* alg = (cusparseLtMatmulAlgSelection_t*)alg_storage;
    cusparseLtMatmulPlan_t* plan = (cusparseLtMatmulPlan_t*)plan_storage;

    cusparseLtMatDescriptor_t* matA = (cusparseLtMatDescriptor_t*)sparse->mat_descr_storage;

    status = g_cusparseLtMatmulDescriptorInit(
        g_cusparselt_handle, matmul,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        matA, matB, matC, matC,
        CUSPARSE_COMPUTE_32F
    );
    if (status != CUSPARSE_STATUS_SUCCESS) {
        g_cusparseLtMatDescriptorDestroy(matB);
        g_cusparseLtMatDescriptorDestroy(matC);
        return -6;
    }

    status = g_cusparseLtMatmulAlgSelectionInit(
        g_cusparselt_handle, alg, matmul, CUSPARSELT_MATMUL_ALG_DEFAULT
    );

    status = g_cusparseLtMatmulPlanInit(
        g_cusparselt_handle, plan, matmul, alg
    );
    if (status != CUSPARSE_STATUS_SUCCESS) {
        g_cusparseLtMatDescriptorDestroy(matB);
        g_cusparseLtMatDescriptorDestroy(matC);
        return -7;
    }

    /* Get workspace for this specific matmul */
    size_t workspace_size;
    g_cusparseLtMatmulGetWorkspace(g_cusparselt_handle, plan, &workspace_size);

    void* workspace = NULL;
    if (workspace_size > 0) {
        g_cuda_malloc(&workspace, workspace_size);
    }

    /* Execute sparse GEMM! */
    status = g_cusparseLtMatmul(
        g_cusparselt_handle,
        plan,
        &alpha,
        sparse->d_compressed,  /* Compressed sparse A */
        d_B,                   /* Dense B */
        &beta,
        d_C,                   /* Input C */
        d_C,                   /* Output D */
        workspace,
        NULL, 0                /* Default stream */
    );

    if (workspace) g_cuda_free(workspace);
    g_cusparseLtMatmulPlanDestroy(plan);
    g_cusparseLtMatDescriptorDestroy(matB);
    g_cusparseLtMatDescriptorDestroy(matC);

    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] sparse_matmul: Matmul failed: %d\n", status);
        return -8;
    }

    g_cuda_sync();
    return 0;
}

#else /* _WIN32 */

/* Windows stubs - cuSPARSELt not supported via this code path */

typedef struct {
    void* d_compressed;
    void* d_workspace;
    size_t compressed_size;
    size_t workspace_size;
    int64_t rows;
    int64_t cols;
    int dtype;
    char mat_descr_storage[1024];
    char matmul_descr_storage[1024];
    char alg_sel_storage[1024];
    char plan_storage[1024];
} SparseTensorInternal;

int cusparselt_init(void) { return 0; }
int cusparselt_available(void) { return 0; }

int sparse_tensor_create_fp16(
    const uint16_t* d_dense, int64_t rows, int64_t cols,
    SparseTensorInternal* out_sparse
) {
    (void)d_dense; (void)rows; (void)cols; (void)out_sparse;
    return -1;
}

void sparse_tensor_free(SparseTensorInternal* sparse) {
    (void)sparse;
}

int sparse_matmul_fp16(
    SparseTensorInternal* sparse, const uint16_t* d_B, uint16_t* d_C,
    int64_t N, float alpha, float beta
) {
    (void)sparse; (void)d_B; (void)d_C; (void)N; (void)alpha; (void)beta;
    return -1;
}

#endif /* _WIN32 */
