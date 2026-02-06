/* _GNU_SOURCE must be defined BEFORE any includes for pthread_setaffinity_np */
#if !defined(_WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1
#endif

/**
 * nif_entry.c - Erlang NIF interface for Zig SIMD tensor operations
 *
 * Two APIs:
 *   1. Legacy list-based (backward compatible): nif_simd_dot, nif_simd_sum,
 * etc.
 *   2. NIF Resource-based (zero-copy): nt_*, operates on native tensor refs
 *
 * NIF Resources keep tensor data in contiguous C arrays. Erlang only holds
 * an opaque reference. No list<->array conversion on every operation.
 * GC calls the destructor to free native memory automatically.
 *
 * Intelligent Thread Management (MKL-style):
 *   - Auto-detects CPU topology on NIF load
 *   - Configures thread affinity for optimal cache usage
 *   - Exports cpu_info struct to Zig for runtime decisions
 */

#include "erl_nif.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
/* Intel MKL for optimized GEMM on Windows (600+ GFLOPS) */
#include <mkl_cblas.h>
#define BLAS_BACKEND_MKL 1
#define BLAS_BACKEND_NAME "Intel MKL"
#else
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <dlfcn.h>  /* Dynamic loading for runtime backend selection */

/* =========================================================================
 * Dynamic BLAS Backend Selection (MKL > OpenBLAS-tuned > OpenBLAS)
 * ========================================================================= */

typedef enum {
  BLAS_MKL = 1,
  BLAS_OPENBLAS_TUNED = 2,
  BLAS_OPENBLAS = 3,
  BLAS_ZIG_GEMM = 4
} BlasBackend;

/* Function pointer type for cblas_dgemm
 * cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
 * CBLAS_ORDER: CblasRowMajor=101, CblasColMajor=102
 * CBLAS_TRANSPOSE: CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113
 */
typedef void (*dgemm_fn)(const int Order, const int TransA, const int TransB,
                         const int M, const int N, const int K,
                         const double alpha, const double *A, const int lda,
                         const double *B, const int ldb,
                         const double beta, double *C, const int ldc);

/* Function pointer type for openblas_set_num_threads */
typedef void (*set_threads_fn)(int);

/* Global BLAS state */
static BlasBackend g_blas_backend = BLAS_ZIG_GEMM;
static void *g_blas_handle = NULL;
static dgemm_fn g_dgemm = NULL;
static set_threads_fn g_set_threads = NULL;
static const char *g_blas_name = "Zig GEMM";
static int g_blas_detected = 0;

/* Try to load a BLAS library dynamically */
static int try_load_blas(const char *libname, const char *backend_name, BlasBackend backend_type) {
  void *handle = dlopen(libname, RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    /* Debug: show why load failed */
    /* fprintf(stderr, "[viva_tensor] dlopen(%s) failed: %s\n", libname, dlerror()); */
    return 0;
  }

  dgemm_fn dgemm = (dgemm_fn)dlsym(handle, "cblas_dgemm");
  if (!dgemm) {
    dlclose(handle);
    return 0;
  }

  g_blas_handle = handle;
  g_dgemm = dgemm;
  g_blas_backend = backend_type;
  g_blas_name = backend_name;

  /* Try to get thread control function */
  if (backend_type == BLAS_MKL) {
    g_set_threads = (set_threads_fn)dlsym(handle, "mkl_set_num_threads");
  } else {
    g_set_threads = (set_threads_fn)dlsym(handle, "openblas_set_num_threads");
  }

  /* Auto-configure optimal thread count (16 is good default for modern CPUs) */
  if (g_set_threads) {
    int ncpus = sysconf(_SC_NPROCESSORS_ONLN);
    int optimal = ncpus > 0 ? ncpus : 16;
    g_set_threads(optimal);
    fprintf(stderr, "[viva_tensor] BLAS threads: %d\n", optimal);
  }

  return 1;
}

/* Detect and load the best available BLAS backend */
static void detect_blas_backend(void) {
  if (g_blas_detected) return;
  g_blas_detected = 1;

  /* Priority: Intel MKL > OpenBLAS-tuned > OpenBLAS system > Zig GEMM */

  /* 1. Try Intel MKL first (800+ GFLOPS on Linux!) */
  if (try_load_blas("libmkl_rt.so", "Intel MKL", BLAS_MKL)) {
    fprintf(stderr, "[viva_tensor] Backend: Intel MKL (800+ GFLOPS)\n");
    return;
  }
  /* Try alternate MKL paths */
  if (try_load_blas("/usr/lib/x86_64-linux-gnu/libmkl_rt.so", "Intel MKL", BLAS_MKL)) {
    fprintf(stderr, "[viva_tensor] Backend: Intel MKL (800+ GFLOPS)\n");
    return;
  }
  if (try_load_blas("/opt/intel/mkl/lib/intel64/libmkl_rt.so", "Intel MKL", BLAS_MKL)) {
    fprintf(stderr, "[viva_tensor] Backend: Intel MKL (800+ GFLOPS)\n");
    return;
  }
  if (try_load_blas("/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so", "Intel MKL", BLAS_MKL)) {
    fprintf(stderr, "[viva_tensor] Backend: Intel MKL (800+ GFLOPS)\n");
    return;
  }

  /* 2. Try our tuned OpenBLAS (HASWELL-optimized, 500+ GFLOPS) */
  char tuned_path[512];
  if (getcwd(tuned_path, sizeof(tuned_path) - 100)) {
    strcat(tuned_path, "/deps/openblas-tuned/lib/libopenblas.so");
  } else {
    strcpy(tuned_path, "deps/openblas-tuned/lib/libopenblas.so");
  }
  if (try_load_blas(tuned_path, "OpenBLAS-HASWELL", BLAS_OPENBLAS_TUNED)) {
    fprintf(stderr, "[viva_tensor] Backend: OpenBLAS-HASWELL (tuned, 500+ GFLOPS)\n");
    return;
  }

  /* 3. Try system OpenBLAS as fallback */
  if (try_load_blas("libopenblas.so.0", "OpenBLAS", BLAS_OPENBLAS)) {
    fprintf(stderr, "[viva_tensor] Backend: OpenBLAS system\n");
    return;
  }
  if (try_load_blas("libopenblas.so", "OpenBLAS", BLAS_OPENBLAS)) {
    fprintf(stderr, "[viva_tensor] Backend: OpenBLAS system\n");
    return;
  }

  /* 4. Fallback to Zig GEMM (still good: 200+ GFLOPS) */
  fprintf(stderr, "[viva_tensor] Backend: Zig GEMM (native, 200+ GFLOPS)\n");
}

/* Call cblas_dgemm via the loaded backend
 * Row-major: lda=K, ldb=N, ldc=N for C = A @ B where A is MxK, B is KxN
 */
static void blas_dgemm(int M, int N, int K, double alpha,
                       const double *A, int lda,
                       const double *B, int ldb,
                       double beta, double *C, int ldc) {
  if (g_dgemm) {
    /* CblasRowMajor=101, CblasNoTrans=111 */
    g_dgemm(101, 111, 111, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
}

/* Set number of threads for BLAS operations */
static void blas_set_threads(int n) {
  if (g_set_threads) {
    g_set_threads(n);
  }
}

#define BLAS_BACKEND_NAME g_blas_name
#endif

/* =========================================================================
 * CPU Topology Detection (MKL-style intelligent runtime)
 * ========================================================================= */

typedef struct {
  int logical_cpus;     /* Total logical CPUs (includes HT) */
  int physical_cores;   /* Physical cores (no HT) */
  int sockets;          /* Number of CPU sockets */
  int l1_cache_kb;      /* L1 data cache per core (KB) */
  int l2_cache_kb;      /* L2 cache per core (KB) */
  int l3_cache_kb;      /* L3 cache total (KB) */
  int has_avx2;         /* AVX2 support */
  int has_avx512;       /* AVX-512 support */
  int has_hybrid;       /* Intel hybrid (P+E cores) */
  int p_cores;          /* Performance cores (if hybrid) */
  int e_cores;          /* Efficiency cores (if hybrid) */
  int threads_per_core; /* HT threads per core */
  int optimal_threads;  /* Computed optimal thread count for GEMM */
} CpuTopology;

/* Global CPU topology - initialized once at NIF load */
static CpuTopology g_cpu_info = {0};
static int g_cpu_detected = 0;

#ifdef _WIN32
static void detect_cpu_topology_windows(void) {
  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);
  g_cpu_info.logical_cpus = sysInfo.dwNumberOfProcessors;

  /* Use GetLogicalProcessorInformation for detailed topology */
  DWORD bufLen = 0;
  GetLogicalProcessorInformation(NULL, &bufLen);
  if (bufLen == 0) {
    g_cpu_info.physical_cores = g_cpu_info.logical_cpus / 2;
    g_cpu_info.threads_per_core = 2;
    goto compute_optimal;
  }

  SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buf =
    (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc(bufLen);
  if (!buf) {
    g_cpu_info.physical_cores = g_cpu_info.logical_cpus / 2;
    g_cpu_info.threads_per_core = 2;
    goto compute_optimal;
  }

  if (GetLogicalProcessorInformation(buf, &bufLen)) {
    int cores = 0, l1 = 0, l2 = 0, l3 = 0;
    DWORD offset = 0;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *ptr = buf;
    while (offset < bufLen) {
      switch (ptr->Relationship) {
        case RelationProcessorCore:
          cores++;
          break;
        case RelationCache:
          if (ptr->Cache.Level == 1 && ptr->Cache.Type == CacheData)
            l1 = ptr->Cache.Size / 1024;
          else if (ptr->Cache.Level == 2)
            l2 = ptr->Cache.Size / 1024;
          else if (ptr->Cache.Level == 3)
            l3 = ptr->Cache.Size / 1024;
          break;
        default:
          break;
      }
      offset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
      ptr++;
    }
    g_cpu_info.physical_cores = cores > 0 ? cores : g_cpu_info.logical_cpus / 2;
    g_cpu_info.l1_cache_kb = l1;
    g_cpu_info.l2_cache_kb = l2;
    g_cpu_info.l3_cache_kb = l3;
  }
  free(buf);

  g_cpu_info.threads_per_core = g_cpu_info.logical_cpus / g_cpu_info.physical_cores;
  g_cpu_info.sockets = 1;

  /* Check for AVX2/AVX-512 via CPUID */
  int cpuInfo[4];
  __cpuid(cpuInfo, 7);
  g_cpu_info.has_avx2 = (cpuInfo[1] & (1 << 5)) != 0;
  g_cpu_info.has_avx512 = (cpuInfo[1] & (1 << 16)) != 0;

  /* Check hybrid architecture (Intel 12th gen+) */
  __cpuid(cpuInfo, 0x1A);
  g_cpu_info.has_hybrid = (cpuInfo[0] != 0);
  if (g_cpu_info.has_hybrid) {
    /* Heuristic: assume ~1/3 are E-cores on typical hybrid CPUs */
    g_cpu_info.p_cores = (g_cpu_info.physical_cores * 2) / 3;
    g_cpu_info.e_cores = g_cpu_info.physical_cores - g_cpu_info.p_cores;
  }

compute_optimal:
  /* Compute optimal threads for GEMM */
  if (g_cpu_info.has_hybrid) {
    /* Hybrid: use P-cores only for max single-thread perf, all cores for throughput */
    g_cpu_info.optimal_threads = g_cpu_info.logical_cpus;
  } else {
    /* Non-hybrid: use all logical CPUs (HT helps hide memory latency) */
    g_cpu_info.optimal_threads = g_cpu_info.logical_cpus;
  }
}
#else /* Linux/macOS */
static void detect_cpu_topology_linux(void) {
  /* Read /proc/cpuinfo for detailed topology */
  FILE *f = fopen("/proc/cpuinfo", "r");
  if (!f) {
    g_cpu_info.logical_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    g_cpu_info.physical_cores = g_cpu_info.logical_cpus / 2;
    g_cpu_info.threads_per_core = 2;
    goto compute_optimal;
  }

  char line[256];
  int logical = 0, phys_ids[64] = {0}, core_ids[64] = {0};
  int unique_phys = 0, unique_cores = 0;
  int l1 = 0, l2 = 0, l3 = 0;
  int avx2 = 0, avx512 = 0;

  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "processor", 9) == 0) {
      logical++;
    } else if (strncmp(line, "physical id", 11) == 0) {
      int id;
      if (sscanf(line, "physical id : %d", &id) == 1 && id < 64) {
        if (!phys_ids[id]) {
          phys_ids[id] = 1;
          unique_phys++;
        }
      }
    } else if (strncmp(line, "core id", 7) == 0) {
      int id;
      if (sscanf(line, "core id : %d", &id) == 1 && id < 64) {
        if (!core_ids[id]) {
          core_ids[id] = 1;
          unique_cores++;
        }
      }
    } else if (strncmp(line, "cache size", 10) == 0) {
      int sz;
      if (sscanf(line, "cache size : %d KB", &sz) == 1) {
        l3 = sz; /* Usually L3 is reported here */
      }
    } else if (strstr(line, "avx2")) {
      avx2 = 1;
    } else if (strstr(line, "avx512")) {
      avx512 = 1;
    }
  }
  fclose(f);

  g_cpu_info.logical_cpus = logical > 0 ? logical : sysconf(_SC_NPROCESSORS_ONLN);
  g_cpu_info.sockets = unique_phys > 0 ? unique_phys : 1;
  g_cpu_info.physical_cores = unique_cores > 0 ? unique_cores * g_cpu_info.sockets
                                               : g_cpu_info.logical_cpus / 2;
  g_cpu_info.threads_per_core = g_cpu_info.logical_cpus / g_cpu_info.physical_cores;
  g_cpu_info.l3_cache_kb = l3;
  g_cpu_info.has_avx2 = avx2;
  g_cpu_info.has_avx512 = avx512;

  /* Try to read cache info from sysfs */
  FILE *cache_f;
  char buf[64];

  cache_f = fopen("/sys/devices/system/cpu/cpu0/cache/index0/size", "r");
  if (cache_f) {
    if (fgets(buf, sizeof(buf), cache_f)) {
      sscanf(buf, "%dK", &l1);
      g_cpu_info.l1_cache_kb = l1;
    }
    fclose(cache_f);
  }

  cache_f = fopen("/sys/devices/system/cpu/cpu0/cache/index2/size", "r");
  if (cache_f) {
    if (fgets(buf, sizeof(buf), cache_f)) {
      sscanf(buf, "%dK", &l2);
      g_cpu_info.l2_cache_kb = l2;
    }
    fclose(cache_f);
  }

  /* Check for Intel hybrid via cpu_capacity (Linux 5.8+) */
  cache_f = fopen("/sys/devices/system/cpu/cpu0/cpu_capacity", "r");
  if (cache_f) {
    int cap;
    if (fgets(buf, sizeof(buf), cache_f) && sscanf(buf, "%d", &cap) == 1) {
      if (cap < 1024) { /* Capacity < 1024 indicates E-core */
        g_cpu_info.has_hybrid = 1;
      }
    }
    fclose(cache_f);
  }

  /* Count P-cores and E-cores if hybrid */
  if (g_cpu_info.has_hybrid) {
    int p_count = 0, e_count = 0;
    for (int i = 0; i < g_cpu_info.logical_cpus; i++) {
      char path[128];
      snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpu_capacity", i);
      cache_f = fopen(path, "r");
      if (cache_f) {
        int cap;
        if (fgets(buf, sizeof(buf), cache_f) && sscanf(buf, "%d", &cap) == 1) {
          if (cap >= 1024) p_count++;
          else e_count++;
        }
        fclose(cache_f);
      }
    }
    g_cpu_info.p_cores = p_count / g_cpu_info.threads_per_core;
    g_cpu_info.e_cores = e_count; /* E-cores typically don't have HT */
  }

compute_optimal:
  /* Compute optimal threads for GEMM based on detected topology */
  if (g_cpu_info.has_hybrid && g_cpu_info.p_cores > 0) {
    /* Hybrid: P-cores with HT for best GEMM performance */
    g_cpu_info.optimal_threads = g_cpu_info.p_cores * g_cpu_info.threads_per_core;
  } else {
    /* Standard: all logical CPUs */
    g_cpu_info.optimal_threads = g_cpu_info.logical_cpus;
  }
}
#endif

static void detect_cpu_topology(void) {
  if (g_cpu_detected) return;

#ifdef _WIN32
  detect_cpu_topology_windows();
#else
  detect_cpu_topology_linux();
#endif

  g_cpu_detected = 1;
}

/* Export CPU topology to Zig */
int vt_get_optimal_threads(void) {
  return g_cpu_info.optimal_threads > 0 ? g_cpu_info.optimal_threads : 8;
}

int vt_get_physical_cores(void) {
  return g_cpu_info.physical_cores > 0 ? g_cpu_info.physical_cores : 4;
}

int vt_get_logical_cpus(void) {
  return g_cpu_info.logical_cpus > 0 ? g_cpu_info.logical_cpus : 8;
}

int vt_get_l2_cache_kb(void) {
  return g_cpu_info.l2_cache_kb > 0 ? g_cpu_info.l2_cache_kb : 256;
}

int vt_get_l3_cache_kb(void) {
  return g_cpu_info.l3_cache_kb > 0 ? g_cpu_info.l3_cache_kb : 8192;
}

int vt_is_hybrid_cpu(void) {
  return g_cpu_info.has_hybrid;
}

int vt_has_avx512(void) {
  return g_cpu_info.has_avx512;
}

/* Thread affinity helpers - called from Zig */
#ifdef _WIN32
int vt_set_thread_affinity(void* thread_handle, int core_id) {
  DWORD_PTR mask = 1ULL << core_id;
  return SetThreadAffinityMask((HANDLE)thread_handle, mask) != 0;
}
#else
int vt_set_thread_affinity_self(int core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0;
}
#endif

/* 64-byte aligned allocation for AVX-512 / cache-line alignment */
#define TENSOR_ALIGN 64

#ifdef _WIN32
#include <malloc.h>
static inline void *aligned_tensor_alloc(size_t size) {
  return _aligned_malloc(size, TENSOR_ALIGN);
}
static inline void aligned_tensor_free(void *ptr) { _aligned_free(ptr); }
#else
static inline void *aligned_tensor_alloc(size_t size) {
  /* aligned_alloc requires size to be multiple of alignment */
  size_t aligned_size = (size + TENSOR_ALIGN - 1) & ~(TENSOR_ALIGN - 1);
  return aligned_alloc(TENSOR_ALIGN, aligned_size);
}
static inline void aligned_tensor_free(void *ptr) { free(ptr); }
#endif

/* Zig SIMD functions (pure math, no NIF deps) */
extern double vt_simd_dot(const double *a, const double *b, size_t len);
extern double vt_simd_sum(const double *data, size_t len);
extern void vt_simd_scale(const double *data, double scalar, double *result,
                          size_t len);
extern void vt_simd_add(const double *a, const double *b, double *result,
                        size_t len);
extern void vt_simd_mul(const double *a, const double *b, double *result,
                        size_t len);
extern void vt_simd_matmul(const double *a, const double *b, double *c,
                           size_t m, size_t n, size_t k);
extern void vt_simd_sub(const double *a, const double *b, double *result,
                        size_t len);
extern void vt_simd_negate(const double *data, double *result, size_t len);
extern void vt_simd_relu(const double *data, double *result, size_t len);
extern double vt_simd_max(const double *data, size_t len);
extern double vt_simd_min(const double *data, size_t len);
extern void vt_simd_exp(const double *data, double *result, size_t len);
extern void vt_simd_sigmoid(const double *data, double *result, size_t len);
extern void vt_simd_log(const double *data, double *result, size_t len);
/* In-place mutation ops */
extern void vt_simd_add_mut(double *a, const double *b, size_t len);
extern void vt_simd_scale_mut(double *a, double scalar, size_t len);
extern void vt_simd_negate_mut(double *a, size_t len);
extern void vt_simd_relu_mut(double *a, size_t len);
/* Retro/fused kernels */
extern void vt_saturn_blend(const double *texture, const double *shade,
                            double bias, double *result, size_t len);
extern void vt_fused_linear_relu(const double *a, const double *b,
                                 const double *bias, double *c, size_t m,
                                 size_t n, size_t k);
/* Resonance kernels (Log-Number System) - f64 version */
extern void vt_resonance_mul(const double *a, const double *b, double *result,
                             size_t len);
extern void vt_resonance_power(const double *data, double exponent,
                               double *result, size_t len);

/* LNS (True Log-Number System) - f32 via IADD, 8x throughput */
extern void vt_lns_mul_f32(const float *a, const float *b, float *result,
                           size_t len);
extern void vt_lns_mul_corrected_f32(const float *a, const float *b,
                                     float *result, size_t len);
extern void vt_lns_div_f32(const float *a, const float *b, float *result,
                           size_t len);
extern void vt_lns_sqrt_f32(const float *data, float *result, size_t len);
extern void vt_lns_rsqrt_f32(const float *data, float *result, size_t len);

/* Horde (SoA Physics) */
extern void vt_horde_integrate(double *positions, const double *velocities,
                               double dt, size_t count);
extern void vt_horde_dampen(double *velocities, double friction, size_t count);
extern void vt_horde_accelerate(double *velocities,
                                const double *accelerations, double dt,
                                size_t count);
extern void vt_horde_wrap(double *positions, double max_bound, size_t count);
extern void vt_horde_gravity_2d(double *accelerations, double gravity,
                                size_t entity_count);
extern double vt_horde_kinetic_energy(const double *velocities, size_t count);

/* HDC (Hyperdimensional Computing) */
extern void vt_hdc_bind(const uint64_t *a, const uint64_t *b, uint64_t *result,
                        size_t len);
extern uint64_t vt_hdc_hamming(const uint64_t *a, const uint64_t *b,
                               size_t len);
extern double vt_hdc_similarity(const uint64_t *a, const uint64_t *b,
                                size_t len, size_t dim);
extern void vt_hdc_bundle(const uint64_t *inputs, size_t n_vectors,
                          size_t words, uint64_t *result);
extern void vt_hdc_permute(const uint64_t *input, uint64_t *output,
                           size_t words, size_t shift);
extern void vt_hdc_random(uint64_t *output, size_t words, uint64_t seed);
extern void vt_hdc_weighted_bundle(const uint64_t *inputs, const double *weights,
                                   size_t n_vectors, size_t words,
                                   uint64_t *result);

/* =========================================================================
 * NativeTensor - The core data structure
 * ========================================================================= */

typedef struct {
  double *data;  /* Contiguous row-major array */
  int *shape;    /* Shape array [d0, d1, ...] */
  int *strides;  /* Strides in elements [s0, s1, ...] */
  int ndim;      /* Number of dimensions */
  int size;      /* Total elements */
  int owns_data; /* 1 = free data on destroy, 0 = view */
} NativeTensor;

static ErlNifResourceType *TENSOR_RESOURCE = NULL;

static void tensor_destructor(ErlNifEnv *env, void *obj) {
  (void)env;
  NativeTensor *t = (NativeTensor *)obj;
  if (t->owns_data && t->data)
    aligned_tensor_free(t->data);
  if (t->shape)
    free(t->shape);
  if (t->strides)
    free(t->strides);
}

/** Allocate a new NativeTensor resource with given shape. Data is zeroed. */
static NativeTensor *alloc_tensor(int ndim, const int *shape) {
  NativeTensor *t = (NativeTensor *)enif_alloc_resource(TENSOR_RESOURCE,
                                                        sizeof(NativeTensor));
  if (!t)
    return NULL;

  t->ndim = ndim;
  t->owns_data = 1;

  /* Compute size and strides (row-major) */
  t->size = 1;
  for (int i = 0; i < ndim; i++)
    t->size *= shape[i];

  t->shape = (int *)malloc(ndim * sizeof(int));
  t->strides = (int *)malloc(ndim * sizeof(int));
  if (!t->shape || !t->strides) {
    if (t->shape)
      free(t->shape);
    if (t->strides)
      free(t->strides);
    t->shape = NULL;
    t->strides = NULL;
    t->data = NULL;
    enif_release_resource(t);
    return NULL;
  }
  memcpy(t->shape, shape, ndim * sizeof(int));

  /* Row-major strides */
  int stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    t->strides[i] = stride;
    stride *= shape[i];
  }

  /* Allocate 64-byte aligned zeroed data */
  t->data = (double *)aligned_tensor_alloc(t->size * sizeof(double));
  if (!t->data) {
    free(t->shape);
    free(t->strides);
    t->shape = NULL;
    t->strides = NULL;
    enif_release_resource(t);
    return NULL;
  }
  memset(t->data, 0, t->size * sizeof(double));

  return t;
}

/** Allocate NativeTensor with uninitialized data (use when overwriting all). */
static NativeTensor *alloc_tensor_uninit(int ndim, const int *shape) {
  NativeTensor *t = (NativeTensor *)enif_alloc_resource(TENSOR_RESOURCE,
                                                        sizeof(NativeTensor));
  if (!t)
    return NULL;

  t->ndim = ndim;
  t->owns_data = 1;

  t->size = 1;
  for (int i = 0; i < ndim; i++)
    t->size *= shape[i];

  t->shape = (int *)malloc(ndim * sizeof(int));
  t->strides = (int *)malloc(ndim * sizeof(int));
  if (!t->shape || !t->strides) {
    if (t->shape)
      free(t->shape);
    if (t->strides)
      free(t->strides);
    t->shape = NULL;
    t->strides = NULL;
    t->data = NULL;
    enif_release_resource(t);
    return NULL;
  }
  memcpy(t->shape, shape, ndim * sizeof(int));

  int stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    t->strides[i] = stride;
    stride *= shape[i];
  }

  t->data = (double *)aligned_tensor_alloc(t->size * sizeof(double));
  if (!t->data) {
    free(t->shape);
    free(t->strides);
    t->shape = NULL;
    t->strides = NULL;
    enif_release_resource(t);
    return NULL;
  }

  return t;
}

/** Get NativeTensor from an Erlang resource term */
static NativeTensor *get_tensor(ErlNifEnv *env, ERL_NIF_TERM term) {
  NativeTensor *t;
  if (!enif_get_resource(env, term, TENSOR_RESOURCE, (void **)&t))
    return NULL;
  return t;
}

/** Wrap a NativeTensor as an Erlang term (transfers ownership to GC) */
static ERL_NIF_TERM make_tensor_term(ErlNifEnv *env, NativeTensor *t) {
  ERL_NIF_TERM term = enif_make_resource(env, t);
  enif_release_resource(t); /* GC now owns it */
  return term;
}

/** Parse shape from Erlang list of ints */
static int parse_shape(ErlNifEnv *env, ERL_NIF_TERM list, int *out_shape,
                       int *out_ndim) {
  unsigned len;
  if (!enif_get_list_length(env, list, &len) || len == 0 || len > 8)
    return 0;
  *out_ndim = (int)len;

  ERL_NIF_TERM head, tail = list;
  int i = 0;
  while (enif_get_list_cell(env, tail, &head, &tail)) {
    int val;
    if (!enif_get_int(env, head, &val) || val <= 0)
      return 0;
    out_shape[i++] = val;
  }
  return 1;
}

/* =========================================================================
 * Helpers (legacy list-based API)
 * ========================================================================= */

static double *list_to_doubles(ErlNifEnv *env, ERL_NIF_TERM list,
                               unsigned *out_len) {
  unsigned length;
  if (!enif_get_list_length(env, list, &length))
    return NULL;
  double *arr = (double *)malloc(length * sizeof(double));
  if (!arr)
    return NULL;

  ERL_NIF_TERM head, tail = list;
  unsigned i = 0;
  while (enif_get_list_cell(env, tail, &head, &tail)) {
    double val;
    if (enif_get_double(env, head, &val)) {
      arr[i++] = val;
    } else {
      int ival;
      long lval;
      if (enif_get_int(env, head, &ival))
        arr[i++] = (double)ival;
      else if (enif_get_long(env, head, &lval))
        arr[i++] = (double)lval;
      else {
        free(arr);
        return NULL;
      }
    }
  }
  *out_len = length;
  return arr;
}

static ERL_NIF_TERM doubles_to_list(ErlNifEnv *env, const double *arr,
                                    unsigned len) {
  ERL_NIF_TERM result = enif_make_list(env, 0);
  for (unsigned i = len; i > 0;) {
    i--;
    result = enif_make_list_cell(env, enif_make_double(env, arr[i]), result);
  }
  return result;
}

static ERL_NIF_TERM make_ok(ErlNifEnv *env, ERL_NIF_TERM value) {
  return enif_make_tuple2(env, enif_make_atom(env, "ok"), value);
}

static ERL_NIF_TERM make_ok_nil(ErlNifEnv *env) {
  return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_atom(env, "nil"));
}

static ERL_NIF_TERM make_error(ErlNifEnv *env, const char *reason) {
  return enif_make_tuple2(env, enif_make_atom(env, "error"),
                          enif_make_atom(env, reason));
}

static double get_number(ErlNifEnv *env, ERL_NIF_TERM term, int *ok) {
  double val;
  if (enif_get_double(env, term, &val)) {
    *ok = 1;
    return val;
  }
  int ival;
  if (enif_get_int(env, term, &ival)) {
    *ok = 1;
    return (double)ival;
  }
  long lval;
  if (enif_get_long(env, term, &lval)) {
    *ok = 1;
    return (double)lval;
  }
  *ok = 0;
  return 0.0;
}

/* =========================================================================
 * NIF Resource API — Constructors
 * ========================================================================= */

/** nt_zeros(Shape) -> {ok, Ref} */
static ERL_NIF_TERM nt_zeros(ErlNifEnv *env, int argc,
                             const ERL_NIF_TERM argv[]) {
  (void)argc;
  int shape[8], ndim;
  if (!parse_shape(env, argv[0], shape, &ndim))
    return make_error(env, "invalid_shape");

  NativeTensor *t = alloc_tensor(ndim, shape);
  if (!t)
    return make_error(env, "out_of_memory");
  /* data already zeroed by calloc */

  return make_ok(env, make_tensor_term(env, t));
}

/** nt_ones(Shape) -> {ok, Ref} */
static ERL_NIF_TERM nt_ones(ErlNifEnv *env, int argc,
                            const ERL_NIF_TERM argv[]) {
  (void)argc;
  int shape[8], ndim;
  if (!parse_shape(env, argv[0], shape, &ndim))
    return make_error(env, "invalid_shape");

  NativeTensor *t = alloc_tensor_uninit(ndim, shape);
  if (!t)
    return make_error(env, "out_of_memory");
  for (int i = 0; i < t->size; i++)
    t->data[i] = 1.0;

  return make_ok(env, make_tensor_term(env, t));
}

/** nt_fill(Shape, Value) -> {ok, Ref} */
static ERL_NIF_TERM nt_fill(ErlNifEnv *env, int argc,
                            const ERL_NIF_TERM argv[]) {
  (void)argc;
  int shape[8], ndim;
  if (!parse_shape(env, argv[0], shape, &ndim))
    return make_error(env, "invalid_shape");
  int ok;
  double val = get_number(env, argv[1], &ok);
  if (!ok)
    return make_error(env, "invalid_value");

  NativeTensor *t = alloc_tensor_uninit(ndim, shape);
  if (!t)
    return make_error(env, "out_of_memory");
  for (int i = 0; i < t->size; i++)
    t->data[i] = val;

  return make_ok(env, make_tensor_term(env, t));
}

/** nt_from_list(Data, Shape) -> {ok, Ref} */
static ERL_NIF_TERM nt_from_list(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  (void)argc;
  int shape[8], ndim;
  if (!parse_shape(env, argv[1], shape, &ndim))
    return make_error(env, "invalid_shape");

  unsigned data_len;
  double *data = list_to_doubles(env, argv[0], &data_len);
  if (!data)
    return make_error(env, "invalid_data");

  /* Validate size matches shape */
  int expected_size = 1;
  for (int i = 0; i < ndim; i++)
    expected_size *= shape[i];
  if ((int)data_len != expected_size) {
    free(data);
    return make_error(env, "size_mismatch");
  }

  NativeTensor *t = alloc_tensor_uninit(ndim, shape);
  if (!t) {
    free(data);
    return make_error(env, "out_of_memory");
  }
  memcpy(t->data, data, data_len * sizeof(double));
  free(data);

  return make_ok(env, make_tensor_term(env, t));
}

/* =========================================================================
 * NIF Resource API — Accessors
 * ========================================================================= */

/** nt_to_list(Ref) -> {ok, List} */
static ERL_NIF_TERM nt_to_list(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *t = get_tensor(env, argv[0]);
  if (!t)
    return make_error(env, "invalid_tensor");
  return make_ok(env, doubles_to_list(env, t->data, t->size));
}

/** nt_shape(Ref) -> {ok, ShapeList} */
static ERL_NIF_TERM nt_shape(ErlNifEnv *env, int argc,
                             const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *t = get_tensor(env, argv[0]);
  if (!t)
    return make_error(env, "invalid_tensor");

  ERL_NIF_TERM shape_list = enif_make_list(env, 0);
  for (int i = t->ndim - 1; i >= 0; i--)
    shape_list =
        enif_make_list_cell(env, enif_make_int(env, t->shape[i]), shape_list);
  return make_ok(env, shape_list);
}

/** nt_size(Ref) -> {ok, Int} */
static ERL_NIF_TERM nt_size(ErlNifEnv *env, int argc,
                            const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *t = get_tensor(env, argv[0]);
  if (!t)
    return make_error(env, "invalid_tensor");
  return make_ok(env, enif_make_int(env, t->size));
}

/* =========================================================================
 * NIF Resource API — Element-wise Operations (resource → resource)
 * ========================================================================= */

/** nt_add(RefA, RefB) -> {ok, RefC} */
static ERL_NIF_TERM nt_add(ErlNifEnv *env, int argc,
                           const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");
  if (a->size != b->size)
    return make_error(env, "size_mismatch");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_add(a->data, b->data, c->data, a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_sub(RefA, RefB) -> {ok, RefC} */
static ERL_NIF_TERM nt_sub(ErlNifEnv *env, int argc,
                           const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");
  if (a->size != b->size)
    return make_error(env, "size_mismatch");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_sub(a->data, b->data, c->data, a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_mul(RefA, RefB) -> {ok, RefC} */
static ERL_NIF_TERM nt_mul(ErlNifEnv *env, int argc,
                           const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");
  if (a->size != b->size)
    return make_error(env, "size_mismatch");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_mul(a->data, b->data, c->data, a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_scale(Ref, Scalar) -> {ok, RefC} */
static ERL_NIF_TERM nt_scale(ErlNifEnv *env, int argc,
                             const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");
  int ok;
  double scalar = get_number(env, argv[1], &ok);
  if (!ok)
    return make_error(env, "invalid_scalar");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_scale(a->data, scalar, c->data, a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_negate(Ref) -> {ok, RefC} */
static ERL_NIF_TERM nt_negate(ErlNifEnv *env, int argc,
                              const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_negate(a->data, c->data, a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/* =========================================================================
 * NIF Resource API — Reductions (resource → scalar)
 * ========================================================================= */

/** nt_dot(RefA, RefB) -> {ok, Float} */
static ERL_NIF_TERM nt_dot(ErlNifEnv *env, int argc,
                           const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");
  if (a->size != b->size)
    return make_error(env, "size_mismatch");

  double result = vt_simd_dot(a->data, b->data, a->size);
  return make_ok(env, enif_make_double(env, result));
}

/** nt_sum(Ref) -> {ok, Float} */
static ERL_NIF_TERM nt_sum(ErlNifEnv *env, int argc,
                           const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  double result = vt_simd_sum(a->data, a->size);
  return make_ok(env, enif_make_double(env, result));
}

/** nt_max(Ref) -> {ok, Float} */
static ERL_NIF_TERM nt_max(ErlNifEnv *env, int argc,
                           const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  double mx = vt_simd_max(a->data, a->size);
  return make_ok(env, enif_make_double(env, mx));
}

/** nt_min(Ref) -> {ok, Float} */
static ERL_NIF_TERM nt_min(ErlNifEnv *env, int argc,
                           const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  double mn = vt_simd_min(a->data, a->size);
  return make_ok(env, enif_make_double(env, mn));
}

/* =========================================================================
 * NIF Resource API — Matrix Operations
 * ========================================================================= */

/** nt_matmul(RefA, RefB, M, N, K) -> {ok, RefC} */
static ERL_NIF_TERM nt_matmul(ErlNifEnv *env, int argc,
                              const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");

  int m_int, n_int, k_int;
  if (!enif_get_int(env, argv[2], &m_int) ||
      !enif_get_int(env, argv[3], &n_int) ||
      !enif_get_int(env, argv[4], &k_int))
    return make_error(env, "invalid_dimensions");

  size_t m = (size_t)m_int, n = (size_t)n_int, k = (size_t)k_int;
  if (a->size != (int)(m * k) || b->size != (int)(k * n))
    return make_error(env, "size_mismatch");

  int out_shape[2] = {m_int, n_int};
  NativeTensor *c = alloc_tensor_uninit(2, out_shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_matmul(a->data, b->data, c->data, m, n, k);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_matmul_blas(RefA, RefB, M, N, K) -> {ok, RefC}
 *  Uses Intel MKL (Windows) or OpenBLAS (Unix) for optimized GEMM (600+ GFLOPS)
 *  On Linux: dynamically selects best available backend at runtime
 */
static ERL_NIF_TERM nt_matmul_blas(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");

  int m_int, n_int, k_int;
  if (!enif_get_int(env, argv[2], &m_int) ||
      !enif_get_int(env, argv[3], &n_int) ||
      !enif_get_int(env, argv[4], &k_int))
    return make_error(env, "invalid_dimensions");

  size_t m = (size_t)m_int, n = (size_t)n_int, k = (size_t)k_int;
  if (a->size != (int)(m * k) || b->size != (int)(k * n))
    return make_error(env, "size_mismatch");

  int out_shape[2] = {m_int, n_int};
  NativeTensor *c = alloc_tensor_uninit(2, out_shape);
  if (!c)
    return make_error(env, "out_of_memory");

  /* C = alpha * A @ B + beta * C
   * cblas_dgemm(order, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
   * Row-major: lda=k, ldb=n, ldc=n
   */
#ifdef _WIN32
  /* Windows: directly use MKL */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              (int)m, (int)n, (int)k,
              1.0, a->data, (int)k,
              b->data, (int)n,
              0.0, c->data, (int)n);
#else
  /* Linux: use dynamically loaded backend */
  if (g_dgemm) {
    blas_dgemm((int)m, (int)n, (int)k,
               1.0, a->data, (int)k,
               b->data, (int)n,
               0.0, c->data, (int)n);
  } else {
    /* Fallback to Zig GEMM if no BLAS available */
    vt_simd_matmul(a->data, b->data, c->data, m, n, k);
  }
#endif

  return make_ok(env, make_tensor_term(env, c));
}

/** nt_transpose(Ref) -> {ok, RefC}  (creates contiguous transposed copy) */
static ERL_NIF_TERM nt_transpose(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a || a->ndim != 2)
    return make_error(env, "invalid_tensor");

  int rows = a->shape[0], cols = a->shape[1];
  int out_shape[2] = {cols, rows};
  NativeTensor *c = alloc_tensor_uninit(2, out_shape);
  if (!c)
    return make_error(env, "out_of_memory");

  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      c->data[j * rows + i] = a->data[i * cols + j];

  return make_ok(env, make_tensor_term(env, c));
}

/* =========================================================================
 * NIF Resource API — Activation Functions
 * ========================================================================= */

/** nt_relu(Ref) -> {ok, RefC} */
static ERL_NIF_TERM nt_relu(ErlNifEnv *env, int argc,
                            const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_relu(a->data, c->data, a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_sigmoid(Ref) -> {ok, RefC} */
static ERL_NIF_TERM nt_sigmoid(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_sigmoid(a->data, c->data, (size_t)a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_exp(Ref) -> {ok, RefC} */
static ERL_NIF_TERM nt_exp_nif(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_exp(a->data, c->data, (size_t)a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_log(Ref) -> {ok, RefC} */
static ERL_NIF_TERM nt_log_nif(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_log(a->data, c->data, (size_t)a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/* =========================================================================
 * In-Place Mutation NIFs
 * "Quebrar a imutabilidade dentro do Zig para economizar RAM"
 * ========================================================================= */

/** nt_add_mut(RefA, RefB) -> ok. Modifies A in-place: A += B */
static ERL_NIF_TERM nt_add_mut(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");
  if (a->size != b->size)
    return make_error(env, "size_mismatch");

  vt_simd_add_mut(a->data, b->data, (size_t)a->size);
  return enif_make_atom(env, "ok");
}

/** nt_scale_mut(RefA, Scalar) -> ok. Modifies A in-place: A *= scalar */
static ERL_NIF_TERM nt_scale_mut(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  double scalar;
  if (!enif_get_double(env, argv[1], &scalar))
    return make_error(env, "invalid_scalar");

  vt_simd_scale_mut(a->data, scalar, (size_t)a->size);
  return enif_make_atom(env, "ok");
}

/** nt_negate_mut(RefA) -> ok. Modifies A in-place: A = -A */
static ERL_NIF_TERM nt_negate_mut(ErlNifEnv *env, int argc,
                                  const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  vt_simd_negate_mut(a->data, (size_t)a->size);
  return enif_make_atom(env, "ok");
}

/** nt_relu_mut(RefA) -> ok. Modifies A in-place: A = max(0, A) */
static ERL_NIF_TERM nt_relu_mut(ErlNifEnv *env, int argc,
                                const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  vt_simd_relu_mut(a->data, (size_t)a->size);
  return enif_make_atom(env, "ok");
}

/* =========================================================================
 * Retro / Fused Kernels
 * ========================================================================= */

/** nt_saturn_blend(Texture, Shade, Bias) -> {ok, RefC}
 * VDP1-inspired: result = texture + (shade - bias). Pure addition, no mul. */
static ERL_NIF_TERM nt_saturn_blend(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *texture = get_tensor(env, argv[0]);
  NativeTensor *shade = get_tensor(env, argv[1]);
  if (!texture || !shade)
    return make_error(env, "invalid_tensor");
  if (texture->size != shade->size)
    return make_error(env, "size_mismatch");

  double bias;
  if (!enif_get_double(env, argv[2], &bias))
    return make_error(env, "invalid_bias");

  NativeTensor *c = alloc_tensor_uninit(texture->ndim, texture->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_saturn_blend(texture->data, shade->data, bias, c->data,
                  (size_t)texture->size);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_fused_linear_relu(A, B, Bias, M, N, K) -> {ok, RefC}
 * Fused: C = max(0, A@B + bias). Single pass, saves 2 tensor traversals. */
static ERL_NIF_TERM nt_fused_linear_relu_nif(ErlNifEnv *env, int argc,
                                             const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  NativeTensor *bias = get_tensor(env, argv[2]);
  if (!a || !b || !bias)
    return make_error(env, "invalid_tensor");

  int m, n, k;
  if (!enif_get_int(env, argv[3], &m) || !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k))
    return make_error(env, "invalid_dims");

  if (a->size != m * k || b->size != k * n || bias->size != n)
    return make_error(env, "shape_mismatch");

  int out_shape[2] = {m, n};
  NativeTensor *c = alloc_tensor_uninit(2, out_shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_fused_linear_relu(a->data, b->data, bias->data, c->data, (size_t)m,
                       (size_t)n, (size_t)k);
  return make_ok(env, make_tensor_term(env, c));
}

/* =========================================================================
 * Resonance Kernels (Log-Number System)
 * "Multiplicação como soma no domínio logarítmico"
 * ========================================================================= */

/** nt_resonance_mul(RefA, RefB) -> {ok, RefC}
 * LNS element-wise multiply: result = sign * exp(log|a| + log|b|)
 * Turns multiply into add in log domain. Better precision for chains. */
static ERL_NIF_TERM nt_resonance_mul(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");
  if (a->size != b->size)
    return make_error(env, "size_mismatch");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_resonance_mul(a->data, b->data, c->data, (size_t)a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_resonance_power(Ref, Exponent) -> {ok, RefC}
 * LNS power: result = sign * |x|^exponent via exp(exponent * log|x|)
 * Power = multiply in log domain. Sign preserved for bipolar states. */
static ERL_NIF_TERM nt_resonance_power(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  int ok;
  double exponent = get_number(env, argv[1], &ok);
  if (!ok)
    return make_error(env, "invalid_exponent");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_resonance_power(a->data, exponent, c->data, (size_t)a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/* =========================================================================
 * LNS Tensor (True Log-Number System) - f32 via IADD
 * 8x throughput vs FMA by turning multiply into integer add
 * ========================================================================= */

typedef struct {
  float *data;
  int *shape;
  int ndim;
  int size;
} LnsTensor;

static ErlNifResourceType *LNS_RESOURCE = NULL;

static void lns_destructor(ErlNifEnv *env, void *obj) {
  (void)env;
  LnsTensor *t = (LnsTensor *)obj;
  if (t->data)
    aligned_tensor_free(t->data);
  if (t->shape)
    free(t->shape);
}

static LnsTensor *alloc_lns(int ndim, const int *shape) {
  LnsTensor *t =
      (LnsTensor *)enif_alloc_resource(LNS_RESOURCE, sizeof(LnsTensor));
  if (!t)
    return NULL;
  t->ndim = ndim;
  t->size = 1;
  for (int i = 0; i < ndim; i++)
    t->size *= shape[i];
  t->shape = (int *)malloc(ndim * sizeof(int));
  if (!t->shape) {
    enif_release_resource(t);
    return NULL;
  }
  memcpy(t->shape, shape, ndim * sizeof(int));
  t->data = (float *)aligned_tensor_alloc(t->size * sizeof(float));
  if (!t->data) {
    free(t->shape);
    enif_release_resource(t);
    return NULL;
  }
  return t;
}

static LnsTensor *get_lns(ErlNifEnv *env, ERL_NIF_TERM term) {
  LnsTensor *t;
  if (!enif_get_resource(env, term, LNS_RESOURCE, (void **)&t))
    return NULL;
  return t;
}

/** lns_from_f64(NativeTensorRef) -> {ok, LnsRef}
 * Convert f64 tensor to f32 LNS tensor */
static ERL_NIF_TERM lns_from_f64(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *src = get_tensor(env, argv[0]);
  if (!src)
    return make_error(env, "invalid_tensor");

  LnsTensor *dst = alloc_lns(src->ndim, src->shape);
  if (!dst)
    return make_error(env, "out_of_memory");

  for (int i = 0; i < src->size; i++)
    dst->data[i] = (float)src->data[i];

  ERL_NIF_TERM term = enif_make_resource(env, dst);
  enif_release_resource(dst);
  return make_ok(env, term);
}

/** lns_to_f64(LnsRef) -> {ok, NativeTensorRef}
 * Convert back to f64 */
static ERL_NIF_TERM lns_to_f64(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  (void)argc;
  LnsTensor *src = get_lns(env, argv[0]);
  if (!src)
    return make_error(env, "invalid_lns");

  NativeTensor *dst = alloc_tensor_uninit(src->ndim, src->shape);
  if (!dst)
    return make_error(env, "out_of_memory");

  for (int i = 0; i < src->size; i++)
    dst->data[i] = (double)src->data[i];

  return make_ok(env, make_tensor_term(env, dst));
}

/** lns_mul(LnsA, LnsB) -> {ok, LnsC}
 * Fast LNS multiply via IADD (~11% max error) */
static ERL_NIF_TERM lns_mul(ErlNifEnv *env, int argc,
                            const ERL_NIF_TERM argv[]) {
  (void)argc;
  LnsTensor *a = get_lns(env, argv[0]);
  LnsTensor *b = get_lns(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_lns");
  if (a->size != b->size)
    return make_error(env, "size_mismatch");

  LnsTensor *c = alloc_lns(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_lns_mul_f32(a->data, b->data, c->data, (size_t)a->size);

  ERL_NIF_TERM term = enif_make_resource(env, c);
  enif_release_resource(c);
  return make_ok(env, term);
}

/** lns_mul_corrected(LnsA, LnsB) -> {ok, LnsC}
 * Mitchell's corrected LNS multiply (~2% max error) */
static ERL_NIF_TERM lns_mul_corrected(ErlNifEnv *env, int argc,
                                      const ERL_NIF_TERM argv[]) {
  (void)argc;
  LnsTensor *a = get_lns(env, argv[0]);
  LnsTensor *b = get_lns(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_lns");
  if (a->size != b->size)
    return make_error(env, "size_mismatch");

  LnsTensor *c = alloc_lns(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_lns_mul_corrected_f32(a->data, b->data, c->data, (size_t)a->size);

  ERL_NIF_TERM term = enif_make_resource(env, c);
  enif_release_resource(c);
  return make_ok(env, term);
}

/** lns_div(LnsA, LnsB) -> {ok, LnsC}
 * LNS division via ISUB */
static ERL_NIF_TERM lns_div(ErlNifEnv *env, int argc,
                            const ERL_NIF_TERM argv[]) {
  (void)argc;
  LnsTensor *a = get_lns(env, argv[0]);
  LnsTensor *b = get_lns(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_lns");
  if (a->size != b->size)
    return make_error(env, "size_mismatch");

  LnsTensor *c = alloc_lns(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_lns_div_f32(a->data, b->data, c->data, (size_t)a->size);

  ERL_NIF_TERM term = enif_make_resource(env, c);
  enif_release_resource(c);
  return make_ok(env, term);
}

/** lns_sqrt(Lns) -> {ok, LnsC}
 * LNS sqrt via bit shift */
static ERL_NIF_TERM lns_sqrt(ErlNifEnv *env, int argc,
                             const ERL_NIF_TERM argv[]) {
  (void)argc;
  LnsTensor *a = get_lns(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_lns");

  LnsTensor *c = alloc_lns(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_lns_sqrt_f32(a->data, c->data, (size_t)a->size);

  ERL_NIF_TERM term = enif_make_resource(env, c);
  enif_release_resource(c);
  return make_ok(env, term);
}

/** lns_rsqrt(Lns) -> {ok, LnsC}
 * Fast inverse sqrt (Quake III trick) */
static ERL_NIF_TERM lns_rsqrt(ErlNifEnv *env, int argc,
                              const ERL_NIF_TERM argv[]) {
  (void)argc;
  LnsTensor *a = get_lns(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_lns");

  LnsTensor *c = alloc_lns(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_lns_rsqrt_f32(a->data, c->data, (size_t)a->size);

  ERL_NIF_TERM term = enif_make_resource(env, c);
  enif_release_resource(c);
  return make_ok(env, term);
}

/* =========================================================================
 * Horde - SoA Physics Engine
 * 10K+ entities at 60fps with zero GC pressure
 * ========================================================================= */

typedef struct {
  double *positions;
  double *velocities;
  double *accelerations;
  int entity_count;
  int dims;
} Horde;

static ErlNifResourceType *HORDE_RESOURCE = NULL;

static void horde_destructor(ErlNifEnv *env, void *obj) {
  (void)env;
  Horde *h = (Horde *)obj;
  if (h->positions)
    aligned_tensor_free(h->positions);
  if (h->velocities)
    aligned_tensor_free(h->velocities);
  if (h->accelerations)
    aligned_tensor_free(h->accelerations);
}

/** horde_create(EntityCount, Dims) -> {ok, HordeRef} */
static ERL_NIF_TERM horde_create(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  (void)argc;
  int count, dims;
  if (!enif_get_int(env, argv[0], &count) ||
      !enif_get_int(env, argv[1], &dims) || count <= 0 || dims < 1 || dims > 3)
    return make_error(env, "invalid_params");

  Horde *h = (Horde *)enif_alloc_resource(HORDE_RESOURCE, sizeof(Horde));
  if (!h)
    return make_error(env, "out_of_memory");

  h->entity_count = count;
  h->dims = dims;
  size_t size = count * dims * sizeof(double);

  h->positions = (double *)aligned_tensor_alloc(size);
  h->velocities = (double *)aligned_tensor_alloc(size);
  h->accelerations = NULL; /* Lazy alloc */

  if (!h->positions || !h->velocities) {
    if (h->positions)
      aligned_tensor_free(h->positions);
    if (h->velocities)
      aligned_tensor_free(h->velocities);
    enif_release_resource(h);
    return make_error(env, "out_of_memory");
  }

  memset(h->positions, 0, size);
  memset(h->velocities, 0, size);

  ERL_NIF_TERM term = enif_make_resource(env, h);
  enif_release_resource(h);
  return make_ok(env, term);
}

static Horde *get_horde(ErlNifEnv *env, ERL_NIF_TERM term) {
  Horde *h;
  if (!enif_get_resource(env, term, HORDE_RESOURCE, (void **)&h))
    return NULL;
  return h;
}

/** horde_set_positions(HordeRef, DataList) -> ok */
static ERL_NIF_TERM horde_set_positions(ErlNifEnv *env, int argc,
                                        const ERL_NIF_TERM argv[]) {
  (void)argc;
  Horde *h = get_horde(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_horde");

  unsigned len;
  double *data = list_to_doubles(env, argv[1], &len);
  if (!data)
    return make_error(env, "invalid_data");

  size_t expected = h->entity_count * h->dims;
  if (len != expected) {
    free(data);
    return make_error(env, "size_mismatch");
  }

  memcpy(h->positions, data, len * sizeof(double));
  free(data);
  return make_ok_nil(env);
}

/** horde_set_velocities(HordeRef, DataList) -> ok */
static ERL_NIF_TERM horde_set_velocities(ErlNifEnv *env, int argc,
                                         const ERL_NIF_TERM argv[]) {
  (void)argc;
  Horde *h = get_horde(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_horde");

  unsigned len;
  double *data = list_to_doubles(env, argv[1], &len);
  if (!data)
    return make_error(env, "invalid_data");

  size_t expected = h->entity_count * h->dims;
  if (len != expected) {
    free(data);
    return make_error(env, "size_mismatch");
  }

  memcpy(h->velocities, data, len * sizeof(double));
  free(data);
  return make_ok_nil(env);
}

/** horde_integrate(HordeRef, Dt) -> ok
 * Euler step: pos += vel * dt (FMA) */
static ERL_NIF_TERM horde_integrate_nif(ErlNifEnv *env, int argc,
                                        const ERL_NIF_TERM argv[]) {
  (void)argc;
  Horde *h = get_horde(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_horde");

  double dt;
  if (!enif_get_double(env, argv[1], &dt))
    return make_error(env, "invalid_dt");

  vt_horde_integrate(h->positions, h->velocities, dt,
                     h->entity_count * h->dims);
  return make_ok_nil(env);
}

/** horde_dampen(HordeRef, Friction) -> ok */
static ERL_NIF_TERM horde_dampen_nif(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
  (void)argc;
  Horde *h = get_horde(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_horde");

  double friction;
  if (!enif_get_double(env, argv[1], &friction))
    return make_error(env, "invalid_friction");

  vt_horde_dampen(h->velocities, friction, h->entity_count * h->dims);
  return make_ok_nil(env);
}

/** horde_wrap(HordeRef, MaxBound) -> ok
 * Toroidal boundary conditions */
static ERL_NIF_TERM horde_wrap_nif(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  (void)argc;
  Horde *h = get_horde(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_horde");

  double max_bound;
  if (!enif_get_double(env, argv[1], &max_bound))
    return make_error(env, "invalid_bound");

  vt_horde_wrap(h->positions, max_bound, h->entity_count * h->dims);
  return make_ok_nil(env);
}

/** horde_get_positions(HordeRef) -> {ok, List} */
static ERL_NIF_TERM horde_get_positions(ErlNifEnv *env, int argc,
                                        const ERL_NIF_TERM argv[]) {
  (void)argc;
  Horde *h = get_horde(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_horde");

  return make_ok(env,
                 doubles_to_list(env, h->positions, h->entity_count * h->dims));
}

/** horde_get_velocities(HordeRef) -> {ok, List} */
static ERL_NIF_TERM horde_get_velocities(ErlNifEnv *env, int argc,
                                         const ERL_NIF_TERM argv[]) {
  (void)argc;
  Horde *h = get_horde(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_horde");

  return make_ok(
      env, doubles_to_list(env, h->velocities, h->entity_count * h->dims));
}

/** horde_count(HordeRef) -> {ok, Int} */
static ERL_NIF_TERM horde_count_nif(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
  (void)argc;
  Horde *h = get_horde(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_horde");

  return make_ok(env, enif_make_int(env, h->entity_count));
}

/** horde_kinetic_energy(HordeRef) -> {ok, Float} */
static ERL_NIF_TERM horde_kinetic_energy_nif(ErlNifEnv *env, int argc,
                                             const ERL_NIF_TERM argv[]) {
  (void)argc;
  Horde *h = get_horde(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_horde");

  double ke = vt_horde_kinetic_energy(h->velocities, h->entity_count * h->dims);
  return make_ok(env, enif_make_double(env, ke));
}

/* =========================================================================
 * HDC - Hyperdimensional Computing
 * One-shot learning via binary vectors and popcount similarity
 * ========================================================================= */

typedef struct {
  uint64_t *data;
  int words; /* Number of u64 words */
  int dim;   /* Total bits = words * 64 */
} HdcVector;

static ErlNifResourceType *HDC_RESOURCE = NULL;

static void hdc_destructor(ErlNifEnv *env, void *obj) {
  (void)env;
  HdcVector *h = (HdcVector *)obj;
  if (h->data)
    aligned_tensor_free(h->data);
}

static HdcVector *get_hdc(ErlNifEnv *env, ERL_NIF_TERM term) {
  HdcVector *h;
  if (!enif_get_resource(env, term, HDC_RESOURCE, (void **)&h))
    return NULL;
  return h;
}

/** hdc_create(Dim) -> {ok, HdcRef}
 * Dim must be multiple of 64 */
static ERL_NIF_TERM hdc_create_nif(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  (void)argc;
  int dim;
  if (!enif_get_int(env, argv[0], &dim) || dim <= 0 || dim % 64 != 0)
    return make_error(env, "dim_must_be_multiple_of_64");

  HdcVector *h =
      (HdcVector *)enif_alloc_resource(HDC_RESOURCE, sizeof(HdcVector));
  if (!h)
    return make_error(env, "out_of_memory");

  h->dim = dim;
  h->words = dim / 64;
  h->data = (uint64_t *)aligned_tensor_alloc(h->words * sizeof(uint64_t));
  if (!h->data) {
    enif_release_resource(h);
    return make_error(env, "out_of_memory");
  }
  memset(h->data, 0, h->words * sizeof(uint64_t));

  ERL_NIF_TERM term = enif_make_resource(env, h);
  enif_release_resource(h);
  return make_ok(env, term);
}

/** hdc_random(Dim, Seed) -> {ok, HdcRef}
 * Create random hypervector */
static ERL_NIF_TERM hdc_random_nif(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  (void)argc;
  int dim;
  unsigned long seed;
  if (!enif_get_int(env, argv[0], &dim) || dim <= 0 || dim % 64 != 0)
    return make_error(env, "dim_must_be_multiple_of_64");
  if (!enif_get_ulong(env, argv[1], &seed))
    return make_error(env, "invalid_seed");

  HdcVector *h =
      (HdcVector *)enif_alloc_resource(HDC_RESOURCE, sizeof(HdcVector));
  if (!h)
    return make_error(env, "out_of_memory");

  h->dim = dim;
  h->words = dim / 64;
  h->data = (uint64_t *)aligned_tensor_alloc(h->words * sizeof(uint64_t));
  if (!h->data) {
    enif_release_resource(h);
    return make_error(env, "out_of_memory");
  }

  vt_hdc_random(h->data, h->words, seed);

  ERL_NIF_TERM term = enif_make_resource(env, h);
  enif_release_resource(h);
  return make_ok(env, term);
}

/** hdc_bind(HdcA, HdcB) -> {ok, HdcC}
 * XOR binding: associates two concepts */
static ERL_NIF_TERM hdc_bind_nif(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  (void)argc;
  HdcVector *a = get_hdc(env, argv[0]);
  HdcVector *b = get_hdc(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_hdc");
  if (a->dim != b->dim)
    return make_error(env, "dim_mismatch");

  HdcVector *c =
      (HdcVector *)enif_alloc_resource(HDC_RESOURCE, sizeof(HdcVector));
  if (!c)
    return make_error(env, "out_of_memory");

  c->dim = a->dim;
  c->words = a->words;
  c->data = (uint64_t *)aligned_tensor_alloc(c->words * sizeof(uint64_t));
  if (!c->data) {
    enif_release_resource(c);
    return make_error(env, "out_of_memory");
  }

  vt_hdc_bind(a->data, b->data, c->data, a->words);

  ERL_NIF_TERM term = enif_make_resource(env, c);
  enif_release_resource(c);
  return make_ok(env, term);
}

/** hdc_similarity(HdcA, HdcB) -> {ok, Float}
 * Cosine-like similarity via Hamming distance */
static ERL_NIF_TERM hdc_similarity_nif(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  (void)argc;
  HdcVector *a = get_hdc(env, argv[0]);
  HdcVector *b = get_hdc(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_hdc");
  if (a->dim != b->dim)
    return make_error(env, "dim_mismatch");

  double sim = vt_hdc_similarity(a->data, b->data, a->words, a->dim);
  return make_ok(env, enif_make_double(env, sim));
}

/** hdc_permute(Hdc, Shift) -> {ok, HdcC}
 * Circular bit permutation for sequence encoding */
static ERL_NIF_TERM hdc_permute_nif(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
  (void)argc;
  HdcVector *a = get_hdc(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_hdc");

  int shift;
  if (!enif_get_int(env, argv[1], &shift))
    return make_error(env, "invalid_shift");

  HdcVector *c =
      (HdcVector *)enif_alloc_resource(HDC_RESOURCE, sizeof(HdcVector));
  if (!c)
    return make_error(env, "out_of_memory");

  c->dim = a->dim;
  c->words = a->words;
  c->data = (uint64_t *)aligned_tensor_alloc(c->words * sizeof(uint64_t));
  if (!c->data) {
    enif_release_resource(c);
    return make_error(env, "out_of_memory");
  }

  vt_hdc_permute(a->data, c->data, a->words, (size_t)shift);

  ERL_NIF_TERM term = enif_make_resource(env, c);
  enif_release_resource(c);
  return make_ok(env, term);
}

/** hdc_dim(Hdc) -> {ok, Int} */
static ERL_NIF_TERM hdc_dim_nif(ErlNifEnv *env, int argc,
                                const ERL_NIF_TERM argv[]) {
  (void)argc;
  HdcVector *h = get_hdc(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_hdc");

  return make_ok(env, enif_make_int(env, h->dim));
}

/* =========================================================================
 * Legacy list-based NIFs (backward compatibility)
 * ========================================================================= */

static ERL_NIF_TERM nif_simd_dot(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  if (argc != 2)
    return enif_make_badarg(env);
  unsigned la, lb;
  double *a = list_to_doubles(env, argv[0], &la);
  if (!a)
    return make_error(env, "invalid_input");
  double *b = list_to_doubles(env, argv[1], &lb);
  if (!b) {
    free(a);
    return make_error(env, "invalid_input");
  }
  if (la != lb) {
    free(a);
    free(b);
    return make_error(env, "length_mismatch");
  }
  double r = vt_simd_dot(a, b, la);
  free(a);
  free(b);
  return make_ok(env, enif_make_double(env, r));
}

static ERL_NIF_TERM nif_simd_sum(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  if (argc != 1)
    return enif_make_badarg(env);
  unsigned len;
  double *d = list_to_doubles(env, argv[0], &len);
  if (!d)
    return make_error(env, "invalid_input");
  double r = vt_simd_sum(d, len);
  free(d);
  return make_ok(env, enif_make_double(env, r));
}

static ERL_NIF_TERM nif_simd_scale(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  if (argc != 2)
    return enif_make_badarg(env);
  unsigned len;
  double *d = list_to_doubles(env, argv[0], &len);
  if (!d)
    return make_error(env, "invalid_input");
  int ok;
  double s = get_number(env, argv[1], &ok);
  if (!ok) {
    free(d);
    return make_error(env, "invalid_scalar");
  }
  double *r = (double *)malloc(len * sizeof(double));
  if (!r) {
    free(d);
    return make_error(env, "out_of_memory");
  }
  vt_simd_scale(d, s, r, len);
  ERL_NIF_TERM rl = doubles_to_list(env, r, len);
  free(d);
  free(r);
  return make_ok(env, rl);
}

static ERL_NIF_TERM nif_simd_add(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  if (argc != 2)
    return enif_make_badarg(env);
  unsigned la, lb;
  double *a = list_to_doubles(env, argv[0], &la);
  if (!a)
    return make_error(env, "invalid_input");
  double *b = list_to_doubles(env, argv[1], &lb);
  if (!b) {
    free(a);
    return make_error(env, "invalid_input");
  }
  if (la != lb) {
    free(a);
    free(b);
    return make_error(env, "length_mismatch");
  }
  double *r = (double *)malloc(la * sizeof(double));
  if (!r) {
    free(a);
    free(b);
    return make_error(env, "out_of_memory");
  }
  vt_simd_add(a, b, r, la);
  ERL_NIF_TERM rl = doubles_to_list(env, r, la);
  free(a);
  free(b);
  free(r);
  return make_ok(env, rl);
}

static ERL_NIF_TERM nif_simd_mul(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  if (argc != 2)
    return enif_make_badarg(env);
  unsigned la, lb;
  double *a = list_to_doubles(env, argv[0], &la);
  if (!a)
    return make_error(env, "invalid_input");
  double *b = list_to_doubles(env, argv[1], &lb);
  if (!b) {
    free(a);
    return make_error(env, "invalid_input");
  }
  if (la != lb) {
    free(a);
    free(b);
    return make_error(env, "length_mismatch");
  }
  double *r = (double *)malloc(la * sizeof(double));
  if (!r) {
    free(a);
    free(b);
    return make_error(env, "out_of_memory");
  }
  vt_simd_mul(a, b, r, la);
  ERL_NIF_TERM rl = doubles_to_list(env, r, la);
  free(a);
  free(b);
  free(r);
  return make_ok(env, rl);
}

static ERL_NIF_TERM nif_simd_matmul(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
  if (argc != 5)
    return enif_make_badarg(env);
  int mi, ni, ki;
  if (!enif_get_int(env, argv[2], &mi) || !enif_get_int(env, argv[3], &ni) ||
      !enif_get_int(env, argv[4], &ki))
    return make_error(env, "invalid_dimensions");
  unsigned la, lb;
  double *a = list_to_doubles(env, argv[0], &la);
  if (!a)
    return make_error(env, "invalid_input");
  double *b = list_to_doubles(env, argv[1], &lb);
  if (!b) {
    free(a);
    return make_error(env, "invalid_input");
  }
  if ((int)la != mi * ki || (int)lb != ki * ni) {
    free(a);
    free(b);
    return make_error(env, "size_mismatch");
  }
  double *c = (double *)malloc(mi * ni * sizeof(double));
  if (!c) {
    free(a);
    free(b);
    return make_error(env, "out_of_memory");
  }
  vt_simd_matmul(a, b, c, mi, ni, ki);
  ERL_NIF_TERM rl = doubles_to_list(env, c, mi * ni);
  free(a);
  free(b);
  free(c);
  return make_ok(env, rl);
}

static ERL_NIF_TERM nif_simd_available(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  (void)argc;
  (void)argv;
  return enif_make_atom(env, "true");
}

static ERL_NIF_TERM nif_backend_info(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
  (void)argc;
  (void)argv;
  char info[512];
#ifdef _WIN32
  const char *blas_name = "Intel MKL";
#else
  const char *blas_name = g_blas_name ? g_blas_name : "Zig GEMM";
#endif
  snprintf(info, sizeof(info),
           "%s + Zig SIMD (Vec8 f64) | %d cores (%d logical) | L2:%dKB L3:%dKB | %s%s| %d threads",
           blas_name,
           g_cpu_info.physical_cores,
           g_cpu_info.logical_cpus,
           g_cpu_info.l2_cache_kb,
           g_cpu_info.l3_cache_kb,
           g_cpu_info.has_avx512 ? "AVX-512 " : (g_cpu_info.has_avx2 ? "AVX2 " : ""),
           g_cpu_info.has_hybrid ? "Hybrid " : "",
           g_cpu_info.optimal_threads);
  ErlNifBinary bin;
  if (!enif_alloc_binary(strlen(info), &bin))
    return enif_make_atom(env, "error");
  memcpy(bin.data, info, strlen(info));
  return enif_make_binary(env, &bin);
}

/** cpu_topology() -> {ok, Map}
 * Returns detected CPU topology as Erlang map */
static ERL_NIF_TERM nif_cpu_topology(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
  (void)argc;
  (void)argv;

  ERL_NIF_TERM keys[12], vals[12];
  int i = 0;

  keys[i] = enif_make_atom(env, "logical_cpus");
  vals[i++] = enif_make_int(env, g_cpu_info.logical_cpus);

  keys[i] = enif_make_atom(env, "physical_cores");
  vals[i++] = enif_make_int(env, g_cpu_info.physical_cores);

  keys[i] = enif_make_atom(env, "sockets");
  vals[i++] = enif_make_int(env, g_cpu_info.sockets);

  keys[i] = enif_make_atom(env, "threads_per_core");
  vals[i++] = enif_make_int(env, g_cpu_info.threads_per_core);

  keys[i] = enif_make_atom(env, "l1_cache_kb");
  vals[i++] = enif_make_int(env, g_cpu_info.l1_cache_kb);

  keys[i] = enif_make_atom(env, "l2_cache_kb");
  vals[i++] = enif_make_int(env, g_cpu_info.l2_cache_kb);

  keys[i] = enif_make_atom(env, "l3_cache_kb");
  vals[i++] = enif_make_int(env, g_cpu_info.l3_cache_kb);

  keys[i] = enif_make_atom(env, "has_avx2");
  vals[i++] = enif_make_atom(env, g_cpu_info.has_avx2 ? "true" : "false");

  keys[i] = enif_make_atom(env, "has_avx512");
  vals[i++] = enif_make_atom(env, g_cpu_info.has_avx512 ? "true" : "false");

  keys[i] = enif_make_atom(env, "has_hybrid");
  vals[i++] = enif_make_atom(env, g_cpu_info.has_hybrid ? "true" : "false");

  keys[i] = enif_make_atom(env, "optimal_threads");
  vals[i++] = enif_make_int(env, g_cpu_info.optimal_threads);

  if (g_cpu_info.has_hybrid) {
    keys[i] = enif_make_atom(env, "p_cores");
    vals[i++] = enif_make_int(env, g_cpu_info.p_cores);
  }

  ERL_NIF_TERM map;
  enif_make_map_from_arrays(env, keys, vals, i, &map);
  return make_ok(env, map);
}

/* =========================================================================
 * NIF Init
 * ========================================================================= */

static int nif_load(ErlNifEnv *env, void **priv, ERL_NIF_TERM info) {
  (void)priv;
  (void)info;

  /* Detect CPU topology once at NIF load (MKL-style runtime init) */
  detect_cpu_topology();

#ifndef _WIN32
  /* Linux: detect best BLAS backend (MKL > OpenBLAS-tuned > OpenBLAS > Zig) */
  detect_blas_backend();

  /* Auto-tune thread count based on matrix size heuristics */
  if (g_set_threads && g_cpu_info.optimal_threads > 0) {
    blas_set_threads(g_cpu_info.optimal_threads);
  }
#endif

  TENSOR_RESOURCE = enif_open_resource_type(
      env, NULL, "NativeTensor", tensor_destructor, ERL_NIF_RT_CREATE, NULL);
  if (!TENSOR_RESOURCE)
    return -1;

  LNS_RESOURCE = enif_open_resource_type(env, NULL, "LnsTensor", lns_destructor,
                                         ERL_NIF_RT_CREATE, NULL);
  if (!LNS_RESOURCE)
    return -1;

  HORDE_RESOURCE = enif_open_resource_type(env, NULL, "Horde", horde_destructor,
                                           ERL_NIF_RT_CREATE, NULL);
  if (!HORDE_RESOURCE)
    return -1;

  HDC_RESOURCE = enif_open_resource_type(env, NULL, "HdcVector", hdc_destructor,
                                         ERL_NIF_RT_CREATE, NULL);
  if (!HDC_RESOURCE)
    return -1;

  return 0;
}

static ErlNifFunc nif_funcs[] = {
    /* Legacy list-based API */
    {"nif_simd_dot", 2, nif_simd_dot, 0},
    {"nif_simd_sum", 1, nif_simd_sum, 0},
    {"nif_simd_scale", 2, nif_simd_scale, 0},
    {"nif_simd_add", 2, nif_simd_add, 0},
    {"nif_simd_mul", 2, nif_simd_mul, 0},
    {"nif_simd_matmul", 5, nif_simd_matmul, 0},
    {"simd_available", 0, nif_simd_available, 0},
    {"backend_info", 0, nif_backend_info, 0},
    {"cpu_topology", 0, nif_cpu_topology, 0},

    /* NIF Resource API — constructors */
    {"nt_zeros", 1, nt_zeros, 0},
    {"nt_ones", 1, nt_ones, 0},
    {"nt_fill", 2, nt_fill, 0},
    {"nt_from_list", 2, nt_from_list, 0},

    /* NIF Resource API — accessors */
    {"nt_to_list", 1, nt_to_list, 0},
    {"nt_shape", 1, nt_shape, 0},
    {"nt_size", 1, nt_size, 0},

    /* NIF Resource API — element-wise ops */
    {"nt_add", 2, nt_add, 0},
    {"nt_sub", 2, nt_sub, 0},
    {"nt_mul", 2, nt_mul, 0},
    {"nt_scale", 2, nt_scale, 0},
    {"nt_negate", 1, nt_negate, 0},

    /* NIF Resource API — reductions */
    {"nt_dot", 2, nt_dot, 0},
    {"nt_sum", 1, nt_sum, 0},
    {"nt_max", 1, nt_max, 0},
    {"nt_min", 1, nt_min, 0},

    /* NIF Resource API — matrix ops */
    {"nt_matmul", 5, nt_matmul, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_matmul_blas", 5, nt_matmul_blas, ERL_NIF_DIRTY_JOB_CPU_BOUND},  /* MKL/OpenBLAS */
    {"nt_transpose", 1, nt_transpose, ERL_NIF_DIRTY_JOB_CPU_BOUND},

    /* NIF Resource API — activations (dirty: SIMD polynomial approx on large
       tensors) */
    {"nt_relu", 1, nt_relu, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_sigmoid", 1, nt_sigmoid, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_exp", 1, nt_exp_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_log", 1, nt_log_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},

    /* In-place mutation (dirty: modifies large tensors) */
    {"nt_add_mut", 2, nt_add_mut, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_scale_mut", 2, nt_scale_mut, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_negate_mut", 1, nt_negate_mut, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_relu_mut", 1, nt_relu_mut, ERL_NIF_DIRTY_JOB_CPU_BOUND},

    /* Retro / fused kernels (dirty) */
    {"nt_saturn_blend", 3, nt_saturn_blend, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_fused_linear_relu", 6, nt_fused_linear_relu_nif,
     ERL_NIF_DIRTY_JOB_CPU_BOUND},

    /* Resonance kernels — LNS f64 (dirty) */
    {"nt_resonance_mul", 2, nt_resonance_mul, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_resonance_power", 2, nt_resonance_power, ERL_NIF_DIRTY_JOB_CPU_BOUND},

    /* LNS (True Log-Number System) — f32 via IADD, 8x throughput */
    {"lns_from_f64", 1, lns_from_f64, 0},
    {"lns_to_f64", 1, lns_to_f64, 0},
    {"lns_mul", 2, lns_mul, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"lns_mul_corrected", 2, lns_mul_corrected, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"lns_div", 2, lns_div, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"lns_sqrt", 1, lns_sqrt, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"lns_rsqrt", 1, lns_rsqrt, ERL_NIF_DIRTY_JOB_CPU_BOUND},

    /* Horde — SoA Physics, 10K+ entities at 60fps */
    {"horde_create", 2, horde_create, 0},
    {"horde_set_positions", 2, horde_set_positions, 0},
    {"horde_set_velocities", 2, horde_set_velocities, 0},
    {"horde_integrate", 2, horde_integrate_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"horde_dampen", 2, horde_dampen_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"horde_wrap", 2, horde_wrap_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"horde_get_positions", 1, horde_get_positions, 0},
    {"horde_get_velocities", 1, horde_get_velocities, 0},
    {"horde_count", 1, horde_count_nif, 0},
    {"horde_kinetic_energy", 1, horde_kinetic_energy_nif, 0},

    /* HDC — Hyperdimensional Computing, one-shot learning */
    {"hdc_create", 1, hdc_create_nif, 0},
    {"hdc_random", 2, hdc_random_nif, 0},
    {"hdc_bind", 2, hdc_bind_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"hdc_similarity", 2, hdc_similarity_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"hdc_permute", 2, hdc_permute_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"hdc_dim", 1, hdc_dim_nif, 0},
};

ERL_NIF_INIT(viva_tensor_zig, nif_funcs, nif_load, NULL, NULL, NULL)
