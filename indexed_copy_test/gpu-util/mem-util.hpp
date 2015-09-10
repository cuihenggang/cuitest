#ifndef GPU_UTIL_MEM_UTIL_HPP_
#define GPU_UTIL_MEM_UTIL_HPP_

#include "syncedmem.hpp"

inline void myfree(void *ptr, int flag) {
  if (flag) {
    CUDA_CHECK(cudaFree(ptr));
  } else {
    free(ptr);
  }
}

inline void mymalloc(void **ptr, size_t size, int flag) {
  if (flag) {
    CUDA_CHECK(cudaMalloc(ptr, size));
  } else {
    *ptr = malloc(size);
  }
}

template <typename T>
inline void mytypedmalloc(T **ptr, size_t count, int flag) {
  size_t size = count * sizeof(T);
  mymalloc(reinterpret_cast<void **>(ptr), size, flag);
}

inline void mymemcpy(void *dst, const void *src, size_t size, int flag) {
  if (flag) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
  } else {
    memcpy(dst, src, size);
  }
}

template <typename T>
inline void mytypedmemcpy(T *dst, const T *src, size_t count, int flag) {
  size_t size = count * sizeof(T);
  mymemcpy(reinterpret_cast<void *>(dst),
      reinterpret_cast<const void *>(src), size, flag);
}

inline void mymemzero(void *ptr, size_t size, int flag) {
  if (flag) {
    CUDA_CHECK(cudaMemset(ptr, 0, size));
  } else {
    memset(ptr, 0, size);
  }
}

#endif  // GPU_UTIL_MEM_UTIL_HPP_
