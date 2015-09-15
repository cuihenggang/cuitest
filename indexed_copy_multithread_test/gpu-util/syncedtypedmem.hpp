#ifndef GPU_UTIL_SYNCEDTYPEDMEM_HPP_
#define GPU_UTIL_SYNCEDTYPEDMEM_HPP_

#include "syncedmem.hpp"

using caffe::SyncedMemory;

/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
 template <typename T>
class SyncedTypedMemory : SyncedMemory {
 public:
  explicit SyncedTypedMemory(size_t size)
      : SyncedMemory(size * sizeof(T)) {}
  const T *cpu_data() {
    return reinterpret_cast<const T *>(SyncedMemory::cpu_data());
  }
  const T *gpu_data() {
    return reinterpret_cast<const T *>(SyncedMemory::gpu_data());
  }
  T *mutable_cpu_data() {
    return reinterpret_cast<T *>(SyncedMemory::mutable_cpu_data());
  }
  T *mutable_gpu_data() {
    return reinterpret_cast<T *>(SyncedMemory::mutable_gpu_data());
  }
  const T *data(int flag) {
    if (flag) {
      return gpu_data();
    } else {
      return cpu_data();
    }
  }
  T *mutable_data(int flag) {
    if (flag) {
      return mutable_gpu_data();
    } else {
      return mutable_cpu_data();
    }
  }
  DISABLE_COPY_AND_ASSIGN(SyncedTypedMemory);
};  // class SyncedTypedMemory

#endif  // GPU_UTIL_SYNCEDTYPEDMEM_HPP_
