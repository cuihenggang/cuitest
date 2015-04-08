#ifndef __PORTABLE_BYTES_HPP__
#define __PORTABLE_BYTES_HPP__

/*
 * Copyright(C) 2013 by Carnegie Mellon University.
 *
 */

#include <stddef.h>
#include <stdio.h>

#include <vector>
#include <string>

#include <iostream>
using std::cerr;
using std::endl;

#include <zmq.hpp>

typedef void(free_func_t)(void *data, void *hint);

class PortableBytes {
 public:
  virtual int init() = 0;
  virtual int init_size(size_t size_) = 0;
  virtual int init_data(void *data_, size_t size_,
                        free_func_t *ffn_, void *hint_) = 0;
  virtual void *data() = 0;
  virtual size_t size() = 0;

  template<class T>
  void pack(const T& t) {
    size_t data_size = sizeof(T);
    init_size(data_size);
    *((T *)data()) = t;
  }

  template<class T>
  void unpack(T& t) {
    assert(size() >= sizeof(T));
    t = *((T *)data());
  }

  template<class T>
  void pack_vector(const std::vector<T>& vec) {
    size_t data_size = vec.size() * sizeof(T);
    init_size(data_size);
    memcpy(data(), vec.data(), data_size);
  }

  template<class T>
  void unpack_vector(std::vector<T>& vec) {
    size_t vec_size = size() / sizeof(T);
    vec.resize(vec_size);
    memcpy(vec.data(), data(), size());
  }

  void pack_string(const std::string& str) {
    init_size(str.size());
    memcpy(data(), str.data(), str.size());
  }

  void unpack_string(std::string& str) {
    str.assign(reinterpret_cast<char *>(data()), size());
  }
};

#endif  // __PORTABLE_BYTES_HPP__
