#include <iostream>
#include <string>
#include <vector>

#include "syncedmem.hpp"
#include "math_functions.hpp"

using namespace std;
using namespace caffe;

typedef unsigned int uint;

int main(int argc, char* argv[])
{
  cout << "start\n";
  Caffe::SetDevice(0);
  Caffe::DeviceQuery();
  uint data_size = 1000;
  SyncedMemory a(data_size * sizeof(float));
  SyncedMemory b(data_size * sizeof(float));
  SyncedMemory c(data_size * sizeof(float));
  float d;
  float *a_cptr = static_cast<float *>(a.mutable_cpu_data());
  float *b_cptr = static_cast<float *>(b.mutable_cpu_data());
  float *c_cptr = static_cast<float *>(c.mutable_cpu_data());
  cout << "allocated cpu memory\n";
  cout << "a_cptr = " << a_cptr << endl;
  for (uint i = 0; i < data_size; i++) {
    a_cptr[i] = 2;
    b_cptr[i] = 3;
    c_cptr[i] = 0;
  }
  d = 0;
  cout << "initialized value\n";
  float *a_gptr = static_cast<float *>(a.mutable_gpu_data());
  float *b_gptr = static_cast<float *>(b.mutable_gpu_data());
  float *c_gptr = static_cast<float *>(c.mutable_gpu_data());
  cout << "switched to gpu memory\n";
  cout << "a_gptr = " << a_gptr << endl;
  caffe_gpu_dot(data_size, a_gptr, b_gptr, &d);
  // caffe_gpu_asum(data_size, a_gptr, &d);
  caffe_gpu_add(data_size, a_gptr, b_gptr, c_gptr);
  // caffe_gpu_exp(data_size, a_gptr, b_gptr);
  cout << "calculated\n";
  c_cptr = static_cast<float *>(c.mutable_cpu_data());
  cout << "copied back to cpu\n";
  cout << "c_cptr[99] = " << c_cptr[99] << endl;
  cout << "d = " << d << endl;
}
