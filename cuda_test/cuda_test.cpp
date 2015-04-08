#include <iostream>
#include <string>
#include <vector>

#include "math_functions.hpp"

using namespace std;

typedef unsigned int uint;

int main(int argc, char* argv[])
{
  uint data_size = 1000;
  vector<float> a(data_size);
  vector<float> b(data_size);
  caffe::caffe_gpu_dot<float>(data_size, a.data(), a.data(), b.data());
}
