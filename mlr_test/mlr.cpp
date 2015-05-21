/*
 * mlr.hpp
 *
 *  Created on: Feb 5, 2015
 *      Author: aaron
 */

#include <stdint.h>
#include <boost/unordered_set.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/format.hpp>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <limits>
#include <fstream>

#include <tbb/tick_count.h>

// #include <glog/logging.h>

#include "syncedmem.hpp"
#include "math_functions.hpp"

#include "lazytable-types.hpp"
#include "mlr-util.hpp"
#include "metafile-reader.hpp"

using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::istringstream;

typedef uint32_t uint;

using caffe::SyncedMemory;

#if !defined(CPU_ONLY)
void SoftmaxAndAdjust_gpu(float *vec, size_t size, uint label);
#endif

void SoftmaxAndAdjust(float *vec, size_t size, uint label) {
  Softmax(vec, size);
  vec[label] -= 1.; // See Bishop PRML (2006) Eq. (4.109)
}

#if !defined(CPU_ONLY)
void empty_gpu_func();
#endif

class mlr_computer {
 public:
  uint num_compobj_;
  uint compobj_rank_;
  uint staleness_;
  int accuracy_;
  uint unit_of_work_per_clock_;
                             /* How work_per_clock_ is specified */
  uint work_per_clock_;  /* Work per clock */

  uint num_train_data_;
  uint feature_dim_;
  uint num_labels_;
  string read_format_;
  bool feature_one_based_;
  bool label_one_based_;
  bool snappy_compressed_;
  float learning_rate_;
  float decay_rate_;

  uint cur_work_;
  uint cur_clock_;
  uint batch_offset_;
  uint batch_size_;

  string inputfile_prefix_;

  uint cpu_worker_;
  vector<SyncedMemory *> train_feature_mems_;
  vector<int> train_labels_;
  SyncedMemory *w_cache_mem_;
  SyncedMemory *w_delta_mem_;
  SyncedMemory *y_mem_;

  double refresh_weights_time;
  double alloc_mem_time;
  double dotproduct_time;
  double softmax_time;
  double outer_product_time;
  double change_weights_time;

 public:
  mlr_computer()
    : num_compobj_(1), compobj_rank_(0),
      staleness_(0),
      accuracy_(-1),
      unit_of_work_per_clock_(1),
      work_per_clock_(1),
      learning_rate_(0.4),
      decay_rate_(0.95),
      cur_clock_(0),
      cpu_worker_(0)
  { }

  void read_metafile(string datafile_prefix) {
    string meta_file = datafile_prefix + ".meta";
    MetafileReader mreader(meta_file);
    num_train_data_ = mreader.get_int32("num_train_this_partition");
    feature_dim_ = mreader.get_int32("feature_dim");
    num_labels_ = mreader.get_int32("num_labels");
    read_format_ = mreader.get_string("format");
    feature_one_based_ = mreader.get_bool("feature_one_based");
    label_one_based_ = mreader.get_bool("label_one_based");
    snappy_compressed_ = mreader.get_bool("snappy_compressed");
    assert(feature_dim_ <= ROW_DATA_SIZE);
  }

  /* Read data */
  void read_data(string datafile_prefix) {
    read_metafile(datafile_prefix);
    string train_file = datafile_prefix
      + (1 ? "" : "." + boost::lexical_cast<uint>(compobj_rank_));
    // LOG(INFO) << "Reading train file: " << train_file;
    vector<vector<float> > train_features_tmp;
    if (read_format_ == "bin") {
      // if (compobj_rank_ == 0) {
      if (false) {
        ReadDataLabelBinary2(train_file, feature_dim_, num_train_data_,
            &train_features_tmp, &train_labels_, feature_one_based_,
            label_one_based_);
      } else {
        ReadDataLabelBinary(train_file, feature_dim_, num_train_data_,
            &train_features_tmp, &train_labels_, feature_one_based_,
            label_one_based_);
      }
    } else if (read_format_ == "libsvm") {
      ReadDataLabelLibSVM(train_file, feature_dim_, num_train_data_,
          &train_features_tmp, &train_labels_, feature_one_based_,
          label_one_based_, snappy_compressed_);
    }

    train_feature_mems_.resize(train_features_tmp.size());
    uint num_row_bytes = ROW_DATA_SIZE * sizeof(val_t);
    assert(feature_dim_ <= ROW_DATA_SIZE);
    for (uint i = 0; i < train_features_tmp.size(); i++) {
      train_feature_mems_[i] = new SyncedMemory(num_row_bytes);
      void *train_feature_mem_ptr = train_feature_mems_[i]->mutable_cpu_data();
      memcpy(train_feature_mem_ptr, train_features_tmp[i].data(),
             feature_dim_ * sizeof(val_t));
      if (!cpu_worker_) {
        train_feature_mems_[i]->mutable_gpu_data();
      }
    }

    uint div = train_feature_mems_.size() / num_compobj_;
    uint res = train_feature_mems_.size() % num_compobj_;
    batch_offset_ =
        div * compobj_rank_ + (res > compobj_rank_ ? compobj_rank_ : res);
    batch_size_ = div + (res > compobj_rank_ ? 1 : 0);
  }

  void initialize() {
    uint w_mem_size = num_labels_ * ROW_DATA_SIZE * sizeof(val_t);
    w_cache_mem_ = new SyncedMemory(w_mem_size);
    w_delta_mem_ = new SyncedMemory(w_mem_size);
    y_mem_ = new SyncedMemory(num_labels_ * sizeof(val_t));

    /* Pre-allocate memory */
    if (cpu_worker_) {
      y_mem_->mutable_cpu_data();
      w_cache_mem_->mutable_cpu_data();
      w_delta_mem_->mutable_cpu_data();
    } else {
      y_mem_->mutable_gpu_data();
      w_cache_mem_->mutable_cpu_data();
      w_delta_mem_->mutable_cpu_data();
      w_cache_mem_->mutable_gpu_data();
      w_delta_mem_->mutable_gpu_data();
    }

    refresh_weights_time = 0;
    alloc_mem_time = 0;
    dotproduct_time = 0;
    softmax_time = 0;
    outer_product_time = 0;
    change_weights_time = 0;
  }

  void refresh_weights() {
    /* Shift memory */
    tbb::tick_count refresh_weights_start = tbb::tick_count::now();
    if (!cpu_worker_) {
      w_cache_mem_->mutable_cpu_data();
      w_delta_mem_->mutable_cpu_data();
      w_cache_mem_->mutable_gpu_data();
      w_delta_mem_->mutable_gpu_data();
    }
    refresh_weights_time +=
      (tbb::tick_count::now() - refresh_weights_start).seconds();
  }

  void change_weights() {
    /* Shift memory */
    tbb::tick_count change_weights_start = tbb::tick_count::now();
    w_cache_mem_->mutable_cpu_data();
    w_delta_mem_->mutable_cpu_data();
    change_weights_time +=
      (tbb::tick_count::now() - change_weights_start).seconds();

    val_t *w_cache = reinterpret_cast<val_t *>(w_cache_mem_->mutable_cpu_data());
    val_t sum = 0.0;
    uint num_possitives = 0;
    for (uint i = 0; i < num_labels_ * ROW_DATA_SIZE; i++) {
      sum += w_cache[i];
      if (w_cache[i] > 0) {
        num_possitives++;
      }
    }
    // cout << "W cache row sums:" << endl;
    // for (uint i = 0; i < num_labels_; i++) {
      // val_t row_sum = 0.0;
      // for (uint j = 0; j < ROW_DATA_SIZE; j++) {
        // row_sum += w_cache[i * ROW_DATA_SIZE + j];
      // }
      // cout << row_sum << ' ';
      // sum += row_sum;
    // }
    // cout << endl;
    cout << "num_possitives = " << num_possitives << endl;
    cout << "sum = " << sum << endl;
  }

  void SingleDataSGD(SyncedMemory *feature_mem, uint label, val_t learning_rate) {
    tbb::tick_count alloc_mem_start = tbb::tick_count::now();
    val_t *y;
    val_t *feature;
    val_t *w_cache;
    val_t *w_delta;
    if (cpu_worker_) {
      y = reinterpret_cast<val_t *>(y_mem_->mutable_cpu_data());
      feature = reinterpret_cast<val_t *>(feature_mem->mutable_cpu_data());
      w_cache = reinterpret_cast<val_t *>(w_cache_mem_->mutable_cpu_data());
      w_delta = reinterpret_cast<val_t *>(w_delta_mem_->mutable_cpu_data());
    } else {
      y = reinterpret_cast<val_t *>(y_mem_->mutable_gpu_data());
      feature = reinterpret_cast<val_t *>(feature_mem->mutable_gpu_data());
      w_cache = reinterpret_cast<val_t *>(w_cache_mem_->mutable_gpu_data());
      w_delta = reinterpret_cast<val_t *>(w_delta_mem_->mutable_gpu_data());
    }
    empty_gpu_func();
    tbb::tick_count predict_start = tbb::tick_count::now();
    alloc_mem_time += (predict_start - alloc_mem_start).seconds();

    if (cpu_worker_) {
      caffe::caffe_cpu_gemv<val_t>(
        CblasNoTrans, num_labels_, ROW_DATA_SIZE, 1, w_cache, feature, 0, y);
    } else {
      caffe::caffe_gpu_gemv<val_t>(
        CblasNoTrans, num_labels_, ROW_DATA_SIZE, 1, w_cache, feature, 0, y);
      cudaDeviceSynchronize();
    }
    tbb::tick_count dotproduct_end = tbb::tick_count::now();
    dotproduct_time += (dotproduct_end - predict_start).seconds();

    if (cpu_worker_) {
      SoftmaxAndAdjust(y, num_labels_, label);
    } else {
      SoftmaxAndAdjust_gpu(y, num_labels_, label);
      cudaDeviceSynchronize();
    }
    tbb::tick_count softmax_end = tbb::tick_count::now();
    softmax_time += (softmax_end - dotproduct_end).seconds();

    // outer product
    if (cpu_worker_) {
      caffe::caffe_cpu_gemm<val_t>(
        CblasNoTrans, CblasNoTrans, num_labels_, ROW_DATA_SIZE, 1,
        -learning_rate, y, feature, 1, w_cache);
      caffe::caffe_cpu_gemm<val_t>(
        CblasNoTrans, CblasNoTrans, num_labels_, ROW_DATA_SIZE, 1,
        -learning_rate, y, feature, 1, w_delta);
    } else {
      caffe::caffe_gpu_gemm<val_t>(
        CblasNoTrans, CblasNoTrans, num_labels_, ROW_DATA_SIZE, 1,
        -learning_rate, y, feature, 1, w_cache);
      caffe::caffe_gpu_gemm<val_t>(
        CblasNoTrans, CblasNoTrans, num_labels_, ROW_DATA_SIZE, 1,
        -learning_rate, y, feature, 1, w_delta);
      cudaDeviceSynchronize();
    }
    outer_product_time +=
      (tbb::tick_count::now() - softmax_end).seconds();
  }

  void compute() {
    refresh_weights();
    
    val_t curr_learning_rate = learning_rate_ * pow(decay_rate_, cur_clock_);
    for (uint i = batch_offset_; i < batch_offset_ + batch_size_; i++) {
      SingleDataSGD(train_feature_mems_[i], train_labels_[i], curr_learning_rate);
    }
    // for (uint r = 0; r < 2; r++) {
      // if (r != 0) {
        // refresh_weights();
      // }
      // for (uint i = 0; i < 100; i++) {
        // SingleDataSGD(train_feature_mems_[i], train_labels_[i], curr_learning_rate);
      // }
    // }

    change_weights();

    cur_clock_++;
  }
};

int main(int argc, char* argv[]) {
  uint cpu_worker = 0;
  uint batch_size = 1000;
  if (argc > 1) {
    cpu_worker = atoi(argv[1]);
  }
  if (argc > 2) {
    batch_size = atoi(argv[2]);
  }

  /* Create cublas_handle */
  if (!cpu_worker) {
    caffe::Caffe::cublas_handle();
  }

  mlr_computer computer;
  computer.cpu_worker_ = cpu_worker;

  /* Read data */
  // string data_file = "/proj/BigLearning/hengganc/data/mlr_data/imagenet_llc/imnet.train.50.train";
  string data_file = "/tank/projects/biglearning/jinlianw/data/mlr_data/imagenet_llc/imnet.train.1000.train";
  computer.read_data(data_file);
  computer.batch_size_ = batch_size;

  /* Set initial values */
  computer.initialize();

  uint num_iterations = 1;
  for (uint i = 0; i < num_iterations; ++i) {
    computer.compute();
  }

  cout << "refresh_weights_time = " << computer.refresh_weights_time << endl;
  cout << "alloc_mem_time = " << computer.alloc_mem_time << endl;
  cout << "dotproduct_time = " << computer.dotproduct_time << endl;
  cout << "softmax_time = " << computer.softmax_time << endl;
  cout << "outer_product_time = " << computer.outer_product_time << endl;
  cout << "change_weights_time = " << computer.change_weights_time << endl;
  double total_compute_time =
      computer.alloc_mem_time + computer.dotproduct_time +
      computer.softmax_time + computer.outer_product_time;
  cout << "total_compute_time = " << total_compute_time << endl;
  double total_time = total_compute_time +
      computer.refresh_weights_time + computer.change_weights_time;
  cout << "total_time = " << total_time << endl;
}
