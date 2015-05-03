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

using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::istringstream;

typedef uint32_t uint;

#define ROW_DATA_SIZE 21504
typedef float val_t;
struct ArrayData {
  val_t data[ROW_DATA_SIZE];
  void init() {
    for (uint32_t i = 0; i < ROW_DATA_SIZE; i++) {
      data[i] = 0;
    }
  }
  ArrayData() {
    init();
  }
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & data;
  }
};
typedef ArrayData RowData;
typedef ArrayData RowOpVal;


#include "mlr-util.hpp"
#include "metafile-reader.hpp"

#include "syncedmem.hpp"
#include "math_functions.hpp"

using caffe::SyncedMemory;

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

  uint w_row_size_;
  vector<SyncedMemory *> train_feature_mems_;
  vector<int> train_labels_;
  SyncedMemory *w_cache_mem_;
  SyncedMemory *w_delta_mem_;

  double make_y_time;
  double predict_time;
  double outer_product_time;
  double dotproduct_time;
  double softmax_time;

 public:
  mlr_computer()
    : num_compobj_(1), compobj_rank_(0),
      staleness_(0),
      accuracy_(-1),
      unit_of_work_per_clock_(1),
      work_per_clock_(1),
      learning_rate_(0.4),
      decay_rate_(0.95),
      cur_clock_(0)
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
      memcpy(train_feature_mem_ptr, train_features_tmp[i].data(), feature_dim_ * sizeof(float));
    }

    uint div = train_feature_mems_.size() / num_compobj_;
    uint res = train_feature_mems_.size() % num_compobj_;
    batch_offset_ =
        div * compobj_rank_ + (res > compobj_rank_ ? compobj_rank_ : res);
    batch_size_ = div + (res > compobj_rank_ ? 1 : 0);
  }

  void initialize() {
    w_row_size_ = ROW_DATA_SIZE * sizeof(val_t);
    uint w_mem_size = num_labels_ * w_row_size_;
    w_cache_mem_ = new SyncedMemory(w_mem_size);
    w_delta_mem_ = new SyncedMemory(w_mem_size);

    make_y_time = 0;
    predict_time = 0;
    outer_product_time = 0;
    dotproduct_time = 0;
    softmax_time = 0;
  }

  void refresh_weights() {

  }

  void change_weights() {
    float *w_cache = reinterpret_cast<float *>(w_cache_mem_->mutable_cpu_data());
    float count = 0.0;
    for (uint i = 0; i < num_labels_ * ROW_DATA_SIZE; i++) {
      count += w_cache[i];
    }
    cout << "count = " << count << endl;
  }

  void Predict(float *y, float *feature, float *w_cache) {
    tbb::tick_count make_y_end = tbb::tick_count::now();
#if defined(CPU_ONLY)
    caffe::caffe_cpu_gemv<float>(
#else
    caffe::caffe_gpu_gemv<float>(
#endif
      CblasNoTrans, num_labels_, ROW_DATA_SIZE, 1, w_cache, feature, 0, y);
    tbb::tick_count dotproduct_end = tbb::tick_count::now();
    dotproduct_time += (dotproduct_end - make_y_end).seconds();
    
    Softmax(y, num_labels_);
    softmax_time += (tbb::tick_count::now() - dotproduct_end).seconds();
  }

  void SingleDataSGD(SyncedMemory *feature_mem, uint label, float learning_rate) {
    tbb::tick_count predict_start = tbb::tick_count::now();
    SyncedMemory y_mem(num_labels_ * sizeof(float));
#if defined(CPU_WORKER)
    float *y = reinterpret_cast<float *>(y_mem.mutable_cpu_data());
    float *feature = reinterpret_cast<float *>(feature_mem->mutable_cpu_data());
    float *w_cache = reinterpret_cast<float *>(w_cache_mem_->mutable_cpu_data());
    float *w_delta = reinterpret_cast<float *>(w_delta_mem_->mutable_cpu_data());
#else
    float *y = reinterpret_cast<float *>(y_mem.mutable_gpu_data());
    float *feature = reinterpret_cast<float *>(feature_mem->mutable_gpu_data());
    float *w_cache = reinterpret_cast<float *>(w_cache_mem_->mutable_gpu_data());
    float *w_delta = reinterpret_cast<float *>(w_delta_mem_->mutable_gpu_data());
#endif
    Predict(y, feature, w_cache);
    y[label] -= 1.; // See Bishop PRML (2006) Eq. (4.109)
    tbb::tick_count predict_end = tbb::tick_count::now();
    predict_time += (predict_end - predict_start).seconds();

    // outer product
#if defined(CPU_WORKER)
    caffe::caffe_cpu_gemm<float>(
#else
    caffe::caffe_gpu_gemm<float>(
#endif
      CblasNoTrans, CblasNoTrans, num_labels_, ROW_DATA_SIZE, 1,
      -learning_rate, y, feature, 1, w_cache);
#if defined(CPU_WORKER)
    caffe::caffe_cpu_gemm<float>(
#else
    caffe::caffe_gpu_gemm<float>(
#endif
      CblasNoTrans, CblasNoTrans, num_labels_, ROW_DATA_SIZE, 1,
      -learning_rate, y, feature, 1, w_delta);
    outer_product_time +=
      (tbb::tick_count::now() - predict_end).seconds();
  }

  void compute() {
    refresh_weights();

    float curr_learning_rate = learning_rate_ * pow(decay_rate_, cur_clock_);
    for (uint i = batch_offset_; i < batch_offset_ + batch_size_; i++) {
      SingleDataSGD(train_feature_mems_[i], train_labels_[i], curr_learning_rate);
    }

    change_weights();
    cur_clock_++;
  }
};

int main(int argc, char* argv[]) {
  mlr_computer computer;

  /* Read data */
  // string data_file = "/proj/BigLearning/hengganc/data/mlr_data/imagenet_llc/imnet.train.50.train";
  string data_file = "/tank/projects/biglearning/jinlianw/data/mlr_data/imagenet_llc/imnet.train.50.train";
  computer.read_data(data_file);

  /* Set initial values */
  computer.initialize();

  uint num_iterations = 1;
  for (uint i = 0; i < num_iterations; ++i) {
    computer.compute();
  }

  cout << "predict_time = " << computer.predict_time << endl;
  cout << "outer_product_time = " << computer.outer_product_time << endl;
  cout << "make_y_time = " << computer.make_y_time << endl;
  cout << "dotproduct_time = " << computer.dotproduct_time << endl;
  cout << "softmax_time = " << computer.softmax_time << endl;
}
