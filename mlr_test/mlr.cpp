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
// #include "math_functions.hpp"

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

  // vector<RowData> train_features_;
  // vector<int> train_labels_;
  // vector<RowData> w_cache_mems_;
  // vector<RowData> w_delta_mems_;

  uint num_trains_;
  vector<SyncedMemory *> train_feature_mems_;
  vector<int> train_labels_;
  vector<SyncedMemory *> w_cache_mems_;
  vector<SyncedMemory *> w_delta_mems_;

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
    w_cache_mems_.resize(num_labels_);
    w_delta_mems_.resize(num_labels_);
    for (uint i = 0; i < num_labels_; i++) {
      w_cache_mems_[i] = new SyncedMemory(ROW_DATA_SIZE * sizeof(val_t));
      w_delta_mems_[i] = new SyncedMemory(ROW_DATA_SIZE * sizeof(val_t));
    }

    make_y_time = 0;
    predict_time = 0;
    outer_product_time = 0;
    dotproduct_time = 0;
    softmax_time = 0;
  }

  void refresh_weights() {

  }

  void change_weights() {
    // Zero delta.
    // for (uint i = 0; i < num_labels_; ++i) {
      // RowData& w_delta_i = w_delta_mems_[i];
      // for (uint j = 0; j < feature_dim_; j++) {
        // w_delta_i.data[j] = 0;
      // }
    // }
  }

  void Predict(vector<float> &y_vec, SyncedMemory *feature_mem) {
    tbb::tick_count make_y_start = tbb::tick_count::now();
    y_vec.resize(num_labels_);
    tbb::tick_count make_y_end = tbb::tick_count::now();
    make_y_time += (make_y_end - make_y_start).seconds();

    for (uint i = 0; i < num_labels_; ++i) {
      y_vec[i] =
        DenseDenseFeatureDotProduct(
          reinterpret_cast<const float *>(feature_mem->cpu_data()),
          reinterpret_cast<const float *>(w_cache_mems_[i]->cpu_data()),
          feature_dim_);
    }
    tbb::tick_count dotproduct_end = tbb::tick_count::now();
    dotproduct_time += (dotproduct_end - make_y_end).seconds();
    
    Softmax(&y_vec);
    softmax_time += (tbb::tick_count::now() - dotproduct_end).seconds();
  }

  void SingleDataSGD(SyncedMemory *feature_mem, uint label, float learning_rate) {
    tbb::tick_count predict_start = tbb::tick_count::now();
    vector<float> y_vec;
    Predict(y_vec, feature_mem);
    y_vec[label] -= 1.; // See Bishop PRML (2006) Eq. (4.109)
    tbb::tick_count predict_end = tbb::tick_count::now();
    predict_time += (predict_end - predict_start).seconds();

    // outer product
    for (uint i = 0; i < num_labels_; ++i) {
      // w_cache_mems_[i] += -\eta * y_vec[i] * feature
      FeatureScaleAndAdd(
        -learning_rate * y_vec[i],
        reinterpret_cast<const float *>(feature_mem->cpu_data()),
        reinterpret_cast<float *>(w_cache_mems_[i]->mutable_cpu_data()),
        feature_dim_);
      FeatureScaleAndAdd(
        -learning_rate * y_vec[i],
        reinterpret_cast<const float *>(feature_mem->cpu_data()),
        reinterpret_cast<float *>(w_delta_mems_[i]->mutable_cpu_data()), feature_dim_);
    }
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
  string data_file = "/proj/BigLearning/hengganc/data/mlr_data/imagenet_llc/imnet.train.50.train";
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
