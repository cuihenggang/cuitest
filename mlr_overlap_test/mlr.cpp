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

size_t count = 21504 * 1000;
size_t size = count * sizeof(float);
void *cpu_ptr;
void *cpu_ptr2;
void *gpu_ptr;
void *gpu_ptr2;

static void *thread_run(void *arg) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  tbb::tick_count tick_start = tbb::tick_count::now();
  for (size_t r = 0; r < 100; r++) {
    cudaMemcpyAsync(gpu_ptr2, gpu_ptr, size, cudaMemcpyDefault, stream);
    cudaStreamSynchronize(stream);
    cudaMemcpyAsync(gpu_ptr, cpu_ptr, size, cudaMemcpyDefault, stream);
    cudaStreamSynchronize(stream);
    cudaMemcpyAsync(cpu_ptr, gpu_ptr2, size, cudaMemcpyDefault, stream);
    cudaStreamSynchronize(stream);
  }

  double compute_time = (tbb::tick_count::now() - tick_start).seconds();
  cout << "memcpy_time = " << compute_time << endl;
  cudaStreamDestroy(stream);
}

using caffe::SyncedMemory;

void SoftmaxBatchAndAdjust_gpu(
    cudaStream_t cuda_stream,
    size_t n, size_t size, val_t *vecs, const uint *labels);
void SoftmaxBatchAndEntropyLoss_gpu(
    cudaStream_t cuda_stream,
    size_t n, size_t size, val_t *vecs, const uint *labels, val_t *losses);
void SoftmaxBatchAndZeroOneLoss_gpu(
    cudaStream_t cuda_stream,
    size_t n, size_t size, val_t *vecs, const uint *labels, val_t *losses);

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

  cudaStream_t cuda_stream_;
  cublasHandle_t cublas_handle_;

  SyncedMemory *train_feature_mem_;
  SyncedMemory *train_label_mem_;
  SyncedMemory *w_cache_mem_;
  SyncedMemory *w_delta_mem_;
  SyncedMemory *y_batch_mem_;
  SyncedMemory *loss_y_batch_mem_;
  SyncedMemory *loss_mem_;
  // RowData loss_row_;
  vector<uint> row_ids_;

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
      cur_clock_(0)
  {
    cudaStreamCreate(&cuda_stream_);
    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
      assert(0);
    }
    cublasSetStream(cublas_handle_, cuda_stream_);
  }

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
    vector<int> train_labels_tmp;
    if (read_format_ == "bin") {
      // if (compobj_rank_ == 0) {
      if (false) {
        ReadDataLabelBinary2(train_file, feature_dim_, num_train_data_,
            &train_features_tmp, &train_labels_tmp, feature_one_based_,
            label_one_based_);
      } else {
        ReadDataLabelBinary(train_file, feature_dim_, num_train_data_,
            &train_features_tmp, &train_labels_tmp, feature_one_based_,
            label_one_based_);
      }
    } else if (read_format_ == "libsvm") {
      ReadDataLabelLibSVM(train_file, feature_dim_, num_train_data_,
          &train_features_tmp, &train_labels_tmp, feature_one_based_,
          label_one_based_, snappy_compressed_);
    }

    assert(feature_dim_ <= ROW_DATA_SIZE);
    train_feature_mem_ = new SyncedMemory(
        train_features_tmp.size() * ROW_DATA_SIZE * sizeof(val_t));
    RowData *train_features =
        reinterpret_cast<RowData *>(train_feature_mem_->mutable_cpu_data());
    for (uint i = 0; i < train_features_tmp.size(); i++) {
      RowData *train_feature = &(train_features[i]);
      memcpy(reinterpret_cast<void *>(train_feature),
          train_features_tmp[i].data(), feature_dim_ * sizeof(val_t));
    }
    train_feature_mem_->mutable_gpu_data();

    assert(sizeof(uint) == sizeof(int));
    train_label_mem_ = new SyncedMemory(
        train_labels_tmp.size() * sizeof(uint));
    void *train_labels_ptr =
        reinterpret_cast<void *>(train_label_mem_->mutable_cpu_data());
    memcpy(train_labels_ptr, train_labels_tmp.data(),
        train_label_mem_->size());
    train_label_mem_->mutable_gpu_data();

    uint div = train_features_tmp.size() / num_compobj_;
    uint res = train_features_tmp.size() % num_compobj_;
    batch_offset_ =
        div * compobj_rank_ + (res > compobj_rank_ ? compobj_rank_ : res);
    batch_size_ = div + (res > compobj_rank_ ? 1 : 0);
  }

  void initialize() {
    w_cache_mem_ = new SyncedMemory(
        num_labels_ * feature_dim_ * sizeof(val_t));
    w_delta_mem_ = new SyncedMemory(
        num_labels_ * feature_dim_ * sizeof(val_t));
    y_batch_mem_ = new SyncedMemory(
        batch_size_ * num_labels_ * sizeof(val_t));
    /* For calculating loss */
    loss_y_batch_mem_ = new SyncedMemory(
        batch_size_ * num_labels_ * sizeof(val_t));
    loss_mem_ = new SyncedMemory(
        batch_size_ * sizeof(val_t));

    /* Pre-allocate memory */
    y_batch_mem_->mutable_gpu_data();
    w_cache_mem_->mutable_gpu_data();
    w_delta_mem_->mutable_gpu_data();
    loss_y_batch_mem_->mutable_gpu_data();
    loss_mem_->mutable_gpu_data();

    row_ids_.resize(num_labels_);
  }

  void batch_sgd(uint cur_clock) {
    val_t learning_rate = learning_rate_ * pow(decay_rate_, cur_clock);
    // tbb::tick_count alloc_mem_start = tbb::tick_count::now();
    val_t *y_batch;
    const RowData *feature_rows;
    const val_t *feature_batch;
    const uint *labels;
    const val_t *w_cache;
    val_t *w_delta;
    val_t *loss_y_batch;
    val_t *losses;

    y_batch = reinterpret_cast<val_t *>(y_batch_mem_->mutable_gpu_data());
    feature_rows = reinterpret_cast<const RowData *>(
        train_feature_mem_->gpu_data());
    feature_batch =
        reinterpret_cast<const val_t *>(&feature_rows[batch_offset_]);
    const uint *all_labels =
        reinterpret_cast<const uint *>(train_label_mem_->gpu_data());
    labels = &all_labels[batch_offset_];
    w_cache = reinterpret_cast<const val_t *>(w_cache_mem_->gpu_data());
    w_delta = reinterpret_cast<val_t *>(w_delta_mem_->mutable_gpu_data());
    loss_y_batch =
        reinterpret_cast<val_t *>(loss_y_batch_mem_->mutable_gpu_data());
    losses = reinterpret_cast<val_t *>(loss_mem_->mutable_gpu_data());
    cudaStreamSynchronize(cuda_stream_);

    tbb::tick_count predict_start = tbb::tick_count::now();
    // alloc_mem_time += (predict_start - alloc_mem_start).seconds();

    // caffe::caffe_gpu_gemm<val_t>(cublas_handle_,
      // CblasNoTrans, CblasTrans, batch_size_, num_labels_, feature_dim_,
      // 1, feature_batch, w_cache, 0, y_batch);
    // cudaStreamSynchronize(cuda_stream_);

    tbb::tick_count dotproduct_end = tbb::tick_count::now();
    dotproduct_time += (dotproduct_end - predict_start).seconds();

    // cudaMemcpyAsync(loss_y_batch, y_batch, y_batch_mem_->size(),
        // cudaMemcpyDefault, cuda_stream_);
    // SoftmaxBatchAndAdjust_gpu(
        // cuda_stream_, batch_size_, num_labels_, y_batch, labels);
    // SoftmaxBatchAndEntropyLoss_gpu(
    // SoftmaxBatchAndZeroOneLoss_gpu(
        // cuda_stream_, batch_size_, num_labels_, loss_y_batch, labels, losses);
    // cudaStreamSynchronize(cuda_stream_);

    // assert(cur_clock < ROW_DATA_SIZE);
    // loss_row_.init();
    // loss_row_.data[cur_clock] = loss;
    tbb::tick_count softmax_end = tbb::tick_count::now();
    softmax_time += (softmax_end - dotproduct_end).seconds();

    // outer product
    caffe::caffe_gpu_gemm<val_t>(cublas_handle_,
      CblasTrans, CblasNoTrans, num_labels_, feature_dim_, batch_size_,
      -learning_rate, y_batch, feature_batch, 0, w_delta);
    cudaStreamSynchronize(cuda_stream_);

    /* If we do multiple batches per clock, we should add w_delta to w_cache */
    outer_product_time +=
      (tbb::tick_count::now() - softmax_end).seconds();

    // val_t loss = 0;
    // if (gpu_worker_) {
// #if !defined(CPU_ONLY)
      // loss = caffe::caffe_gpu_asum<val_t>(cublas_handle_, batch_size_, losses);
// #endif
    // } else {
      // loss = caffe::caffe_cpu_asum<val_t>(batch_size_, losses);
    // }
    // cout << "loss = " << loss << endl;
  }

  void compute() {
    batch_sgd(1);
  }
};

int main(int argc, char* argv[]) {
  uint batch_size = 1000;
  if (argc > 2) {
    batch_size = atoi(argv[2]);
  }

  mlr_computer computer;

  /* Read data */
  // string data_file = "/proj/BigLearning/hengganc/data/mlr_data/imagenet_llc/imnet.train.50.train";
  string data_file = "/tank/projects/biglearning/jinlianw/data/mlr_data/imagenet_llc/imnet.train.1000.train";
  computer.read_data(data_file);
  computer.batch_size_ = batch_size;

  /* Set initial values */
  computer.initialize();

  cudaMallocHost(&cpu_ptr, size);
  cudaMallocHost(&cpu_ptr2, size);
  cudaMalloc(&gpu_ptr, size);
  cudaMalloc(&gpu_ptr2, size);
  pthread_t thread_id;
  pthread_attr_t thread_attr;
  void *res;
  pthread_attr_init(&thread_attr);
  pthread_create(&thread_id, &thread_attr, thread_run, NULL);

  uint num_iterations = 20;
  tbb::tick_count compute_start = tbb::tick_count::now();
  for (uint i = 0; i < num_iterations; ++i) {
    computer.compute();
  }
  double total_time = (tbb::tick_count::now() - compute_start).seconds();

  cout << "dotproduct_time = " << computer.dotproduct_time << endl;
  cout << "softmax_time = " << computer.softmax_time << endl;
  cout << "outer_product_time = " << computer.outer_product_time << endl;
  cout << "total_time = " << total_time << endl;
  
  pthread_join(thread_id, &res);
}
