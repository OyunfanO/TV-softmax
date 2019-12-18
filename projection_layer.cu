#include <algorithm>
#include <vector>

#include "caffe/layers/projection_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ProjectionForward(const int n, const Dtype* in1, const Dtype* in2, Dtype* out1, Dtype* out2,
    const Dtype* norm) {
  CUDA_KERNEL_LOOP(index, n) {
    out1[index] = norm[index] > 1 ? in1[index]/norm[index] : in1[index]; 
    out2[index] = norm[index] > 1 ? in2[index]/norm[index] : in2[index]; 
  }
}

template <typename Dtype>
void ProjectionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* xi1 = bottom[0]->gpu_data();
  const Dtype* xi2 = bottom[1]->gpu_data();
  Dtype* eta1 = top[0]->mutable_gpu_data();
  Dtype* eta2 = top[1]->mutable_gpu_data();
  const int count = bottom[0]->count();

  //x1_s = xi1^2, x2_s = xi2^2
  caffe_gpu_mul(count,xi1,xi1,xi1_s->mutable_gpu_data());
  caffe_gpu_mul(count,xi2,xi2,xi2_s->mutable_gpu_data());
  //norm_xi = xi1^2+xi2^2
  caffe_gpu_add(count, xi1_s->cpu_data(),xi2_s->cpu_data(),norm_xi->mutable_cpu_data());
  //norm_xi = sqrt(xi1^2+xi2^2)
  caffe_gpu_powx(count,norm_xi->gpu_data(),Dtype(0.5),norm_xi->mutable_gpu_data());

  ProjectionForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, xi1, xi2, eta1, eta2, norm_xi->gpu_data());


  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void ProjectionBackward(const int n, const Dtype* top1_diff,
    const Dtype* bottom1_data, const Dtype* top2_diff,
    const Dtype* bottom2_data, Dtype* bottom1_diff, Dtype* bottom2_diff, const Dtype* norm) {
  CUDA_KERNEL_LOOP(index, n) {
    bottom1_diff[index] = norm[index] > 1 ? -top2_diff[index]*bottom1_data[index]*bottom2_data[index]*powf(norm[index],-3.0) + top1_diff[index]*(1/norm[index]-powf(bottom1_data[index],Dtype(2))*powf(norm[index],Dtype(-3.0))):top1_diff[index];

    bottom2_diff[index] = norm[index] > 1 ? -top1_diff[index]*bottom1_data[index]*bottom2_data[index]*powf(norm[index],-3.0) + top2_diff[index]*(1/norm[index]-powf(bottom2_data[index],Dtype(2))*powf(norm[index],Dtype(-3.0))):top2_diff[index];
  }
}

template <typename Dtype>
void ProjectionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* xi1 = bottom[0]->gpu_data();
  const Dtype* xi2 = bottom[1]->gpu_data();
  const Dtype* norm = norm_xi->gpu_data();
  const Dtype* eta1_diff = top[0]->gpu_diff();
  const Dtype* eta2_diff = top[1]->gpu_diff();
  const int count = bottom[0]->count();

  Dtype* xi1_diff = bottom[0]->mutable_gpu_diff();
  Dtype* xi2_diff = bottom[1]->mutable_gpu_diff();

  // NOLINT_NEXT_LINE(whitespace/operators)
  ProjectionBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, eta1_diff, xi1, eta2_diff, xi2, xi1_diff, xi2_diff, norm);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(ProjectionLayer);


}  // namespace caffe
