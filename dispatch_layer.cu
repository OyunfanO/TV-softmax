#include <algorithm>
#include <vector>
#include <iostream>

#include "caffe/layers/dispatch_layer.hpp"

namespace caffe {

template <typename Dtype>
void DispatchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int count = bottom[0]->count();

  for (int top_id = 0; top_id < top.size(); ++top_id) {
    Dtype* top_data = top[top_id]->mutable_gpu_data();
    cudaMemcpy(top_data, bottom_data, sizeof(Dtype)*count, cudaMemcpyDefault);
  }

  //Dtype o_asum;
  //caffe_gpu_asum(count,bottom_data,&o_asum);
  //top[2]->mutable_cpu_data()[0] = Dtype(alpha)*o_asum/Dtype(count);

  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
void DispatchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* top0_diff = top[0]->gpu_diff();
  const Dtype* top1_diff = top[1]->gpu_diff();

  const int count = bottom[0]->count();
  caffe_gpu_add(count, top0_diff, top1_diff, bottom_diff);

  for (int top_id = 2; top_id < top.size(); ++top_id) {
    const Dtype* top_diff = top[top_id]->gpu_diff();
    caffe_gpu_add(count, top_diff, bottom_diff, bottom_diff);
  }

  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(DispatchLayer);


}  // namespace caffe
