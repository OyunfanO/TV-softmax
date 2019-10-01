#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/regularizedO_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>

namespace caffe {

template <typename Dtype>
__global__ void RegularizedOForward(const int nthreads,
    const Dtype* const bottom0_data, const Dtype* const bottom1_data, const int channels,
    const int height, const int width, Dtype* const top_data)
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    //current pixel location
    const int pw = index % width;
    const int ph = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    const Dtype* const bottom0_slice = bottom0_data + (n * channels + c) * height * width;
    const Dtype* const bottom1_slice = bottom1_data + (n * channels + c) * height * width;

     if(pw>0 && pw<width-1 && ph>0 && ph<height-1)
    {
      top_data[index] = bottom0_slice[ph*width+pw] - bottom0_slice[ph*width+pw-1] + bottom1_slice[ph*width+pw] - bottom1_slice[(ph-1)*width+pw];
    }
    else if(pw==0 && ph>0 && ph<height-1)
    {
      top_data[index] = bottom0_slice[ph*width+pw] + bottom1_slice[ph*width+pw] - bottom1_slice[(ph-1)*width+pw];
    }
    else if(pw==width-1 && ph>0 && ph<height-1)
    {
      top_data[index] = - bottom0_slice[ph*width+pw-1] + bottom1_slice[ph*width+pw] - bottom1_slice[(ph-1)*width+pw];
    }
    else if(pw>0 && pw<width-1 && ph==0)
    {
      top_data[index] = bottom0_slice[ph*width+pw] - bottom0_slice[ph*width+pw-1] + bottom1_slice[ph*width+pw];
    }
    else if(pw>0 && pw<width-1 && ph==height-1)
    {
      top_data[index] = bottom0_slice[ph*width+pw] - bottom0_slice[ph*width+pw-1] - bottom1_slice[(ph-1)*width+pw];
    }
    else if(pw==0 && ph==0)
    {
      top_data[index] = bottom0_slice[ph*width+pw] + bottom1_slice[ph*width+pw];
    } 
    else if(pw==0 && ph==height-1)
    {
      top_data[index] = bottom0_slice[ph*width+pw] - bottom1_slice[(ph-1)*width+pw];
    }
    else if(pw==width-1 && ph==0)
    {
      top_data[index] = - bottom0_slice[ph*width+pw-1] + bottom1_slice[ph*width+pw];
    }
    else if(pw==width-1 && ph==height-1)
    {
      top_data[index] = - bottom0_slice[ph*width+pw-1] - bottom1_slice[(ph-1)*width+pw];
    }
  }
}

template <typename Dtype>
void RegularizedOLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* p0 = bottom[0]->gpu_data();
  const Dtype* p1 = bottom[1]->gpu_data();
  const Dtype* ok = bottom[2]->gpu_data();
  //const Dtype* lambda = this->blobs_[0]->cpu_data();
  //Dtype* mutable_lambda = this->blobs_[0]->mutable_cpu_data();

  Dtype* div_eta = div_eta_.mutable_gpu_data();

  Dtype* tilde_o = top[0]->mutable_gpu_data();

  int count = bottom[0]->count();

  // NOLINT_NEXT_LINE(whitespace/operators)
  RegularizedOForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, p0, p1, channels_,
      height_, width_, tilde_o);

  CUDA_CHECK(cudaMemcpy(div_eta, tilde_o, sizeof(Dtype) * count, cudaMemcpyDefault));

  //top0 = -lambda*divp
  caffe_gpu_scale(count, -Dtype(lambda_), top[0]->gpu_data(), top[0]->mutable_gpu_data());
  //top0 = ok-lamdbda*divp
  caffe_gpu_axpy(count, Dtype(1), ok, top[0]->mutable_gpu_data());

  iter_++;
  //if(iter_%100==1) std::cout<<"lambda: "<<lambda[0]<<" ";
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void RegularizedOBackward(const int nthreads, const Dtype* const top_diff,
    const int channels, const int height, const int width, Dtype* const bottom0_diff, Dtype* const bottom1_diff ) 
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    //current pixel location
    const int pw = index % width;
    const int ph = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    const Dtype* const top_slice = top_diff + (n * channels + c) * height * width;

    if(pw<width-1)
    {
      bottom0_diff[index] = top_slice[ph*width+pw] - top_slice[ph*width+pw+1];
    }
    else if(pw==width-1)
    {
      bottom0_diff[index] = top_slice[ph*width+pw];
    }
    
    if(ph<height-1)
    {
      bottom1_diff[index] = top_slice[ph*width+pw] - top_slice[(ph+1)*width+pw];
    }
    else if(ph==height-1)
    {
      bottom1_diff[index] = top_slice[ph*width+pw];
    }
  }
}


template <typename Dtype>
void RegularizedOLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom0_diff = bottom[0]->mutable_gpu_diff();
  Dtype* bottom1_diff = bottom[1]->mutable_gpu_diff();
  Dtype* bottom2_diff = bottom[2]->mutable_gpu_diff();
  //const Dtype* lambda = this->blobs_[0]->cpu_data();
  const int count = bottom[0]->count();

  CUDA_CHECK(cudaMemcpy(bottom2_diff, top_diff, sizeof(Dtype) * count, cudaMemcpyDefault));

  // NOLINT_NEXT_LINE(whitespace/operators)
  RegularizedOBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, channels_,
      height_, width_, bottom0_diff, bottom1_diff);

  caffe_gpu_scale(count, -Dtype(lambda_), bottom[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
  caffe_gpu_scale(count, -Dtype(lambda_), bottom[1]->gpu_diff(), bottom[1]->mutable_gpu_diff());
  
  /*
  Dtype* lambda_diff = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* div_eta = div_eta_.gpu_data();
  Dtype result;
  caffe_gpu_dot(count, div_eta, top_diff, &result);
  *lambda_diff = -result;
  if(iter_%100==1) std::cout<<"lambda_diff: "<<lambda_diff[0]<<" ";
  */
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(RegularizedOLayer);

}  // namespace caffe
