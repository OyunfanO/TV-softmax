#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/updateXi_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>

namespace caffe {

template <typename Dtype>
__global__ void UpdateXiForward(const int nthreads,
    const Dtype* const bottom_data, const int channels,
    const int height, const int width, Dtype* const top_data, int direction)
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    //current pixel location
    const int pw = index % width;
    const int ph = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    const Dtype* const bottom_slice = bottom_data + (n * channels + c) * height * width;
    //direction = 0 represents x, direction = 1 represents y
    if(direction == 0)
    {
      if(pw<width-1)
      {
        top_data[index] = bottom_slice[ph*width+pw+1] - bottom_slice[ph*width+pw];
      }
      else if(pw==width-1)
      {
        top_data[index] = 0;
      }
    }
    else if(direction ==1)
    {
      if(ph<height-1)
      {
        top_data[index] = bottom_slice[(ph+1)*width+pw] - bottom_slice[ph*width+pw];
      }
      else if(ph==height-1)
      {
        top_data[index] = 0;
      }
    }
  }
}

template <typename Dtype>
void UpdateXiLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // xi(t) = xi(t-1) - tau*lambda*grad(A(t-1))
  const Dtype* bottom_data = bottom[0]->gpu_data(); //A(t-1)
  const Dtype* bottom_xix;//xi(t-1)
  const Dtype* bottom_xiy;
  if(bottom.size()==3)
  {
    bottom_xix = bottom[1]->gpu_data();
    bottom_xiy = bottom[2]->gpu_data();
  }
  //const Dtype* tau = this->blobs_[0]->cpu_data();
  //Dtype* mutable_tau = this->blobs_[0]->mutable_cpu_data();
  
  Dtype* gradx = top[0]->mutable_gpu_data();
  Dtype* grady = top[1]->mutable_gpu_data();

  int count = bottom[0]->count();

  // NOLINT_NEXT_LINE(whitespace/operators)
  UpdateXiForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, channels_,
      height_, width_, gradx, int(0));
  caffe_gpu_scale(count, -(Dtype)tau_*lambda_, top[0]->gpu_data(), top[0]->mutable_gpu_data());

  UpdateXiForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, channels_,
      height_, width_, grady, int(1));
  caffe_gpu_scale(count, -(Dtype)tau_*lambda_, top[1]->gpu_data(), top[1]->mutable_gpu_data());

  if(bottom.size()==3)
  {
    caffe_gpu_add(count, bottom_xix, top[0]->gpu_data(), top[0]->mutable_gpu_data());
    caffe_gpu_add(count, bottom_xiy, top[1]->gpu_data(), top[1]->mutable_gpu_data());
  }

  if(top.size()==4)
  {
    cudaMemcpy(top[2]->mutable_gpu_data(), top[0]->gpu_data(), sizeof(Dtype)*count, cudaMemcpyDefault);
    cudaMemcpy(top[3]->mutable_gpu_data(), top[1]->gpu_data(), sizeof(Dtype)*count, cudaMemcpyDefault);
  }


  iter_++;
  //if(iter_%100==1) std::cout<<"tau: "<<tau[0]<<" ";
  //caffe_gpu_scale(count, -lambda[0]*tau, top[0]->gpu_data(), top[0]->mutable_gpu_data());
  //caffe_gpu_scale(count, -lambda[0]*tau, top[1]->gpu_data(), top[1]->mutable_gpu_data());
  

  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void UpdateXiBackward(const int nthreads, const Dtype* const top0_diff, const Dtype* const top1_diff, 
    const int channels, const int height, const int width, Dtype* const bottom_diff) 
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    //current pixel location
    const int pw = index % width;
    const int ph = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    const Dtype* const top0_slice = top0_diff + (n * channels + c) * height * width;
    const Dtype* const top1_slice = top1_diff + (n * channels + c) * height * width;

    if(pw>0 && pw<width-1 && ph>0 && ph<height-1)
    {
      bottom_diff[index] = top0_slice[ph*width+pw-1] - top0_slice[ph*width+pw] + top1_slice[(ph-1)*width+pw] - top1_slice[ph*width+pw];
    }
    else if(pw==0 && ph>0 && ph<height-1)
    {
      bottom_diff[index] =  - top0_slice[ph*width+pw] + top1_slice[(ph-1)*width+pw] - top1_slice[ph*width+pw];
    }
    else if(pw==width-1 && ph>0 && ph<height-1)
    {
      bottom_diff[index] = top0_slice[ph*width+pw-1] + top1_slice[(ph-1)*width+pw] - top1_slice[ph*width+pw];
    }
    else if(pw>0 && pw<width-1 && ph==0)
    {
      bottom_diff[index] = top0_slice[ph*width+pw-1] - top0_slice[ph*width+pw] - top1_slice[ph*width+pw];
    }
    else if(pw>0 && pw<width-1 && ph==height-1)
    {
      bottom_diff[index] = top0_slice[ph*width+pw-1] - top0_slice[ph*width+pw] + top1_slice[(ph-1)*width+pw];
    }
    else if(pw==0 && ph==0)
    {
      bottom_diff[index] =  - top0_slice[ph*width+pw] - top1_slice[ph*width+pw];
    } 
    else if(pw==0 && ph==height-1)
    {
      bottom_diff[index] = -top0_slice[ph*width+pw-1]  + top1_slice[(ph-1)*width+pw];
    }
    else if(pw==width-1 && ph==0)
    {
      bottom_diff[index] = top0_slice[ph*width+pw-1] - top1_slice[ph*width+pw];
    }
    else if(pw==width-1 && ph==height-1)
    {
      bottom_diff[index] = top0_slice[ph*width+pw-1]+ top1_slice[(ph-1)*width+pw];
    }
  }
}


template <typename Dtype>
void UpdateXiLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top0_diff = top[0]->gpu_diff();
  const Dtype* top1_diff = top[1]->gpu_diff();
  const int count = bottom[0]->count();
  //const Dtype* lambda = bottom[1]->cpu_data();
  //const Dtype* tau = this->blobs_[0]->cpu_data();

  //Blob<Dtype>* temp_bp0, *temp_bp1;
  //temp_bp0->ReshapeLike(*bottom[0]);
  //temp_bp1->ReshapeLike(*bottom[0]);
  //std::cout<<"temp_bp0 count: "<<temp_bp0.count()<<std::endl;
  cudaMemcpy(temp_bp0.mutable_gpu_diff(), top[0]->gpu_diff(), sizeof(Dtype)*count, cudaMemcpyDefault);
  cudaMemcpy(temp_bp1.mutable_gpu_diff(), top[1]->gpu_diff(), sizeof(Dtype)*count, cudaMemcpyDefault);
  if(top.size()==4)
  {
    caffe_gpu_add(count, temp_bp0.gpu_diff(), top[2]->gpu_diff(), temp_bp0.mutable_gpu_diff());
    caffe_gpu_add(count, temp_bp1.gpu_diff(), top[3]->gpu_diff(), temp_bp1.mutable_gpu_diff());
  }


  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  //std::cout<<"bottom[0] count: "<<bottom[0]->count()<<std::endl;
  // NOLINT_NEXT_LINE(whitespace/operators)
  UpdateXiBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, temp_bp0.gpu_diff(), temp_bp1.gpu_diff(), channels_,
      height_, width_,  bottom_diff);
  caffe_gpu_scale(count, -(Dtype)tau_*lambda_, bottom[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());

  if(bottom.size()==3)
  {
	//std::cout<<"bottom size: "<<bottom.size()<<std::endl;
    cudaMemcpy(bottom[1]->mutable_gpu_diff(),temp_bp0.gpu_diff(), sizeof(Dtype)*count, cudaMemcpyDefault);
    cudaMemcpy(bottom[2]->mutable_gpu_diff(),temp_bp1.gpu_diff(), sizeof(Dtype)*count, cudaMemcpyDefault);
  }

  //caffe_gpu_scale(count, -lambda[0]*tau, bottom[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
  //caffe_gpu_scale(count, -tau[0], bottom[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
  
  //update tau
  //Dtype* tau_diff = this->blobs_[0]->mutable_cpu_diff();

  //top data is grad*tau
  //const Dtype* ksi0 = top[0]->gpu_data();
  //const Dtype* ksi1 = top[1]->gpu_data();
  //Dtype dot0, dot1; 
  //caffe_gpu_dot(count, ksi0, top0_diff, &dot0);
  //caffe_gpu_dot(count, ksi1, top1_diff, &dot1);
  ////*tau_diff = tau[0]==0?0:-(dot0+dot1)/tau[0];
  
  //if(iter_%100==1) std::cout<<"tau_diff: "<<tau_diff[0]<<std::endl;

  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(UpdateXiLayer);

}  // namespace caffe
