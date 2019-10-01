#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>

#include "caffe/layers/dispatch_layer.hpp"

namespace caffe {


template <typename Dtype>
void DispatchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  //DispatchParameter dispatch_param = this->layer_param_.dispatch_param();
  //alpha = dispatch_param.alpha();//coefficiency of lambda

}

template <typename Dtype>
void DispatchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->ReshapeLike(*bottom[0]);
  }
  //top[0]->ReshapeLike(*bottom[0]);
  //top[1]->ReshapeLike(*bottom[0]);
  //vector<int> lambda_shape(0); //lambda is a scalar; 0 axes.
  //top[2]->Reshape(lambda_shape);
  //std::cout<<top[2]->count()<<std::endl;
}

template <typename Dtype>
void DispatchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int count = bottom[0]->count();
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    Dtype* top_data = top[top_id]->mutable_cpu_data();
    memcpy(top_data, bottom_data, sizeof(Dtype)*count);
  }  
  //memcpy(top1_data, bottom_data, sizeof(Dtype)*count);
  //top2_data[0] = caffe_cpu_asum(count,bottom_data);
  //top2_data[0] = alpha*top2_data[0]/count;

}

template <typename Dtype>
void DispatchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top0_diff = top[0]->cpu_diff();
  const Dtype* top1_diff = top[1]->cpu_diff();

  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  const int count = bottom[0]->count();
  caffe_add(count, top0_diff, top1_diff, bottom_diff);
  for (int top_id = 2; top_id < top.size(); ++top_id) {
    const Dtype* top_data = top[top_id]->cpu_diff();
    caffe_add(count, top_data, bottom_diff, bottom_diff);
  }  

}


#ifdef CPU_ONLY
STUB_GPU(DispatchLayer);
#endif

INSTANTIATE_CLASS(DispatchLayer);
REGISTER_LAYER_CLASS(Dispatch);

}  // namespace caffe
