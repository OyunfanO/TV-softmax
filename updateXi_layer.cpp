#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/filler.hpp"
#include "caffe/layers/updateXi_layer.hpp"

namespace caffe {

template <typename Dtype>
void UpdateXiLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //initialize tau
  /*		  
  this->blobs_.resize(1);
  vector<int> tau_shape(1,1);
  this->blobs_[0].reset(new Blob<Dtype>(tau_shape));
  shared_ptr<Filler<Dtype> > tau_filler(GetFiller<Dtype>(this->layer_param_.gradient_param().tau_filler()));
  tau_filler->Fill(this->blobs_[0].get());
  iter_ = 0;
  */
  UpdatexiParameter updatexi_param = this->layer_param_.updatexi_param();
  tau_ = updatexi_param.tau();//step size
  lambda_ = updatexi_param.lambda();
}


template <typename Dtype>
void UpdateXiLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  
  temp_bp0.ReshapeLike(*bottom[0]);
  temp_bp1.ReshapeLike(*bottom[0]);
  caffe_set(temp_bp0.count(), Dtype(0),temp_bp0.mutable_cpu_data());
  caffe_set(temp_bp1.count(), Dtype(0),temp_bp1.mutable_cpu_data());
  for(int top_id=0;top_id<top.size();top_id++)
  {
    top[top_id]->ReshapeLike(*bottom[0]);
    //top[0]->ReshapeLike(*bottom[0]); //xi1(t) to eta1(t)
    //top[1]->ReshapeLike(*bottom[0]); //xi2
    //top[2]->ReshapeLike(*bottom[0]); //xi1(t) tp xi1(t+1)
    //top[3]->ReshapeLike(*bottom[0]); //xi2
  }
  
  /*
  vector<int> tau_multiplier_shape(1, bottom[0]->count());
  tau_multiplier_.Reshape(tau_multiplier_shape);
  caffe_set(tau_multiplier_.count(), Dtype(1),tau_multiplier_.mutable_cpu_data());
  */
}

template <typename Dtype>
void UpdateXiLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void UpdateXiLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}


#ifdef CPU_ONLY
STUB_GPU(UpdateXiLayer);
#endif

INSTANTIATE_CLASS(UpdateXiLayer);
REGISTER_LAYER_CLASS(UpdateXi);
}  // namespace caffe
