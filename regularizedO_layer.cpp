#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/filler.hpp"
#include "caffe/layers/regularizedO_layer.hpp"

namespace caffe {

template <typename Dtype>
void RegularizedOLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //initialize tau		  
  //this->blobs_.resize(1);
  //vector<int> lambda_shape(1,1);
  //this->blobs_[0].reset(new Blob<Dtype>(lambda_shape));
  //shared_ptr<Filler<Dtype> > lambda_filler(GetFiller<Dtype>(this->layer_param_.divergence_param().lambda_filler()));
  //lambda_filler->Fill(this->blobs_[0].get());
  iter_ = 0;
  RegularizedoParameter regularizedo_param = this->layer_param_.regularizedo_param();
  lambda_ = regularizedo_param.lambda();//number of TV_bias filters
}

template <typename Dtype>
void RegularizedOLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->ReshapeLike(*bottom[0]);
  
  vector<int> div_eta_shape(1, bottom[0]->count());
  div_eta_.Reshape(div_eta_shape);
  caffe_set(div_eta_.count(), Dtype(0),div_eta_.mutable_cpu_data());
}

template <typename Dtype>
void RegularizedOLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void RegularizedOLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(RegularizedOLayer);
#endif

INSTANTIATE_CLASS(RegularizedOLayer);
REGISTER_LAYER_CLASS(RegularizedO);
}  // namespace caffe
