#ifndef CAFFE_GRADIENT_LAYER_HPP_
#define CAFFE_GRADIENT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class UpdateXiLayer : public  Layer<Dtype> {
 public:
  explicit UpdateXiLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual inline const char* type() const { return "Gradient"; }

 protected:

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int channels_;
  int height_, width_;
  float tau_,lambda_;
  
 private:
  Blob<Dtype> temp_bp0, temp_bp1;
  int iter_;
  //float tau;
};

}  // namespace caffe

#endif  // CAFFE_RELU_LAYER_HPP_
