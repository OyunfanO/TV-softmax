#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>

#include "caffe/layers/projection_layer.hpp"

namespace caffe {

template <typename Dtype>
void ProjectionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  top[1]->ReshapeLike(*bottom[0]);
  //std::cout<<"top ok"<<std::endl;
  vector<int> bshape(bottom[0]->shape().begin(),bottom[0]->shape().end());
  //std::cout<<"shape ok"<<std::endl;
  norm_xi->Reshape(bshape);
  xi1_s->Reshape(bshape);
  xi2_s->Reshape(bshape);
}

template <typename Dtype>
void ProjectionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* xi1 = bottom[0]->cpu_data();
  const Dtype* xi2 = bottom[1]->cpu_data();
  Dtype* eta1 = top[0]->mutable_cpu_data();
  Dtype* eta2 = top[1]->mutable_cpu_data();
  //Dtype* temp = temp_->mutable_cpu_data();

  const int count = bottom[0]->count();

  //x1_s = xi1^2, x2_s = xi2^2
  caffe_sqr(count, xi1, xi1_s->mutable_cpu_data());
  caffe_sqr(count, xi2, xi2_s->mutable_cpu_data());
  //norm_xi = xi1^2+xi2^2
  caffe_add(count, xi1_s->cpu_data(),xi2_s->cpu_data(),norm_xi->mutable_cpu_data());
  //norm_xi = sqrt(xi1^2+xi2^2)
  caffe_powx(count,norm_xi->cpu_data(),Dtype(0.5),norm_xi->mutable_cpu_data());

  const Dtype* norm = norm_xi->cpu_data();
  for (int i = 0; i < count; ++i) {
      if(norm[i]>1)
      {
        eta1[i] = xi1[i]/norm[i];
        eta2[i] = xi2[i]/norm[i];
      }
      else
      {
        eta1[i] = xi1[i];
        eta2[i] = xi2[i];
      }
  }
}

template <typename Dtype>
void ProjectionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* xi1 = bottom[0]->cpu_data();
  const Dtype* xi2 = bottom[1]->cpu_data();
  const Dtype* xi1s = xi1_s->cpu_data();
  const Dtype* xi2s = xi2_s->cpu_data();
  const Dtype* norm = norm_xi->cpu_data();
  const Dtype* eta1_diff = top[0]->cpu_diff();
  const Dtype* eta2_diff = top[1]->cpu_diff();


  Dtype* xi1_diff = bottom[0]->mutable_cpu_diff();
  Dtype* xi2_diff = bottom[1]->mutable_cpu_diff();

  const int count = bottom[0]->count();
  for(int i=0;i<count;i++)
  {
    if(norm[i]>1)
    {
      xi1_diff[i] = -eta2_diff[i]*xi1[i]*xi2[i]*pow(norm[i],-3.0) + eta1_diff[i]*(1/norm[i]-xi1s[i]*pow(norm[i],-3.0));
      xi2_diff[i] = -eta1_diff[i]*xi1[i]*xi2[i]*pow(norm[i],-3.0) + eta2_diff[i]*(1/norm[i]-xi2s[i]*pow(norm[i],-3.0));
    }
    else
    {
      xi1_diff[i] = eta1_diff[i];
      xi2_diff[i] = eta2_diff[i];
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ProjectionLayer);
#endif

INSTANTIATE_CLASS(ProjectionLayer);
REGISTER_LAYER_CLASS(Projection);
}  // namespace caffe
