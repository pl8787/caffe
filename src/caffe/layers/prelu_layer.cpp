#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void PReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  mode_ = this->layer_param_.prelu_param().mode();
  CHECK(mode_==PReLUParameter_ReLUMode_CWISE ||
        mode_==PReLUParameter_ReLUMode_SHARE)
        << "Parameter ReLU only support CWISE or SHARE.";
  this->blobs_.resize(1);
  
  weight_diff_mat.ReshapeLike(*bottom[0]);
  if (mode_ == PReLUParameter_ReLUMode_SHARE) {
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));
    ones.ReshapeLike(weight_diff_mat);
  } else if (mode_ == PReLUParameter_ReLUMode_CWISE) {
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, bottom[0]->channels()));
	ones.Reshape(1, 1, 1, bottom[0]->height() * bottom[0]->width());
	temp_weight_diff.Reshape(1, 1, 1, bottom[0]->channels());
  }
  
  Dtype* ones_data = ones.mutable_gpu_data();
  caffe_gpu_set(ones.count(), (Dtype)1., ones_data);

  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.prelu_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());
 

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true); 
}

template <typename Dtype>
void PReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  
}

template <typename Dtype>
void PReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int channel_size = bottom[0]->height() * bottom[0]->width();
  const int num_size = bottom[0]->channels() * channel_size;
  const Dtype* negative_slope = this->blobs_[0]->cpu_data();
  
  if (mode_ == PReLUParameter_ReLUMode_SHARE) {
    Dtype negative_slope_share = negative_slope[0];
    for (int i = 0; i < count; ++i) {
      top_data[i] = std::max(bottom_data[i], Dtype(0))
          + negative_slope_share * std::min(bottom_data[i], Dtype(0));
    }
  } else if (mode_ == PReLUParameter_ReLUMode_CWISE) {
    for (int i = 0; i < count; ++i) {
      Dtype negative_slope_channel = negative_slope[(i % num_size) / channel_size];
      top_data[i] = std::max(bottom_data[i], Dtype(0))
          + negative_slope_channel * std::min(bottom_data[i], Dtype(0));
    }
  }
}

template <typename Dtype>
void PReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  const int count = (*bottom)[0]->count();
  const int channel_size = (*bottom)[0]->height() * (*bottom)[0]->width();
  const int num_size = (*bottom)[0]->channels() * channel_size;
  const Dtype* negative_slope = this->blobs_[0]->cpu_data();

  if (this->param_propagate_down_[0]) {
    caffe_set(this->blobs_[0]->count(), (Dtype)0, weight_diff);
    if (mode_ == PReLUParameter_ReLUMode_SHARE) {
      for (int i = 0; i < count; ++i) {
        weight_diff[0] += top_diff[i]*(bottom_data[i] <= 0)*bottom_data[i];
      }
    } else if (mode_ == PReLUParameter_ReLUMode_CWISE) {
      for (int i = 0; i < count; ++i) {
        weight_diff[(i % num_size) / channel_size] += top_diff[i]*(bottom_data[i] <= 0)*bottom_data[i];
      }
    }
  }
  
  if (propagate_down[0]) {
    if (mode_ == PReLUParameter_ReLUMode_SHARE) {
      Dtype negative_slope_share = negative_slope[0];
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
            + negative_slope_share * (bottom_data[i] <= 0));
      }
    } else if (mode_ == PReLUParameter_ReLUMode_CWISE) {
      for (int i = 0; i < count; ++i) {
        Dtype negative_slope_channel = negative_slope[(i % num_size) / channel_size];
        bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
            + negative_slope_channel * (bottom_data[i] <= 0));
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(PReLULayer);
#endif

INSTANTIATE_CLASS(PReLULayer);


}  // namespace caffe
