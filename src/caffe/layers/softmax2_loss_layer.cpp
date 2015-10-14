#include <algorithm>
#include <cfloat>
#include <vector>
#include <fstream>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void Softmax2WithLossLayer<Dtype>::ReadTopLevel() {
  std::ifstream fin(top_level_.c_str());
  for (int i = 0; i < top_cate_; ++i) {
    fin >> top_dict_.mutable_cpu_data()[i];
  }
  fin.close();
  LOG(INFO) << "Read Top Level Dict Over.";
}

template <typename Dtype>
void Softmax2WithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Softmax2LossParameter softmax2loss_param = this->layer_param_.softmax2loss_param();
  top_level_ = softmax2loss_param.top_level();
  top_cate_ = softmax2loss_param.top_cate();
  lambda_ = softmax2loss_param.lambda();

  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);

  top_dict_.Reshape(1, 1, 1, prob_.channels());
  ReadTopLevel();
}

template <typename Dtype>
void Softmax2WithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  (*top)[0]->Reshape(1, 1, 1, 2);
  softmax_layer_->Reshape(softmax_bottom_vec_, &softmax_top_vec_);
  top_prob_.Reshape(prob_.num(), top_cate_, prob_.height(), prob_.width());
  if (top->size() >= 2) {
    // softmax output
    (*top)[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void Softmax2WithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  Dtype* top_prob_data = top_prob_.mutable_cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  
  int num = prob_.num();
  int dim = prob_.count() / num;
  int top_dim = top_prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();
  Dtype loss = 0;
  Dtype top_loss = 0;

  caffe_set(top_prob_.count(), (Dtype)0, top_prob_.mutable_cpu_data());
  for (int i = 0; i < num; ++i) {
    for (int k = 0; k < prob_.channels(); ++k) {
      for (int j = 0; j < spatial_dim; ++j) {
		int label_v = k;
		int top_label_v = top_dict_.cpu_data()[label_v];
        top_prob_data[i * top_dim + top_label_v * spatial_dim + j] +=
		  prob_data[i * dim + label_v * spatial_dim + j];
	  }
	}	
  }

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
	  int label_v = static_cast<int>(label[i * spatial_dim + j]);
	  int top_label_v = top_dict_.cpu_data()[label_v];
      loss -= log(std::max(prob_data[i * dim + label_v * spatial_dim + j],
                           Dtype(FLT_MIN)));
	  top_loss -= log(std::max(top_prob_data[i * top_dim + top_label_v * spatial_dim + j], 
			               Dtype(FLT_MIN)));
    }
  }
  (*top)[0]->mutable_cpu_data()[0] = loss / num / spatial_dim;
  (*top)[0]->mutable_cpu_data()[1] = top_loss / num / spatial_dim;
  if (top->size() == 2) {
    (*top)[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void Softmax2WithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
	const Dtype* top_prob_data = top_prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = (*bottom)[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    int top_dim = top_prob_.count() / num;
    int spatial_dim = prob_.height() * prob_.width();
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
	    int label_v = static_cast<int>(label[i * spatial_dim + j]);
	    int top_label_v = top_dict_.cpu_data()[label_v];
        bottom_diff[i * dim + label_v * spatial_dim + j] -= lambda_;
		for (int k = 0; k < prob_.channels(); ++k) {
          if (top_label_v == top_dict_.cpu_data()[k]) {
            bottom_diff[i * dim + k * spatial_dim +j] -= (1 - lambda_) * prob_data[i * dim + k * spatial_dim + j] / top_prob_data[i * top_dim + top_label_v * spatial_dim + j]; 
		  }
		}
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(prob_.count(), loss_weight / num / spatial_dim, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(Softmax2WithLossLayer);
#endif

INSTANTIATE_CLASS(Softmax2WithLossLayer);


}  // namespace caffe
