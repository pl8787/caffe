#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PReLUForwardShare(const int n, const Dtype* in, Dtype* out,
    const Dtype* negative_slope_share) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope_share[0];
  }
}

template <typename Dtype>
__global__ void PReLUForwardChannel(const int n, const Dtype* in, Dtype* out,
    const Dtype* negative_slope, const int channel_size, const int num_size) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope[(index % num_size) / channel_size];
  }
}

template <typename Dtype>
void PReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int channel_size = bottom[0]->height() * bottom[0]->width();
  const int num_size = bottom[0]->channels() * channel_size;
  const Dtype* negative_slope = this->blobs_[0]->gpu_data();
  
  if (mode_ == PReLUParameter_ReLUMode_SHARE) {
    PReLUForwardShare<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, top_data, negative_slope);
  } else if (mode_ == PReLUParameter_ReLUMode_CWISE) {
    PReLUForwardChannel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, top_data, negative_slope, channel_size, num_size);
  }
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void PReLUBackwardErrorShare(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, const Dtype* negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope[0]);
  }
}

template <typename Dtype>
__global__ void PReLUBackwardErrorChannel(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, const Dtype* negative_slope,
    const int channel_size, const int num_size) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope[(index % num_size) / channel_size]);
  }
}

template <typename Dtype>
__global__ void PReLUBackwardWeight(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * (in_data[index] <= 0) * in_data[index];
  }
}

template <typename Dtype>
void PReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  const int count = (*bottom)[0]->count();
  const int channel_size = (*bottom)[0]->height() * (*bottom)[0]->width();
  const int num_size = (*bottom)[0]->channels() * channel_size;
  const Dtype* negative_slope = this->blobs_[0]->gpu_data();

#if 1
  const Dtype* bottom_data_cpu = (*bottom)[0]->cpu_data();
  const Dtype* top_diff_cpu = top[0]->cpu_diff();
  Dtype* weight_diff_cpu = this->blobs_[0]->mutable_cpu_diff();

  if (this->param_propagate_down_[0]) {
    caffe_set(this->blobs_[0]->count(), (Dtype)0, weight_diff_cpu);
    if (mode_ == PReLUParameter_ReLUMode_SHARE) {
      for (int i = 0; i < count; ++i) {
        weight_diff_cpu[0] += top_diff_cpu[i]*(bottom_data_cpu[i] <= 0)*bottom_data_cpu[i];
      }
    } else if (mode_ == PReLUParameter_ReLUMode_CWISE) {
      for (int i = 0; i < top[0]->num(); ++i) {
		for (int j = 0; j < top[0]->channels(); ++j) {
		  for (int k = 0; k < channel_size; ++k) {
            weight_diff_cpu[j] += top_diff_cpu[k]*(bottom_data_cpu[k] <= 0)*bottom_data_cpu[k];
		  }
		  top_diff_cpu += channel_size;
		  bottom_data_cpu += channel_size;
		}
      }
    }
  }
#else
  if (this->param_propagate_down_[0]) {
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    Dtype* weight_diff_mat_data = weight_diff_mat.mutable_gpu_data();

    caffe_gpu_set(this->blobs_[0]->count(), (Dtype)0, weight_diff);
    PReLUBackwardWeight<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
             count, top_diff, bottom_data, weight_diff_mat_data);
    if (mode_ == PReLUParameter_ReLUMode_SHARE) {
	  Dtype* ones_data = ones.mutable_gpu_data();
      caffe_gpu_dot_gpu(count, weight_diff_mat_data, ones_data, weight_diff);
    } else if (mode_ == PReLUParameter_ReLUMode_CWISE) {
	  Dtype* ones_data = ones.mutable_gpu_data();
  	  Dtype* diff_temp = temp_weight_diff.mutable_gpu_data();

      for (int i = 0; i < top[0]->num(); ++i) {
        for (int j = 0; j < top[0]->channels(); ++j) {
          caffe_gpu_dot_gpu(channel_size, weight_diff_mat_data + i*num_size + j*channel_size, ones_data, diff_temp + j);
	    }
		caffe_gpu_add(top[0]->channels(), weight_diff, diff_temp, weight_diff);
      }
    } 
  }
#endif
  if (propagate_down[0]) {
    if (mode_ == PReLUParameter_ReLUMode_SHARE) {
      PReLUBackwardErrorShare<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, bottom_data, bottom_diff, negative_slope);
    } else if (mode_ == PReLUParameter_ReLUMode_CWISE) {
      PReLUBackwardErrorChannel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, bottom_data, bottom_diff, negative_slope, channel_size, num_size);
    }
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_CLASS(PReLULayer);


}  // namespace caffe
