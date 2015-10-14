#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class PReLULayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PReLULayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1703);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~PReLULayerTest() { delete blob_bottom_; delete blob_top_; }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PReLULayerTest, TestDtypesAndDevices);

TYPED_TEST(PReLULayerTest, TestPReLUForwardShare) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_prelu_param()->set_mode(PReLUParameter_ReLUMode_SHARE);
  layer_param.mutable_prelu_param()->mutable_weight_filler()->set_type("constant");
  layer_param.mutable_prelu_param()->mutable_weight_filler()->set_value(0.01);
  PReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (top_data[i] >= 0) {
      EXPECT_FLOAT_EQ(top_data[i], bottom_data[i]);
    } else {
      EXPECT_FLOAT_EQ(top_data[i], bottom_data[i] * 0.01);
    }
  }
}

TYPED_TEST(PReLULayerTest, TestPReLUGradientShare) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_prelu_param()->set_mode(PReLUParameter_ReLUMode_SHARE);
  layer_param.mutable_prelu_param()->mutable_weight_filler()->set_type("constant");
  layer_param.mutable_prelu_param()->mutable_weight_filler()->set_value(0.01);
  PReLULayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0., 0.005);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(PReLULayerTest, TestPReLUForwardChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_prelu_param()->set_mode(PReLUParameter_ReLUMode_CWISE);
  layer_param.mutable_prelu_param()->mutable_weight_filler()->set_type("constant");
  layer_param.mutable_prelu_param()->mutable_weight_filler()->set_value(0.01);
  PReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (top_data[i] >= 0) {
      EXPECT_FLOAT_EQ(top_data[i], bottom_data[i]);
    } else {
      EXPECT_FLOAT_EQ(top_data[i], bottom_data[i] * 0.01);
    }
  }
}

TYPED_TEST(PReLULayerTest, TestPReLUGradientChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_prelu_param()->set_mode(PReLUParameter_ReLUMode_CWISE);
  layer_param.mutable_prelu_param()->mutable_weight_filler()->set_type("constant");
  layer_param.mutable_prelu_param()->mutable_weight_filler()->set_value(0.01);
  PReLULayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0., 0.005);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}


}  // namespace caffe
