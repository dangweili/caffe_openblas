#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe{

template <typename TypeParam>
class MultiLabelLossLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
    
    protected:
        MultiLabelLossLayerTest()
            : blob_bottom_data_( new Blob<Dtype>(10,5,1,1)),
              blob_bottom_targets_( new Blob<Dtype>(10,5,1,1)),
              blob_top_loss_( new Blob<Dtype>()) {
            // Fill the data vector
            FillerParameter data_filler_param;
            data_filler_param.set_std(1);
            GaussianFiller<Dtype> data_filler(data_filler_param);
            data_filler.Fill(blob_bottom_data_);
            blob_bottom_vec_.push_back(blob_bottom_data_);
            // Fill the targets vector
            FillerParameter targets_filler_param;
            targets_filler_param.set_min(-1);
            targets_filler_param.set_max(1);
            UniformFiller<Dtype> targets_filler(targets_filler_param);
            targets_filler.Fill(blob_bottom_targets_);
            Dtype* y = blob_bottom_targets_->mutable_cpu_data();
            for(int i = 0; i < blob_bottom_targets_->count(); i++)
            {
                y[i] = y[i] == 0 ? 0 : (y[i] > 0 ? 1 : -1 ); 
            }
            blob_bottom_vec_.push_back(blob_bottom_targets_);
            blob_top_vec_.push_back(blob_top_loss_);

        }
        virtual ~MultiLabelLossLayerTest()
        {
            delete blob_bottom_data_;
            delete blob_bottom_targets_;
            delete blob_top_loss_;
        }
        Dtype MultiLabelLossLayerReference(const int count, const int num, 
                const Dtype* input, const Dtype* target) {
            Dtype loss = 0;
            for (int i = 0; i < count; ++i){
                const Dtype prediction = 1 / (1 + exp(-input[i]));
                EXPECT_LE(prediction, 1);
                EXPECT_GE(prediction, 0);
                EXPECT_LE(target[i], 1);
                EXPECT_GE(target[i], -1);
                if( target[i] != 0)
                {
                    Dtype temp = target[i] == -1 ? 0 : 1;
                    loss -= temp * log(prediction + (temp == Dtype(0))) * exp(0.5);
                    loss -= (1 - temp) * log(1 - prediction + (temp == Dtype(1))) * exp(0.5);
                }
            }
            return loss / num;
        }
        void TestForward() {
            LayerParameter layer_param;
            const Dtype kLossWeight = 3.7;
            layer_param.add_loss_weight(kLossWeight);
            FillerParameter data_filler_param;
            data_filler_param.set_std(1);
            GaussianFiller<Dtype> data_filler(data_filler_param);
            FillerParameter targets_filler_param;
            targets_filler_param.set_min(-1.0);
            targets_filler_param.set_max(1.0);
            UniformFiller<Dtype> targets_filler(targets_filler_param);
            Dtype eps = 2e-2;
            for (int i = 0; i < 100; ++i) {
                data_filler.Fill(this->blob_bottom_data_);
                targets_filler.Fill(this->blob_bottom_targets_);
                Dtype* y = this->blob_bottom_targets_->mutable_cpu_data();
                for(int j = 0; j < blob_bottom_targets_->count(); j++)
                {
                    y[j] = y[j] == 0 ? 0 : (y[j] > 0 ? 1 : -1 ); 
                }
                MultiLabelLossLayer<Dtype> layer(layer_param);
                layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
                Dtype layer_loss =
                    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
                const int count = this->blob_bottom_data_->count();
                const int num = this->blob_bottom_data_->num();
                const Dtype* blob_bottom_data = this->blob_bottom_data_->cpu_data();
                const Dtype* blob_bottom_targets =
                            this->blob_bottom_targets_->cpu_data();
                Dtype reference_loss = kLossWeight * MultiLabelLossLayerReference(
                            count, num, blob_bottom_data, blob_bottom_targets);
                EXPECT_NEAR(reference_loss, layer_loss, eps) << "debug: trial #" << i;
            }
        }
    protected:
        Blob<Dtype>* const blob_bottom_data_;
        Blob<Dtype>* const blob_bottom_targets_;
        Blob<Dtype>* const blob_top_loss_;
        vector<Blob<Dtype>*> blob_bottom_vec_;
        vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MultiLabelLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(MultiLabelLossLayerTest, TestMultiLabelLoss) {
  this->TestForward();
}

TYPED_TEST(MultiLabelLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  MultiLabelLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

} // end namespace
