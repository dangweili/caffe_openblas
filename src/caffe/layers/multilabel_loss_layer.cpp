#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // father initial
  LossLayer<Dtype>::LayerSetUp( bottom, top );
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "MULTI_LABEL_LOSS layer inputs must have the same count.";
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Reshape(
const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // father reshape
  LossLayer<Dtype>::Reshape( bottom, top );

  if (top.size() >= 1) {
   // sigmoid cross entropy loss (averaged across batch)
    top[0]->Reshape(1, 1, 1, 1);
  }
/*  if (top.size() == 2) {
   // softmax output
    top[1]->ReshapeLike(*sigmoid_output_.get());
    top[1]->ShareData(*sigmoid_output_.get());
  }
*/
}

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;

  int dim = count/num;
/*  Dtype weight[] = {0.497, 0.3288, 0.1023, 0.0617, 0.1966, 0.1994, 0.8608, 0.8530, 0.1375, 0.1339, 
                   0.1016, 0.0692, 0.3061, 0.2962, 0.0402, 0.2375, 0.5485, 0.2957, 0.0839, 0.7494,
                   0.2759, 0.0266, 0.0765, 0.0204, 0.3633, 0.0347, 0.1418, 0.0455, 0.2161, 0.0172, 
                   0.0291, 0.5151, 0.0842, 0.4556, 0.0118};
*/
//  Dtype weight[] = {0.4865, 0.0789, 0.6699, 0.1386, 0.1123, 0.2427, 0.7945, 0.1314, 0.1352, 0.1820, 0.0539};
  int weight_size = this->layer_param_.multilabel_loss_param().weight_size();
  Dtype* weight = NULL;
  if ( weight_size > 0 )
  {
     CHECK_EQ(weight_size, dim) <<
         "weight must has the same size with channels.";
     weight = new Dtype[dim];
     for(int i = 0; i < dim; i++) 
        weight[i] = this->layer_param_.multilabel_loss_param().weight(i); 
  }
  else
  {  
     weight = new Dtype[dim];
     for(int i = 0; i < dim; i++)
        weight[i] = 0.5;
  }
  for (int i = 0; i < count; ++i) {
    if (target[i] != 0) {
    // Update the loss only if target[i] is not 0
    //  loss -= input_data[i] * ((target[i] > 0) - (input_data[i] >= 0)) -
    //      log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
        Dtype temp = input_data[i] * ((target[i] > 0) - (input_data[i] >= 0)) -
            log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
        if (target[i] > 0)
        {
             loss -= temp*exp(1 - weight[i%dim]);
            //    loss -= temp;
        }
        else 
        {
             loss -= temp*exp(weight[i%dim]) ; 
            //    loss -= temp;
        }
    }
  }

/*  for (int i = 0; i < count; ++i) {
    if (target[i] != 0) {
    // Update the loss only if target[i] is not 0
      loss -= input_data[i] * ((target[i] > 0) - (input_data[i] >= 0)) -
          log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    }
  }
*/
  // average the loss
  if (top.size() >= 1) {
    top[0]->mutable_cpu_data()[0] = loss / num;
  }
  delete [] weight;
}

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    
    int dim = count/num;
/*    Dtype weight[] = {0.497, 0.3288, 0.1023, 0.0617, 0.1966, 0.1994, 0.8608, 0.8530, 0.1375, 0.1339, 
                   0.1016, 0.0692, 0.3061, 0.2962, 0.0402, 0.2375, 0.5485, 0.2957, 0.0839, 0.7494,
                   0.2759, 0.0266, 0.0765, 0.0204, 0.3633, 0.0347, 0.1418, 0.0455, 0.2161, 0.0172, 
                   0.0291, 0.5151, 0.0842, 0.4556, 0.0118};
*/
//    Dtype weight[] = {0.4865, 0.0789, 0.6699, 0.1386, 0.1123, 0.2427, 0.7945, 0.1314, 0.1352, 0.1820, 0.0539};

    int weight_size = this->layer_param_.multilabel_loss_param().weight_size();
    Dtype* weight = NULL;
    if ( weight_size > 0 )
    {
         CHECK_EQ(weight_size, dim) <<
             "weight must has the same size with channels.";
        weight = new Dtype[dim];
        for(int i = 0; i < dim; i++) 
            weight[i] = this->layer_param_.multilabel_loss_param().weight(i); 
    }
    else
    {  
        weight = new Dtype[dim];
        for(int i = 0; i < dim; i++)
            weight[i] = 0.5;
    }

    for(int i=0; i<count; i++)
    {
        if(target[i] != 0)
        {
            if( target[i] > 0)
            {
                 bottom_diff[i] = (sigmoid_output_data[i] - 1)*exp(1 - weight[i%dim]);
                // bottom_diff[i] = sigmoid_output_data[i] - 1;
            }
            else
            {
                 bottom_diff[i] = sigmoid_output_data[i] * exp(weight[i%dim]);
                // bottom_diff[i] = sigmoid_output_data[i];
            }
        }
        else
        {
            bottom_diff[i] = 0;
        }
    }
/*    for (int i = 0; i < count; ++i) {
      if (target[i] != 0) {
        bottom_diff[i] = sigmoid_output_data[i] - (target[i] > 0);
      } else {
        bottom_diff[i] = 0;
      }
    }
*/    // Scale down gradient
    caffe_scal(count, Dtype(1) / num, bottom_diff);
    const Dtype loss_weight = top[0]->mutable_cpu_diff()[0];
    caffe_scal(count, loss_weight, bottom_diff );
    delete [] weight;
  }
}

#ifdef CPU_ONLY
STUB_GPU(MultiLabelLossLayer);
#endif

INSTANTIATE_CLASS(MultiLabelLossLayer);
REGISTER_LAYER_CLASS(MultiLabelLoss);

}  // namespace caffe
