#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

using std::max;
namespace caffe {

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // modified by dangweili
  // as we do not need the accuracy parameters, so here we should not realized it
/*  
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
*/
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // this method should reshape top blobs as needed according to the shapes of the bottom(input) blobs
    // as well as reshapeing any other internal buffers and making any other necessary adjustments so that
    // the layer can accomodate the bottom blobs
    CHECK_EQ( bottom[0]->num(), bottom[1]->num())
        << "the data and label should have the same number of instances";
    CHECK_EQ( bottom[0]->channels(), bottom[1]->channels())
        << "the data and label should have the same number of instances";
    CHECK_EQ( bottom[0]->height(), bottom[1]->height())
        << "the data and label should have the same number of instances"; 
    CHECK_EQ( bottom[0]->width(), bottom[1]->width())
        << "the data and label should have the same number of instances";
    // Top will contain
    // top[0] = Sensitivity or Recall TP/P
    // top[1] = Specificity TN/N
    // top[2] = Harmonic Mean of Sens and Spec 2/( tp/p + tn/n)
    // top[3] = Precision TP/( TP + FP)
    // top[4] = F1 score 2 TP / (2tp + fp +fn)
    top[0]->Reshape(1, 5, 1, 1);
/*
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
*/
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype true_positive = 0;
  Dtype false_positive = 0;
  Dtype true_negative = 0;
  Dtype false_negative = 0;
  int count_pos = 0;
  int count_neg = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  // Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int count = bottom[0]->count();

  for (int ind = 0; ind < count; ++ind) {
    // Accuracy
    int label = static_cast<int>(bottom_label[ind]);
    if (label > 0) {
    // Update Positive accuracy and count
      true_positive += (bottom_data[ind] >= 0);
      false_negative += (bottom_data[ind] < 0);
      count_pos++;
    }
    if (label < 0) {
    // Update Negative accuracy and count
      true_negative += (bottom_data[ind] < 0);
      false_positive += (bottom_data[ind] >= 0);
      count_neg++;
    }
  }
  Dtype sensitivity = (count_pos > 0)? (true_positive / count_pos) : 0;
  Dtype specificity = (count_neg > 0)? (true_negative / count_neg) : 0;
  Dtype harmmean = ((count_pos + count_neg) > 0)?
    2 / (count_pos / true_positive + count_neg / true_negative) : 0;
  Dtype precission = (true_positive > 0)?
    (true_positive / (true_positive + false_positive)) : 0;
  Dtype f1_score = (true_positive > 0)?
    2 * true_positive /
    (2 * true_positive + false_positive + false_negative) : 0;

  DLOG(INFO) << "Sensitivity: " << sensitivity;
  DLOG(INFO) << "Specificity: " << specificity;
  DLOG(INFO) << "Harmonic Mean of Sens and Spec: " << harmmean;
  DLOG(INFO) << "Precission: " << precission;
  DLOG(INFO) << "F1 Score: " << f1_score;
  top[0]->mutable_cpu_data()[0] = sensitivity;
  top[0]->mutable_cpu_data()[1] = specificity;
  top[0]->mutable_cpu_data()[2] = harmmean;
  top[0]->mutable_cpu_data()[3] = precission;
  top[0]->mutable_cpu_data()[4] = f1_score;

  // MultiLabelAccuracy should not be used as a loss function.
  // return Dtype(0);

}

INSTANTIATE_CLASS(MultiLabelAccuracyLayer);
REGISTER_LAYER_CLASS(MultiLabelAccuracy);

}  // namespace caffe
