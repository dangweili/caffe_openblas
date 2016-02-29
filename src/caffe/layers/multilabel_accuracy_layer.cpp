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
    top[0]->Reshape(1, 7, 1, 1);
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
  //Dtype true_positive = 0;
  //Dtype false_positive = 0;
  //Dtype true_negative = 0;
  //Dtype false_negative = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  // Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int count = bottom[0]->count();
  int batchsize = bottom[0]->count(0, 1);
  int labelsize = count/batchsize;
  
  vector<float> label_accuracy_pos(labelsize);
  vector<float> label_accuracy_neg(labelsize);
  vector<float> label_accuracy_all(labelsize);
  vector<int> p_label(count);
  vector<int> g_label(count);
  for(int i=0; i<count; i++)
  {
    p_label[i] = bottom_data[i] >=0 ? 1 : -1;
    g_label[i] = bottom_label[i];
  }
  float instance_accuracy = 0;
  float instance_precision = 0;
  float instance_recall = 0;
  float instance_F1 = 0;
  int effective_cnt = 0;
  vector<int> label_pos_cnt(labelsize);
  vector<int> label_neg_cnt(labelsize);
  // compute each example's accuracy and give a summary
  for (int exa = 0; exa < batchsize; exa++ )
  {
    float intersection_count = 0;
    float union_count = 0;
    float gt_count = 0;
    float pt_count = 0;
    for (int la = 0; la < labelsize; la++ )
    {
        int idx = exa*labelsize+la;
        if(g_label[idx] == 1) gt_count += 1;
        if(p_label[idx] == 1) pt_count += 1;
        if(g_label[idx]==1 && p_label[idx]==1) intersection_count += 1;
        if(g_label[idx]==1 || p_label[idx]==1) union_count += 1;
        // label accuracy
        if (g_label[idx] != 0)
        {
            if (g_label[idx] > 0) {
                label_pos_cnt[la] += 1;
                label_accuracy_pos[la] += p_label[idx] > 0 ? 1 : 0;
            }
            else {
                label_neg_cnt[la] += 1;
                label_accuracy_neg[la] += p_label[idx] < 0 ? 1 : 0;
            }
        }
    }
    // instance 
    if (union_count > 0 && pt_count > 0 && gt_count > 0)
    {
        effective_cnt += 1;
        instance_accuracy += intersection_count*1.0/union_count;
        instance_precision += intersection_count*1.0/pt_count;
        instance_recall += intersection_count*1.0/gt_count;
    }
  }
  // instance
  instance_accuracy = effective_cnt > 0 ? instance_accuracy/effective_cnt : 0;
  instance_precision = effective_cnt > 0 ? instance_precision/effective_cnt : 0;
  instance_recall = effective_cnt > 0 ? instance_recall/effective_cnt : 0;
  instance_F1 = effective_cnt > 0 ? 2 * instance_precision * instance_recall / (instance_precision + instance_recall) : 0;
  float label_accuracy_pos_mean = 0;
  float label_accuracy_neg_mean = 0;
  float label_accuracy_all_mean = 0;
  for (int la = 0; la < labelsize; la ++ )
  {
    label_accuracy_pos[la] /= label_pos_cnt[la] > 0 ? label_pos_cnt[la] : 1;
    label_accuracy_neg[la] /= label_neg_cnt[la] > 0 ? label_neg_cnt[la] : 1;
    label_accuracy_all[la] = ( label_accuracy_pos[la] + label_accuracy_neg[la] )/2.0;
    label_accuracy_pos_mean += label_accuracy_pos[la];
    label_accuracy_neg_mean += label_accuracy_neg[la];
    label_accuracy_all_mean += label_accuracy_all[la];
  }
  label_accuracy_pos_mean /= labelsize;
  label_accuracy_neg_mean /= labelsize;
  label_accuracy_all_mean /= labelsize;
  top[0]->mutable_cpu_data()[0] = label_accuracy_pos_mean;
  top[0]->mutable_cpu_data()[1] = label_accuracy_neg_mean;
  top[0]->mutable_cpu_data()[2] = label_accuracy_all_mean;
  top[0]->mutable_cpu_data()[3] = instance_accuracy;
  top[0]->mutable_cpu_data()[4] = instance_precision;
  top[0]->mutable_cpu_data()[5] = instance_recall;
  top[0]->mutable_cpu_data()[6] = instance_F1;
  
/*
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
*/
  // MultiLabelAccuracy should not be used as a loss function.
  // return Dtype(0);

}

INSTANTIATE_CLASS(MultiLabelAccuracyLayer);
REGISTER_LAYER_CLASS(MultiLabelAccuracy);

}  // namespace caffe
