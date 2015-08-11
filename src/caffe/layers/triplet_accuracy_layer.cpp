#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void TripletAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    this->margin_ = this->layer_param_.triplet_accuracy_param().margin();
}

template <typename Dtype>
void TripletAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    diff_ap_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
    diff_an_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
    summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
    diff_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
    for (int i = 0; i < bottom[0]->channels(); i++)
        summer_vec_.mutable_cpu_data()[i] = 1;
    top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void TripletAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    int count = bottom[0]->count();
    caffe_sub(
        count, 
        bottom[0]->cpu_data(),
        bottom[1]->cpu_data(),
        diff_ap_.mutable_cpu_data() );
    caffe_sub(
        count,
        bottom[0]->cpu_data(),
        bottom[2]->cpu_data(),
        diff_an_.mutable_cpu_data() );
    Dtype sum_ap = 0;
    Dtype sum_an = 0;
    const int channels = bottom[0]->channels();
    for(int i = 0; i < bottom[0]->num(); i++)
    {
        sum_ap = caffe_cpu_dot(channels,
            diff_ap_.cpu_data() + i*channels, diff_ap_.cpu_data() + i*channels);
        sum_an = caffe_cpu_dot(channels,
            diff_an_.cpu_data() + i*channels, diff_an_.cpu_data() + i*channels);
        diff_sq_.mutable_cpu_data()[i] = margin_ + sum_ap - sum_an > 0 ? 0 : 1;
    }
    // accuracy
    Dtype accuracy = caffe_cpu_dot(bottom[0]->num(),
            diff_sq_.cpu_data(), summer_vec_.cpu_data());
    // LOG(INFO) << "Accuracy: " << accuracy;
    top[0]->mutable_cpu_data()[0] = accuracy / bottom[0]->num();
}

INSTANTIATE_CLASS(TripletAccuracyLayer);
REGISTER_LAYER_CLASS(TripletAccuracy);

}  // namespace caffe
