#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/video_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {


template<typename Dtype>
VideoDataLayer<Dtype>::VideoDataLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param), reader_(param) {
}

template<typename Dtype>
VideoDataLayer<Dtype>::~VideoDataLayer() {
  this->StopInternalThread();
}

template<typename Dtype>
void VideoDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  const int_tp batch_size = this->layer_param_.video_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  DatumList& datum_list = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  CHECK(datum_list.datums_size() > 0);
  

  Datum *datum = datum_list.mutable_datums(0);
  // ! Temporary hack for a dataset where encoded was not set correctly.
  datum->set_encoded(true);
  vector<int_tp> frame_shape = this->data_transformer_->InferBlobShape(*datum);
  
  CHECK(frame_shape.size() == 4);

  int_tp temporal_size = this->layer_param_.video_param().temporal_size();

  // Top shape is N x C x T x H x W.
  vector<int_tp> top_shape = frame_shape;
  vector<int_tp>::iterator it = top_shape.begin();
  top_shape.insert(it+2, temporal_size);

  this->transformed_data_.Reshape(frame_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;

  DLOG(INFO) << top_shape[0] << top_shape[1]
	     << top_shape[2] << top_shape[3]
	     << top_shape[4];
  top[0]->Reshape(top_shape);
  for (int_tp i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO)<< "output data size: " << top[0]->shape_string();
  // label
  if (this->output_labels_) {
    vector<int_tp> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int_tp i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

template<typename Dtype>
void _copy_frame(int_tp item_id, int_tp frame_id, const Blob<Dtype> &frame,
		 Blob<Dtype> *sequence) {
  CHECK(item_id < sequence->shape(0));
  CHECK(frame_id < sequence->shape(2));
  vector<int_tp> indices {item_id, 0, frame_id};
  int_tp channels = frame.shape(1);
  for (int_tp ch = 0; ch < channels; ++ch) {
    indices[1] = ch;
    int_tp offset_frame = frame.offset(0, ch);
    int_tp N = frame.count(2);
    int_tp offset_seq = sequence->offset(indices);
    caffe_cpu_copy(N, frame.cpu_data() + offset_frame,
	       sequence->mutable_cpu_data() + offset_seq);
  }
}
// This function is called on prefetch thread
template<typename Dtype>
void VideoDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int_tp batch_size = this->layer_param_.video_param().batch_size();
  DatumList& datum_list = *(reader_.full().peek());

  CHECK(datum_list.datums_size() > 0);

  Datum *datum = datum_list.mutable_datums(0);
  // ! Temporary hack for a dataset where encoded was not set correctly.
  datum->set_encoded(true);
  vector<int_tp> frame_shape = this->data_transformer_->InferBlobShape(*datum);
  CHECK(frame_shape.size() == 4);

  int_tp temporal_size = this->layer_param_.video_param().temporal_size();

  // Top shape is N x C x T x H x W.
  vector<int_tp> top_shape = frame_shape;
  vector<int_tp>::iterator it = top_shape.begin();
  top_shape.insert(it+2, temporal_size);
  
  this->transformed_data_.Reshape(frame_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int_tp item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datumList
    DatumList& datum_list = *(reader_.full().pop());
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    for (int_tp frame_id = 0; frame_id < temporal_size; ++frame_id) {
      Datum *datum = datum_list.mutable_datums(frame_id);
      // ! Temporary hack for a dataset where encoded was not set correctly.
      datum->set_encoded(true);

      DLOG(INFO) << "Transforming datum " << frame_id << ": shape " << datum->channels()
		 <<" " << datum->width() << " " << datum->height()
		 <<" encoded: " << datum->encoded();
      this->data_transformer_->Transform(*datum, &(this->transformed_data_));
      // Copy data.
      _copy_frame(item_id, frame_id, this->transformed_data_, &batch->data_);
    }

    // Copy label.
    if (this->output_labels_) {
      top_label[item_id] = datum_list.datums(0).label();
    }
    trans_time += timer.MicroSeconds();
    reader_.free().push(const_cast<DatumList*>(&datum_list));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO)<< "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO)<< "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO)<< "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(VideoDataLayer);
REGISTER_LAYER_CLASS(VideoData);

}  // namespace caffe
