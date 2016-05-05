#ifndef CAFFE_VIDEO_DATA_LAYER_HPP_
#define CAFFE_VIDEO_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/video_data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class VideoDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit VideoDataLayer(const LayerParameter& param);
  virtual ~VideoDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "VideoData"; }
  virtual inline int_tp ExactNumBottomBlobs() const { return 0; }
  virtual inline int_tp MinTopBlobs() const { return 1; }
  virtual inline int_tp MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);
  VideoDataReader reader_;
};

}  // namespace caffe

#endif  // CAFFE_VIDEO_DATA_LAYER_HPP_
