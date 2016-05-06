#ifndef CAFFE_VIDEO_DATA_READER_HPP_
#define CAFFE_VIDEO_DATA_READER_HPP_

#include <map>
#include <string>
#include <vector>
#include <string>
#include <tuple>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * @brief Reads data from a source to queues available to data layers.
 * A single reading thread is created per source, even if multiple solvers
 * are running in parallel, e.g. for multi-GPU training. This makes sure
 * databases are read sequentially, and that each solver accesses a different
 * subset of the database. Data is distributed to solvers in a round-robin
 * way to keep parallel training deterministic.
 */
class VideoDataReader {
 public:
  explicit VideoDataReader(const LayerParameter& param);
  ~VideoDataReader();

  inline BlockingQueue<DatumList*>& free() const {
    return queue_pair_->free_;
  }
  inline BlockingQueue<DatumList*>& full() const {
    return queue_pair_->full_;
  }

 protected:
  // Queue pairs are shared between a body and its readers
  class QueuePair {
   public:
    explicit QueuePair(int_tp size);
    ~QueuePair();

    BlockingQueue<DatumList*> free_;
    BlockingQueue<DatumList*> full_;

  DISABLE_COPY_AND_ASSIGN(QueuePair);
  };

  // A single body is created per source
  class Body : public InternalThread {
   public:
    explicit Body(const LayerParameter& param, device* device_context);
    virtual ~Body();

   protected:
    void InternalThreadEntry();
    void read_one(db::Transaction* txn, db::Cursor* cur, QueuePair* qp);

    // build index from label file
    void build_index();
    // Random sample a 3d block
    void random_sample(db::Transaction* txn, DatumList* dl, int_tp* label);
    // Uniformly get 3d block
    void uniform_scan(db::Cursor* cur, DatumList* dl);
    bool fetch_one_sample(db::Transaction* txn, DatumList* dl,
                          std::string video_id, int_tp frame_begin);
    // Calculate chunk_id
    inline int_tp chunk_id(int_tp frame_id) {
      return frame_id / param_.video_param().chunk_size();
    }

    const LayerParameter param_;
    BlockingQueue<shared_ptr<QueuePair> > new_queue_pairs_;
    // inverse index: map of class_id : vector<tuple>
    // term: video_name, start_frame_id, duration, label_id
    map<int_tp, vector<std::tuple<std::string, int_tp, int_tp, int_tp> > > label_index;

    bool has_label_file;
    int_tp label_count;
    // Following members are used for uniformly scan only
    bool current_dl_finished = true;
    int_tp current_frame_idx = 0;
    DatumList current_dl;

    friend class VideoDataReader;

  DISABLE_COPY_AND_ASSIGN(Body);
  };

  // A source is uniquely identified by its layer name + path, in case
  // the same database is read from two different locations in the net.
  static inline string source_key(const LayerParameter& param) {
    return param.name() + ":" + param.data_param().source();
  }

  const shared_ptr<QueuePair> queue_pair_;
  shared_ptr<Body> body_;
  device* device_;

  static map<const string, boost::weak_ptr<VideoDataReader::Body> > bodies_;

DISABLE_COPY_AND_ASSIGN(VideoDataReader);
};

}  // namespace caffe

#endif  // CAFFE_VIDEO_DATA_READER_HPP_
