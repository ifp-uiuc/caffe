#include <boost/thread.hpp>
#include <boost/random.hpp>
#include <map>
#include <string>
#include <vector>
#include <tuple>
#include <fstream> // NOLINT(readability/streams)
#include <algorithm>

#include "caffe/common.hpp"
#include "caffe/video_data_reader.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/format.hpp"

namespace caffe {

using boost::weak_ptr;

map<const string, weak_ptr<VideoDataReader::Body> > VideoDataReader::bodies_;
static boost::mutex bodies_mutex_;

VideoDataReader::VideoDataReader(const LayerParameter& param)
    : queue_pair_(new QueuePair(  //
                param.video_param().prefetch() * param.video_param().batch_size())), device_(Caffe::GetDefaultDevice()) {
  // Get or create a body
  boost::mutex::scoped_lock lock(bodies_mutex_);
  string key = source_key(param);
  weak_ptr<Body>& weak = bodies_[key];
  body_ = weak.lock();
  if (!body_) {
    body_.reset(new Body(param, device_));
    bodies_[key] = weak_ptr<Body>(body_);
  }
  body_->new_queue_pairs_.push(queue_pair_);
}

VideoDataReader::~VideoDataReader() {
  string key = source_key(body_->param_);
  body_.reset();
  boost::mutex::scoped_lock lock(bodies_mutex_);
  if (bodies_[key].expired()) {
    bodies_.erase(key);
  }
}

//

VideoDataReader::QueuePair::QueuePair(int_tp size) {
  // Initialize the free queue with requested number of datumlists
  for (int_tp i = 0; i < size; ++i) {
    free_.push(new DatumList());
  }
}

VideoDataReader::QueuePair::~QueuePair() {
  DatumList* datum;
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
}

//

VideoDataReader::Body::Body(const LayerParameter& param, device* device_context)
    : param_(param),
      new_queue_pairs_() {
  StartInternalThread(device_context);
}

VideoDataReader::Body::~Body() {
  StopInternalThread();
}

void VideoDataReader::Body::InternalThreadEntry() {
  shared_ptr<db::DB> db(db::GetDB(param_.video_param().backend()));
  db->Open(param_.video_param().source(), db::READ);
  shared_ptr<db::Transaction> txn(db->NewTransaction());
  shared_ptr<db::Cursor> cur(db->NewCursor());
  vector<shared_ptr<QueuePair> > qps;
  if (param_.video_param().has_label_source()) {
    has_label_file = true;
    // read label file if exists
    // and build inverse index
    build_index();
  } else {
    has_label_file = false;
  }
  try {
    int_tp solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

    // To ensure deterministic runs, only start running once all solvers
    // are ready. But solvers need to peek on one item during initialization,
    // so read one item, then wait for the next solver.
    for (int_tp i = 0; i < solver_count; ++i) {
      shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
      read_one(txn.get(), cur.get(), qp.get());
      qps.push_back(qp);
    }
    // Main loop
    while (!must_stop()) {
      for (int_tp i = 0; i < solver_count; ++i) {
        read_one(txn.get(), cur.get(), qps[i].get());
      }
      // Check no additional readers have been created. This can happen if
      // more than one net is trained at a time per process, whether single
      // or multi solver. It might also happen if two data layers have same
      // name and same source.
      CHECK_EQ(new_queue_pairs_.size(), 0);
    }

  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

void VideoDataReader::Body::build_index() {
  label_count = 0;
  LOG(INFO) << "Start building inverse index for label vs videos ...";
  std::ifstream infile(param_.video_param().label_source());
  string video_id;
  int_tp frame_begin, duration, label;
  while (infile >> video_id >> frame_begin >> duration >> label) {
    label_index[label].push_back(
        std::make_tuple(video_id, frame_begin, duration, label));
    label_count = std::max(label_count, label);
  }
}

/**
 * Random sample:
 * Return the sampled DatumList into dl
 * 1) Sample a label class
 * 2) Sample a video id
 * 3) Sample a start frame (from start_frame + rand(0, duration - temporal_size))
 * return the DatumList
 */
void VideoDataReader::Body::random_sample(
    db::Transaction* txn, DatumList* dl, int_tp* label) {
  typedef boost::uniform_int<> distribution_type;
  typedef boost::variate_generator<caffe::rng_t*, boost::uniform_int<> > generator_type;
  DLOG(INFO) << "Label count :" << label_count;
  distribution_type label_distribution(0, label_count);
  generator_type label_sampler(caffe_rng(), label_distribution);
  int_tp label_choice = label_sampler();
  distribution_type sample_id_distribution(
      0, (label_index[label_choice]).size()-1);
  generator_type sample_id_sampler(caffe_rng(), sample_id_distribution);
  int_tp sample_choice = sample_id_sampler();

  // get sample information
  LOG(INFO) << "label choice: " << label_choice
	    << " sample choice: " << sample_choice;
  std::tuple<std::string, int_tp, int_tp, int_tp> sample_
      = (label_index[label_choice])[sample_choice];
  std::string sample_video_id = std::get<0>(sample_);
  int_tp sample_duration = std::get<2>(sample_);
  int_tp sample_begin_frame = std::get<1>(sample_);
  *label = std::get<3>(sample_);

  CHECK(sample_duration > param_.video_param().temporal_size())
    <<    "Sample duration :" << sample_duration
    <<  "temproal_size: " << param_.video_param().temporal_size();
  
  distribution_type begin_frame_distribution(
      0, sample_duration - param_.video_param().temporal_size());
  generator_type begin_frame_sampler(caffe_rng(), begin_frame_distribution);
  int_tp begin_frame_choice = begin_frame_sampler();
  fetch_one_sample(
      txn, dl, sample_video_id, begin_frame_choice + sample_begin_frame);
}

void VideoDataReader::Body::uniform_scan(
    db::Cursor* cur, DatumList* dl) {
  // Uniformly slide across all frames, this is used for validation and testing
  int_tp temporal_size = param_.video_param().temporal_size();
  int_tp stride = param_.video_param().temporal_stride();
  if (current_dl_finished) {
    current_dl.ParseFromString(cur->value());
    current_frame_idx = 0;
    current_dl_finished = false;
  }
  // get a blob
  for (int i=0; i<temporal_size; ++i) {
    Datum* datum = dl->add_datums();
    datum->CopyFrom(current_dl.datums(i+current_frame_idx));
  }
  current_frame_idx += stride;
  if (current_frame_idx + temporal_size >= current_dl.datums_size()) {
    // remaining frames are not enough for a new sample
    // prepare to fetech next datum_list from cursor
    current_dl_finished = true;
    cur->Next();
    if (!cur->valid()) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      cur->SeekToFirst();
    }
  }
}

/**
 * Fetch one sample from txn
 * The sample could span on multiple chunks
 */
void VideoDataReader::Body::fetch_one_sample(
    db::Transaction* txn, DatumList* dl,
    std::string video_id, int_tp frame_begin) {
  CHECK(dl);
  shared_ptr<DatumList> chunk_aggragated(new caffe::DatumList());
  int_tp temporal_size = param_.video_param().temporal_size();
  int_tp min_chunk_id = chunk_id(frame_begin);
  int_tp max_chunk_id = chunk_id(frame_begin + temporal_size - 1);
  for (int_tp chunk_id=min_chunk_id; chunk_id <= max_chunk_id; ++chunk_id) {
    string key_str = video_id + "_" + caffe::format_int(chunk_id, 6);
    DLOG(INFO) << "Retrieving key: " << key_str;
    vector<char> value;
    txn->Get(key_str, &value);
    DLOG(INFO) << "value length: " << value.size();
	
    caffe::DatumList chunk;
    DLOG(INFO) << "Parsing datumlist";
    CHECK(chunk.ParseFromArray(value.data(), value.size()));
    DLOG(INFO) << "Merging datumlist";
    chunk_aggragated->MergeFrom(chunk);
  }
  int_tp start_idx = frame_begin % 100;
  for (int i=0; i<param_.video_param().temporal_size(); ++i) {
    Datum* datum = dl->add_datums();
    datum->CopyFrom(chunk_aggragated->datums(i+start_idx));
  }
}

void VideoDataReader::Body::read_one(
    db::Transaction* txn, db::Cursor* cur, QueuePair* qp) {
  DatumList* dl = qp->free_.pop();
  dl->clear_datums();
  // by default, use 0 as label
  int_tp label = 0;
  if (has_label_file && param_.video_param().sampling()) {
    random_sample(txn, dl, &label);
  } else {
    uniform_scan(cur, dl);
  }
  // hard encode the label into the first datum in datumlist
  dl->mutable_datums(0)->set_label(label);
  qp->full_.push(dl);
}

}  // namespace caffe
