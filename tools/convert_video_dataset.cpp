// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream> // NOLINT(readability/streams)
#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <tuple>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe; // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
            "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
            "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
              "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(
    check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
            "When this option is on, the encoded image will be save in datum");
DEFINE_string(
    encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char **argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage(
      "Convert a set of images to the leveldb/lmdb\n"
      "format used as input for Caffe.\n"
      "Usage:\n"
      "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
      "The ImageNet dataset for the training demo is at\n"
      "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);
  std::vector<std::tuple<std::string, int_tp, int_tp, int_tp>> lines;
  std::string fname_prefix;
  int_tp chunk_id, frame_begin, chunk_size;
  while (infile >> fname_prefix >> frame_begin >> chunk_size >> chunk_id) {
    lines.push_back(std::make_tuple(fname_prefix, frame_begin, chunk_size, chunk_id));
  }

  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int_tp resize_height = std::max<int_tp>(0, FLAGS_resize_height);
  int_tp resize_width = std::max<int_tp>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  
  int_tp count = 0;
  int_tp data_size = 0;
  bool data_size_initialized = false;

  for (int_tp line_id = 0; line_id < lines.size(); ++line_id) {
    int_tp chunk_size = std::get<2>(lines[line_id]);
    int_tp frame_begin  = std::get<1>(lines[line_id]);
    std::string fname_prefix = std::get<0>(lines[line_id]);
    int_tp chunk_id = std::get<3>(lines[line_id]);

    DatumList datum_chunk;

    for (int_tp frame_id = 0; frame_id < chunk_size; ++frame_id) {
      int_tp abs_frame_id = frame_id + frame_begin;
      std::ostringstream fname_buf;
      fname_buf << fname_prefix << "/frame_" << caffe::format_int(abs_frame_id, 6) << ".jpg";
      bool status;
      std::string enc = encode_type;
      if (encoded && !enc.size()) {
	// Guess the encoding type from the file name
	string fn = fname_buf.str();
	uint_tp p = fn.rfind('.');
	if (p == fn.npos)
	  LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
	enc = fn.substr(p);
	std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
      } 
      // Put dummy label
      Datum *datum = datum_chunk.add_datums();

      status = ReadImageToDatum(root_folder + fname_buf.str(),
				0, resize_height,
				resize_width, is_color, enc, datum);
      if (status == false)
	continue;
      if (check_size) {
	if (!data_size_initialized) {
	  data_size = datum->channels() * datum->height() * datum->width();
	  data_size_initialized = true;
	} else {
	  const std::string &data = datum->data();
	  CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
					   << data.size();
	}
      }
    }
    // sequential
    string key_str = fname_prefix + "_" + caffe::format_int(chunk_id, 6);

    // Put in db
    string out;
    CHECK(datum_chunk.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 100 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " chunks.";
    }
  }
  // write the last batch
  if (count % 100 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " chunks.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif // USE_OPENCV
  return 0;
}
