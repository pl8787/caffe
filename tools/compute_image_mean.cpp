#include <glog/logging.h>
#include <stdint.h>

#include <fstream>
#include <algorithm>
#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using caffe::Datum;
using caffe::BlobProto;
using std::string;
using std::max;

int resize_height = 0;
int resize_width = 0;
bool is_color = true;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

  std::ifstream infile(argv[1]);
  std::vector<std::pair<string, int> > lines;
  string filename;
  int label;
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  Datum datum;
  BlobProto sum_blob;
  int count = 0;

  if (!ReadImageToDatum(lines[0].first, lines[0].second, 
         resize_height, resize_width, is_color, &datum)) {
    return -1;
  }

  sum_blob.set_num(1);
  sum_blob.set_channels(datum.channels());
  sum_blob.set_height(datum.height());
  sum_blob.set_width(datum.width());
  const int data_size = datum.channels() * datum.height() * datum.width();
  int size_in_datum = std::max<int>(datum.data().size(),
                                    datum.float_data_size());
  for (int i = 0; i < size_in_datum; ++i) {
    sum_blob.add_data(0.);
  }

  LOG(INFO) << "Starting Iteration";
  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    if (!ReadImageToDatum(lines[line_id].first, lines[line_id].second, 
           resize_height, resize_width, is_color, &datum)) {
      continue;
    }

    const string& data = datum.data();
    size_in_datum = std::max<int>(datum.data().size(),
        datum.float_data_size());
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum;
    if (data.size() != 0) {
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
      }
    } else {
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) +
            static_cast<float>(datum.float_data(i)));
      }
    }
    ++count;
  }

  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }

  // Write to disk
  LOG(INFO) << "Write to " << argv[2];
  WriteProtoToBinaryFile(sum_blob, argv[2]);

  return 0;
}
