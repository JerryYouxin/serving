#ifndef SERVING_PROCESSOR_SERVING_TRACER_H
#define SERVING_PROCESSOR_SERVING_TRACER_H

#include <fstream>
#include <iostream>
#include <string>
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace serving {

enum class TimelineLocation {
  LOCAL,
  OSS // Not Supported
};

class Tracer {
 public:
  static Tracer* GetTracer() {
    static Tracer t;
    return &t;
  }

  ~Tracer() {}

  Tracer() : tracing_(false), curr_step_(0) {}

  void SetParams(const int64_t start_step,
                 const int64_t interval_step,
                 const int64_t tracing_count,
                 const std::string& path) {
    location_type_ = TimelineLocation::LOCAL;
    tracing_ = true;
    next_tracing_step_ = start_step;
    interval_step_ = interval_step;
    tracing_count_ = tracing_count;
    limit_step_ = start_step +
        interval_step * tracing_count;
    ParseFilePath(path);
    PrintDebugString();
  }


  bool NeedTracing() {
    if (!tracing_) return false;

    if (curr_step_ < limit_step_) {
      int64_t s = curr_step_.fetch_add(1, std::memory_order_relaxed);
      if (s == next_tracing_step_) {
        mutex_lock lock(mu_);
        next_tracing_step_ += interval_step_;
        return true;
      }
    }

    return false;
  }

  void GenTimeline(tensorflow::RunMetadata& run_metadata) {
    static std::atomic<int> counter(0);
    int index = counter.fetch_add(1, std::memory_order_relaxed);
    std::string outfile;
    run_metadata.step_stats().SerializeToString(&outfile);
    string file_name = file_path_dir_ + "timeline-" +
        std::to_string(index);
 
    std::ofstream ofs;
    ofs.open(file_name);
    ofs << outfile;
    ofs.close();
  }

 private:
  void ParseFilePath(const std::string& path) {
    // Local
    if (path[0] == '/') {
      file_path_dir_ = path;
      file_path_dir_ += "/";
    } else {
      LOG(FATAL) << "Valid path must be start with absolute local path.";
    }
  }

  void PrintDebugString() {
    LOG(INFO) << "tracing_: " << tracing_
              << ", next_tracing_step_: " << next_tracing_step_
              << ", interval_step_: " << interval_step_
              << ", tracing_count_: " << tracing_count_
              << ", limit_step_: " << limit_step_
              << ", file_path_dir_: " << file_path_dir_;
  }

 private:
  bool tracing_ = false;
  int64_t next_tracing_step_ = 0;
  int64_t interval_step_ = 1;
  int64_t tracing_count_ = 0;
  int64_t limit_step_ = 0;
  std::atomic<int64_t> curr_step_;
  TimelineLocation location_type_ = TimelineLocation::LOCAL;

  std::string file_path_dir_ = "";

  mutex mu_;
};

} // namespace processor
} // namespace tensorflow

#endif // SERVING_PROCESSOR_SERVING_TRACER_H
