#ifndef SERVING_PROCESSOR_SERVING_TRACER_H
#define SERVING_PROCESSOR_SERVING_TRACER_H

#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
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

class Timer {
  public:
    using Timepoint=std::chrono::time_point<std::chrono::steady_clock>;
    static Timer* GetTimer() {
      static Timer t;
      return &t;
    }

    ~Timer() {}

    Timer() : collect_(false), timer_count_(0) {}

    void Enable(int64_t start, int64_t count, std::string file_path) {
      collect_ = (start==0);
      timer_start_ = start;
      timer_count_ = count;
      timers_ = new double[count];
      file_path_ = file_path;
    }

    void Disable() {
      collect_ = false;
    }

    bool IsEnabled() {
      static std::atomic<int> counter(0);
      int index = counter.fetch_add(1, std::memory_order_relaxed);
      collect_ = (index>=timer_start_ && index<timer_start_+timer_count_);
      return collect_;
    }

    Timepoint Start() {
      return std::chrono::steady_clock::now();
    }

    void Stop(Timepoint start) {
      static std::atomic<int> counter(0);
      Timepoint stop = std::chrono::steady_clock::now();
      int index = counter.fetch_add(1, std::memory_order_relaxed);
      if(index>=timer_count_) {
        collect_ = false;
        return ;
      }
      std::chrono::duration<double, std::milli> fp_ms = stop - start;
      timers_[index] = fp_ms.count();
      if(index==timer_count_-1) {
        GenStatistics(timer_count_);
      }
    }

  private:
    bool collect_ = false;
    int64_t timer_start_ = 0;
    int64_t timer_count_ = 0;
    double* timers_;
    std::string file_path_;

    void GenStatistics(int count) {
      double time_avg = 0;
      double time_std = 0;
      double time_max = timers_[0];
      double time_min = timers_[0];
      for(int i=0; i<count; ++i) {
        time_avg += timers_[i];
        if(time_max < timers_[i]) {
          time_max = timers_[i];
        }
        if(time_min > timers_[i]) {
          time_min = timers_[i];
        }
      }
      time_avg = time_avg / count;
      for(int i=0; i<count; ++i) {
        time_std += (timers_[i]-time_avg)*(timers_[i]-time_avg);
      }
      time_std = sqrt(time_std / count);
      std::ofstream ofs;
      ofs.open(file_path_);
      ofs << time_avg << ",";
      ofs << time_std << ",";
      ofs << time_max << ",";
      ofs << time_min;
      ofs.close();
    }
};

} // namespace processor
} // namespace tensorflow

#endif // SERVING_PROCESSOR_SERVING_TRACER_H
