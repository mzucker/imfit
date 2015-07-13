#ifndef _TIMER_H_
#define _TIMER_H_

#include <mach/mach_time.h>

class Timer {
public:
  
  static const mach_timebase_info_data_t& get_base() {
    static mach_timebase_info_data_t base = { 0, 0 };
    if (!base.denom) { 
      mach_timebase_info(&base);
    }
    return base;
  }

  uint64_t start_time;
  
  Timer() {
    start();
  }
  
  void start() {
    start_time = mach_absolute_time();
  };

  double elapsed() const {
    const mach_timebase_info_data_t& base = get_base();
    uint64_t delta = mach_absolute_time() - start_time;
    return (delta * base.numer / base.denom) * 1e-9;
  }

};

#endif
