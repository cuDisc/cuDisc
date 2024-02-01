
#ifndef _CUDISC_HEADERS_TIMING_H_
#define _CUDISC_HEADERS_TIMING_H_


#include <assert.h>
#include <time.h>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#ifdef _OPENMP
#include "omp.h"
#else
typedef int omp_nest_lock_t ;
inline void omp_set_nest_lock(omp_nest_lock_t*) {} 
inline void omp_unset_nest_lock(omp_nest_lock_t*) {} 
inline void omp_init_nest_lock(omp_nest_lock_t*) {} 
inline void omp_destroy_nest_lock(omp_nest_lock_t*) {} 
inline bool omp_in_parallel() { return false ;}
inline int  omp_get_thread_num() { return 0 ;}
inline int  omp_get_num_threads() { return 1 ;}
#endif

//==============================================================================
//  OmpNestLockGuard
/// A scoped guard for an OpenMP lock, adapted from the book "Pattern-Oriented
/// Software Architecture".
//==============================================================================
class OmpNestLockGuard {
public:
  /// Acquire the lock and store a pointer to it
  OmpNestLockGuard(omp_nest_lock_t *lock) : _lock (lock), _owner (false) {
    acquire();
  }
  /// Set the lock explicitly
  void acquire() {
    omp_set_nest_lock(_lock);
    _owner = true;
  }
  /// Release the lock explicitly (owner thread only!)
  void release() {
    if (_owner) {
      _owner = false;
      omp_unset_nest_lock(_lock);
    }
  }
  ~OmpNestLockGuard() {
    release();
  }
private:
  omp_nest_lock_t *_lock;
  bool _owner;

  // Disallow copies or assignment
  OmpNestLockGuard(const OmpNestLockGuard &);
  void operator=(const OmpNestLockGuard &);
};


//==============================================================================
//  WallClockTime
/// Returns current world clock time (from set point) in seconds
//==============================================================================
inline double WallClockTime()
{
  return clock() / (double) CLOCKS_PER_SEC;
}

struct TimingData {
  double ttot ;
  int Ncalled ;

  TimingData()
    : ttot(0), Ncalled(0)
  { } ;

  TimingData& operator+=(TimingData& other) {
    ttot    += other.ttot ;
    Ncalled += other.Ncalled ;

    return *this ;
  }
  
};

struct TimingNode
{
  std::string block_name;

  TimingData data ;

  int  timing_level;

  int timing_in_progress;


  TimingNode *parent ;
  std::map<std::string, TimingNode> children ;

  TimingNode()
  {
    timing_in_progress  = false;
    timing_level        = -1;
    data.Ncalled        = 0;
    data.ttot           = 0;
    block_name          = "";
    parent              = NULL;
  }

  void StartTiming() {
    timing_in_progress++ ;
  }

  void EndTiming(double time_interval) {
    data.ttot += time_interval ;

    data.Ncalled++ ;
    timing_in_progress--;
  }

  bool IsActive() const {
    return timing_in_progress;      
  }
    
};

inline bool operator<(const TimingNode& l, const TimingNode& r) {
  return l.block_name < r.block_name ;
}

class CodeTiming
{
public:

  class BlockTimer ;

  //============================================================================
  //  Class BlockTimer
  /// \brief   Object orientated way of handling timing blocks.
  /// \details This class starts the timing on construction and finishes timing
  ///          on destruction, copying the result back to its parent CodeTiming
  //           object. 
  /// \author  R. A. Booth
  /// \date    10/10/2016
  //============================================================================
  class BlockTimer {
  public:
    BlockTimer(const std::string& block_name_, CodeTiming* parent_,
               bool delayed_start=false)
      : block_name(block_name_),
        parent(parent_),
        timing_in_progress(false)
    {
      if (not delayed_start)
        StartTiming() ;
    } ;

    BlockTimer(BlockTimer&& timer)
      : block_name(timer.block_name),
        parent(timer.parent),
        timing_block(timer.timing_block),
        tstart(timer.tstart),
        timing_in_progress(timer.timing_in_progress)
    {
      timer.timing_in_progress = false ;
    }

    void StartNewBlock(std::string block_name, bool start=true)  {
#ifndef DISABLE_CODE_TIMING
        EndTiming() ;

        if (start)
            *this = parent->StartNewTimer(block_name) ;
        else    
            *this = parent->NewTimer(block_name) ;
#endif
    }


    ~BlockTimer() {
      EndTiming() ;
    }
    

    void StartTiming() {
#ifndef DISABLE_CODE_TIMING
      assert(not timing_in_progress) ;
      assert(parent != NULL) ;
      timing_block = parent->StartTimingBlock(block_name) ;

      if (timing_block != NULL) {
        tstart = WallClockTime() ;
        timing_in_progress = true ;
      }
#endif
    }

    void EndTiming() {
#ifndef DISABLE_CODE_TIMING
      if (timing_in_progress) {
        cudaDeviceSynchronize() ;
        parent->EndTimingBlock(timing_block, WallClockTime() - tstart) ;
        timing_in_progress = false ;
      }
#endif
    }    

    bool IsActive() const {
      return timing_in_progress ;
    }
    
    friend class CodeTiming ;

  private:
    BlockTimer& operator=(BlockTimer&& timer) {
      assert(not timing_in_progress) ;

      block_name = timer.block_name ;
      parent = timer.parent ;
      timing_block = timer.timing_block ;
      tstart = timer.tstart ;
      timing_in_progress = timer.timing_in_progress ;

      timer.timing_in_progress = false ;

      return *this ;
    }

    std::string block_name ;
    CodeTiming* parent ;
    TimingNode* timing_block ;
    double tstart ;
    bool timing_in_progress;
  };


  CodeTiming(std::string basename_="")
    : active_blocks(1,&root_node),
      global_block("Global", this, true),
      basename(basename_)   
  {
    do_not_time = (basename == "") ;

    omp_init_nest_lock(&timing_lock) ;
    global_block.StartTiming() ;
  } ;

  ~CodeTiming() {
    FinishTiming() ;
    SaveTimingStatistics() ;
  }
  
  double RunningTime() {
    OmpNestLockGuard lock_guard(&timing_lock) ;
    
    TimingNode& blk = root_node.children["Global"] ;
    if (global_block.IsActive())
      return WallClockTime() - global_block.tstart + blk.data.ttot ;
    else
      return blk.data.ttot ;
  }

  void FinishTiming() {
    if (do_not_time)
      return ;

    OmpNestLockGuard lock_guard(&timing_lock) ;   

    TimingNode* active_block = get_active_block() ;
    if (active_block != &root_node && active_block != global_block.timing_block)
      throw std::runtime_error("CodingTiming cannot be reset while timing is "
			       "active") ;

    if (global_block.IsActive()) 
      global_block.EndTiming() ;

    omp_destroy_nest_lock(&timing_lock) ;
  }

  // Create a timer for a new section. StartNewTimer also begins the timing.
  //---------------------------------------------------------------------------
  BlockTimer StartNewTimer(std::string block_name) {
    return BlockTimer(block_name, this) ;
  }
  BlockTimer NewTimer(std::string block_name) {
    return BlockTimer(block_name, this,  true);
  }

  void SaveTimingStatistics();

 private:
  TimingNode* StartTimingBlock(const std::string& block_name) {

    if (do_not_time)
      return NULL ;

    OmpNestLockGuard lock_guard(&timing_lock) ;

    TimingNode*& active_block = get_active_block() ;
    
    TimingNode& child  = active_block->children[block_name] ;
    
    child.timing_level = active_block->timing_level + 1 ;
    child.block_name = block_name ;
    child.parent = active_block ;
    
    child.StartTiming() ;

    active_block = &child ;
    
    return active_block ;
  }
  void EndTimingBlock(TimingNode* node, double time_interval) {

    if (do_not_time)
      return ;

    OmpNestLockGuard lock_guard(&timing_lock) ;
    
    TimingNode*& active_block = get_active_block();

    __check_timing_level(node, active_block) ;

    if (tracking_parallel)
      time_interval /= Nthreads ;
    
    active_block->EndTiming(time_interval) ;

    active_block = active_block->parent ;
  }

  TimingNode*& get_active_block() {

    bool parallel = omp_in_parallel() ;

    int tid = 0 ;

    if (parallel)
      tid = omp_get_thread_num() ;
    
    if (parallel && not tracking_parallel) {
      Nthreads = omp_get_num_threads() ;

      TimingNode* active = active_blocks[0] ;
      active_blocks = std::vector<TimingNode*>(Nthreads, active) ;

      tracking_parallel = true ;
    }

    if (not parallel && tracking_parallel) {
      active_blocks.resize(1) ;
      tracking_parallel = false ;
    }
    
    return active_blocks[tid] ;
  }
    
  void __check_timing_level(TimingNode* node, TimingNode* active_block) const {
    if (node != active_block) {
      throw std::runtime_error("Error: Timing blocks were released in the wrong"
			       " order");
    }
  }

  omp_nest_lock_t timing_lock ;
  TimingNode root_node ;
  std::vector<TimingNode*> active_blocks;
  BlockTimer global_block ;

  int Nthreads = 1 ;
  bool tracking_parallel = false; 
  bool do_not_time = false ;
  
  std::string basename;

};

extern std::unique_ptr<CodeTiming> timer ;


#endif//_CUDISC_HEADERS_TIMING_H_
