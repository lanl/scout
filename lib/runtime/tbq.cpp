/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 *
 *-----
 *
 */


#include "runtime/tbq.h"

using namespace std;
using namespace scout;

#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>
#include <cstring>

#include "runtime/system.h"

using namespace std;
using namespace scout;

namespace{

struct BlockLiteral{
  void* isa;
  int flags;
  int reserved;
  void (*invoke)(void*, ...);
  struct BlockDescriptor{
    unsigned long int reserved;
    unsigned long int size;
    void (*copy_helper)(void* dst, void* src);
    void (*dispose_helper)(void* src);
    const char* signature;
  }*descriptor;

  // some of these fields may not actually be present depending
  // on the mesh dimensions
  uint32_t* xStart;
  uint32_t* xEnd;
  uint32_t* yStart;
  uint32_t* yEnd;
  uint32_t* zStart;
  uint32_t* zEnd;

  // ... void* captured fields
};

void* _runThread(void* t);

class Thread{
public:
  Thread(){

  }

  virtual ~Thread(){

  }

  void start(){
    pthread_create(&thread_, 0, _runThread, (void*)this);
  }

  virtual void run() = 0;

  void await(){
    pthread_join(thread_, 0);
  }

private:
  pthread_t thread_;
};

class Mutex{
public:
  Mutex(){
    pthread_mutex_init(&mutex_, 0);
  }

  ~Mutex(){
    pthread_mutex_destroy(&mutex_);
  }

  void lock(){
    pthread_mutex_lock(&mutex_);
  }

  bool tryLock(){
    return pthread_mutex_trylock(&mutex_) == 0;
  }

  void unlock(){
    pthread_mutex_unlock(&mutex_);
  }

  pthread_mutex_t& mutex(){
    return mutex_;
  }

private:
  pthread_mutex_t mutex_;
};

void* _runThread(void* t){
  Thread* thread = static_cast<Thread*>(t);
  thread->run();
  return 0;
};

class Condition{
public:
  Condition(Mutex& mutex)
    : mutex_(mutex){
    pthread_cond_init(&condition_, 0);
  }

  ~Condition(){
    pthread_cond_destroy(&condition_);
  }

  void await(){
    pthread_cond_wait(&condition_, &mutex_.mutex());
  }

  void signal(){
    pthread_cond_signal(&condition_);
  }

  void broadcast(){
    pthread_cond_broadcast(&condition_);
  }

  pthread_cond_t& condition(){
    return condition_;
  }

private:
  Mutex& mutex_;
  pthread_cond_t condition_;
};

class VSem{
public:
  VSem(int count)
    : count_(count),
      maxCount_(0),
      condition_(mutex_){

  }

  VSem(int count, int maxCount)
  : count_(count),
    maxCount_(maxCount),
    condition_(mutex_){

  }

  void acquire(){
    mutex_.lock();
    while(count_ <= 0){
      condition_.await();
    }
    --count_;
    mutex_.unlock();
  }

  bool tryAcquire(){
    mutex_.lock();
    if(count_ > 0){
      --count_;
      mutex_.unlock();
      return true;
    }
    mutex_.unlock();
    return false;
  }

  void release(){
    mutex_.lock();
    if(maxCount_ == 0 || count_ < maxCount_){
      ++count_;
    }
    condition_.signal();
    mutex_.unlock();
  }

private:
  Mutex mutex_;
  Condition condition_;
  int count_;
  int maxCount_;
};

struct Item{
  void* blockLiteral;
};

class Queue{
public:
  Queue(){

  }

  void add(const Item& item){
    queue_.push_back(item);
  }

  bool get(Item& item){
    mutex_.lock();
    if(queue_.empty()){
      mutex_.unlock();
      return false;
    }

    item = queue_.front();
    queue_.pop_front();
    mutex_.unlock();

    return true;
  }

private:
  typedef list<Item> Queue_;

  Mutex mutex_;
  Queue_ queue_;
};

typedef vector<Queue*> QueueVec;

class MeshThread : public Thread{
public:
  MeshThread(Queue& queue)
    : queue_(queue),
      beginSem_(0),
      finishSem_(0){

  }

  void begin(){
    beginSem_.release();
  }

  void finish(){
    finishSem_.acquire();
  }

  void run(){
    for(;;){
      beginSem_.acquire();

      for(;;){
	Item item;
	if(queue_.get(item)){
	  ((BlockLiteral*)item.blockLiteral)->invoke(item.blockLiteral);
	  free(item.blockLiteral);
	}
	else{
	  break;
	}
      }

      finishSem_.release();
    }
  }

private:
  Queue& queue_;
  VSem beginSem_;
  VSem finishSem_;
};

typedef vector<MeshThread*> ThreadVec;

} // end namespace

namespace scout{

  class tbq_rt_{
  public:
    tbq_rt_(tbq_rt* o)
      : o_(o){

      system_rt sysinfo;

      size_t n = sysinfo.totalProcessingUnits();

      for(size_t i = 0; i < n; ++i){
	MeshThread* ti = new MeshThread(queue_);
	ti->start();
	threadVec_.push_back(ti);
      }
    }

    ~tbq_rt_(){
      size_t n = threadVec_.size();
      for(size_t i = 0; i < n; ++i){
	delete threadVec_[i];
      }
    }

    void* createSubBlock(BlockLiteral* bl,
			 size_t numDimensions,
			 size_t numFields,
			 uint32_t xStart,
			 uint32_t xEnd,
			 uint32_t yStart=0,
			 uint32_t yEnd=0,
			 uint32_t zStart=0,
			 uint32_t zEnd=0){

      void* bp = malloc(sizeof(BlockLiteral) - 6*sizeof(void*)
			+ numFields*sizeof(void*));

      BlockLiteral* b = (BlockLiteral*)bp;
      b->isa = bl->isa;
      b->flags = bl->flags;
      b->reserved = bl->reserved;
      b->invoke = bl->invoke;
      b->descriptor = bl->descriptor;

      for(size_t i = 0; i < numDimensions; ++i){
	uint32_t* start = (uint32_t*)malloc(sizeof(uint32_t));
	uint32_t* end = (uint32_t*)malloc(sizeof(uint32_t));

	switch(i){
	  case 0:
	    *start = xStart;
	    *end = xEnd;
	    b->xStart = start;
	    b->xEnd = end;
	    break;
	  case 1:
	    *start = yStart;
	    *end = yEnd;
	    b->yStart = start;
	    b->yEnd = end;
	    break;
	  case 2:
	    *start = zStart;
	    *end = zEnd;
	    b->zStart = start;
	    b->zEnd = end;
	    break;
	}
      }

      size_t offset = bl->descriptor->size + 2*numDimensions*sizeof(void*);

      memcpy((char*)bp + offset, (char*)bl + offset,
	     (numFields - 2*numDimensions)*sizeof(void*));

      return bp;
    }

    void queue(void* blockLiteral, int numDimensions, int numFields){
      BlockLiteral* bl = (BlockLiteral*)blockLiteral;

      Item item;
      uint32_t extent;
      uint32_t chunk;
      uint32_t end;

      switch(numDimensions){
        case 1:
        {
	  extent = *bl->xEnd - *bl->xStart;
	  chunk = extent/(threadVec_.size()) + 1;

	  for(uint32_t i = 0; i < extent; i += chunk){
	    end = i + chunk;

	    if(end > extent){
	      end = extent;
	    }

	    item.blockLiteral = createSubBlock(bl, numDimensions,
					       numFields,
					       i, end);

	    queue_.add(item);
	  }
	  return;
	}
        case 2:
	{
	  uint32_t x = *bl->xEnd - *bl->xStart;
	  uint32_t y = *bl->yEnd - *bl->yStart;

	  extent = y * x;
          chunk = extent / (threadVec_.size() * 2);

	  if(chunk == 0){
	    chunk = 1;
	  }

	  for(uint32_t i = 0; i < extent; i += chunk){
	    end = i + chunk - 1;

	    if(end > extent){
	      end = extent;
	    }

	    item.blockLiteral = createSubBlock(bl, numDimensions,
					       numFields,
					       i % x, (end % x) + 1,
					       (i / x) % y, ((end / x) % y) + 1);

	    queue_.add(item);
	  }
	  return;
	}
        case 3:
        {
	  assert(false && "runtime/tbq.cpp 3d case not yet implemented");
	  return;
	}
      }
    }

    void run(){
      size_t n = threadVec_.size();
      
      for(size_t i = 0; i < n; ++i){
	threadVec_[i]->begin();
      }

      // run ...

      for(size_t i = 0; i < n; ++i){
	threadVec_[i]->finish();
      }
    }

  private:
    tbq_rt* o_;
    Queue queue_;
    ThreadVec threadVec_;
  };

} // end namespace scout

tbq_rt::tbq_rt(){
  x_ = new tbq_rt_(this);
}

tbq_rt::~tbq_rt(){
  delete x_;
}

void tbq_rt::queue(void* blockLiteral, int numDimensions, int numFields){
  x_->queue(blockLiteral, numDimensions, numFields);
}

void tbq_rt::run(){
  x_->run();
}
