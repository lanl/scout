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
#include <map>

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
  uint32_t dimensions;
  uint32_t xStart;
  uint32_t xEnd;
  uint32_t yStart;
  uint32_t yEnd;
  uint32_t zStart;
  uint32_t zEnd;
};

class Queue{
public:
  Queue()
    : i_(0){

  }

  void reset(){
    i_ = 0;
  }

  void add(Item* item){
    queue_.push_back(item);
  }

  Item* get(){
    mutex_.lock();

    if(i_ >= queue_.size()){
      mutex_.unlock();
      return 0;
    }
    
    Item* item = queue_[i_++];
    mutex_.unlock();

    return item;
  }

private:
  typedef vector<Item*> Queue_;

  Mutex mutex_;
  Queue_ queue_;
  size_t i_;
};

typedef vector<Queue*> QueueVec;

class MeshThread : public Thread{
public:
  MeshThread()
    : beginSem_(0),
      finishSem_(0),
      queue_(0){

  }

  void begin(Queue* queue){
    queue_ = queue;
    beginSem_.release();
  }

  void finish(){
    finishSem_.acquire();
  }

  void run(){
    for(;;){
      beginSem_.acquire();

      for(;;){
	Item* item = queue_->get();

	if(!item){
	  break;
	}

	BlockLiteral* bl = (BlockLiteral*)item->blockLiteral;

	switch(item->dimensions){
	  case 3:
	    bl->zStart = (uint32_t*)malloc(sizeof(uint32_t));
	    *bl->zStart = item->zStart;
	    bl->zEnd = (uint32_t*)malloc(sizeof(uint32_t));
	    *bl->zEnd = item->zEnd;
	  case 2:
	    bl->yStart = (uint32_t*)malloc(sizeof(uint32_t));
	    *bl->yStart = item->yStart;
	    bl->yEnd = (uint32_t*)malloc(sizeof(uint32_t));
	    *bl->yEnd = item->yEnd;
	  case 1:
	    bl->xStart = (uint32_t*)malloc(sizeof(uint32_t));
	    *bl->xStart = item->xStart;
	    bl->xEnd = (uint32_t*)malloc(sizeof(uint32_t));
	    *bl->xEnd = item->xEnd;
	}

	bl->invoke(bl);
      }

      finishSem_.release();
    }
  }

private:
  Queue* queue_;
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
	MeshThread* ti = new MeshThread;
	ti->start();
	threadVec_.push_back(ti);
      }
    }

    ~tbq_rt_(){
      size_t n = threadVec_.size();

      for(size_t i = 0; i < n; ++i){
	delete threadVec_[i];
      }

      for(QueueMap_::iterator itr = queueMap_.begin(),
	    itrEnd = queueMap_.end(); itr != itrEnd; ++itr){
	delete itr->second;
      }
    }

    void* createSubBlock(BlockLiteral* bl,
			 size_t numDimensions,
			 size_t numFields){

      void* bp = malloc(sizeof(BlockLiteral) - 6*sizeof(void*)
			+ numFields*sizeof(void*));

      BlockLiteral* b = (BlockLiteral*)bp;
      b->isa = bl->isa;
      b->flags = bl->flags;
      b->reserved = bl->reserved;
      b->invoke = bl->invoke;
      b->descriptor = bl->descriptor;

      size_t offset = bl->descriptor->size + 2*numDimensions*sizeof(void*);

      memcpy((char*)bp + offset, (char*)bl + offset,
	     (numFields - 2*numDimensions)*sizeof(void*));

      return bp;
    }

    Queue* queue_(void* blockLiteral, int numDimensions, int numFields){
      BlockLiteral* bl = (BlockLiteral*)blockLiteral;
      
      QueueMap_::iterator itr = queueMap_.find((void*)bl->invoke);
      if(itr != queueMap_.end()){
	return itr->second;
      }

      Queue* queue = new Queue;

      Item* item;
      uint32_t extent;
      uint32_t chunk;
      uint32_t end;

      switch(numDimensions){
        case 1:
        {
	  extent = *bl->xEnd - *bl->xStart;
	  chunk = extent / (threadVec_.size() * 2);

	  for(uint32_t i = 0; i < extent; i += chunk){
	    end = i + chunk;

	    if(end > extent){
	      end = extent;
	    }

	    item = new Item;

	    item->blockLiteral = createSubBlock(bl, numDimensions,
						numFields);

	    item->dimensions = 1;
	    item->xStart = i;
	    item->xEnd = end;

	    queue->add(item);
	  }
	  break;
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

	    item = new Item;

	    item->blockLiteral = createSubBlock(bl, numDimensions,
						numFields);

	    item->dimensions = 2;
	    item->xStart = i % x;
	    item->xEnd = end % x + 1;
	    item->yStart = i / x % y;
	    item->yEnd = end / x % y + 1;
	  
	    queue->add(item);
	  }
	  break;
	}
        case 3:
        {
	  assert(false && "runtime/tbq.cpp 3d case not yet implemented");
	  break;
	}
      }

      queueMap_[(void*)bl->invoke] = queue;
      
      return queue;
    }

    void run(void* blockLiteral, int numDimensions, int numFields){
      Queue* queue = queue_(blockLiteral, numDimensions, numFields);
      queue->reset();

      size_t n = threadVec_.size();
      
      for(size_t i = 0; i < n; ++i){
	threadVec_[i]->begin(queue);
      }

      // run ...

      for(size_t i = 0; i < n; ++i){
	threadVec_[i]->finish();
      }
    }

  private:
    typedef map<void*, Queue*> QueueMap_;

    tbq_rt* o_;
    QueueMap_ queueMap_;
    ThreadVec threadVec_;
  };

} // end namespace scout

tbq_rt::tbq_rt(){
  x_ = new tbq_rt_(this);
}

tbq_rt::~tbq_rt(){
  delete x_;
}

void tbq_rt::run(void* blockLiteral, int numDimensions, int numFields){
  x_->run(blockLiteral, numDimensions, numFields);
}
