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

  virtual void run(){

  }

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
    mutex_.lock();
    queue_.push_back(item);
    mutex_.unlock();
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
  MeshThread(QueueVec& queueVec, size_t id)
    : queueVec_(queueVec),
      id_(id),
      currentId_(id),
      queueSize_(queueVec_.size()),
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

      size_t i = 0;

      for(;;){
	Item item;
	if(queueVec_[currentId_ % queueSize_]->get(item)){
	  ((BlockLiteral*)item.blockLiteral)->invoke(item.blockLiteral);
	  free(item.blockLiteral);
	  ++i;
	}
	else{
	  ++currentId_;
	  if(currentId_ - id_ > queueSize_){
	    break;
	  }
	}
      }

      finishSem_.release();
    }
  }
  
private:
  QueueVec& queueVec_;
  size_t id_;
  size_t currentId_;
  size_t queueSize_;
  VSem beginSem_;
  VSem finishSem_;
};

typedef vector<MeshThread*> ThreadVec;

} // end namespace

namespace scout{

  class tbq_rt_{
  public:
    tbq_rt_(tbq_rt* o)
      : o_(o),
	q_(0){
      
      system_rt sysinfo;

      size_t n = sysinfo.totalProcessingUnits();

      //size_t n = 1;

      for(size_t i = 0; i < n; ++i){
	queueVec_.push_back(new Queue);
      }

      for(size_t i = 0; i < n; ++i){
	MeshThread* ti = new MeshThread(queueVec_, i);
	ti->start();
	threadVec_.push_back(ti);
      }

      queueSize_ = n;
    }

    ~tbq_rt_(){
      size_t n = queueVec_.size();
      for(size_t i = 0; i < n; ++i){
	delete queueVec_[i];
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

      uint32_t xStart = *bl->xStart;
      uint32_t xEnd = *bl->xEnd;

      uint32_t yStart;
      uint32_t yEnd;

      uint32_t zStart;
      uint32_t zEnd;

      if(numDimensions > 1){
	yStart = *bl->yStart;
	yEnd = *bl->yEnd;

	if(numDimensions > 2){
	  zStart = *bl->zStart;
	  zEnd = *bl->zEnd;
	}
	else{
	  zStart = 0;
	  zEnd = 0;
	}
      }
      else{	
	yStart = 0;
	yEnd = 0;
	zStart = 0;
	zEnd = 0;
      }
      
      size_t xSpan = (xEnd - xStart)/queueSize_ + 1; 

      if(yEnd == 0){
	for(size_t i = 0; i < xEnd; i += xSpan){

	  uint32_t iEnd = i + xSpan;

	  if(iEnd > xEnd){
	    iEnd = xEnd;
	  }

	  Item item;
	  item.blockLiteral = createSubBlock(bl, numDimensions,
					     numFields, i, iEnd); 

	  queueVec_[q_++ % queueSize_]->add(item);
	}
      }
      else{
	size_t ySpan = (yEnd - yStart)/queueSize_ + 1;
	
	if(zEnd == 0){
	  for(size_t i = 0; i < xEnd; i += xSpan){
	    for(size_t j = 0; j < yEnd; j += ySpan){

	      uint32_t iEnd = i + xSpan;
	      
	      if(iEnd > xEnd){
		iEnd = xEnd;
	      }

	      uint32_t jEnd = j + ySpan;
	      
	      if(jEnd > yEnd){
		jEnd = yEnd;
	      }
	      
	      Item item;
	      item.blockLiteral = createSubBlock(bl, numDimensions, 
						 numFields,
						 i, iEnd,
						 j, jEnd); 

	      queueVec_[q_++ % queueSize_]->add(item);
	    }
	  }
	}
	else{
	  size_t zSpan = (zEnd - zStart)/queueSize_ + 1;

	  for(size_t i = 0; i < xEnd; i += xSpan){
	    for(size_t j = 0; j < yEnd; j += ySpan){
	      for(size_t k = 0; k < zEnd; k += zSpan){

		uint32_t iEnd = i + xSpan;
		
		if(iEnd > xEnd){
		  iEnd = xEnd;
		}

		uint32_t jEnd = j + ySpan;
		
		if(jEnd > yEnd){
		  jEnd = yEnd;
		}
		
		uint32_t kEnd = k + zSpan;
		
		if(kEnd > zEnd){
		  kEnd = zEnd;
		}

		Item item;
		item.blockLiteral = createSubBlock(bl, 
						   numDimensions,
						   numFields,
						   i, iEnd,
						   j, jEnd,
						   k, kEnd); 
		
		queueVec_[q_++ % queueSize_]->add(item);
	      }
	    }
	  }
	}
      }
    }

    void run(){
      for(size_t i = 0; i < queueSize_; ++i){
	threadVec_[i]->begin();
      }
      
      // run ...

      for(size_t i = 0; i < queueSize_; ++i){
	threadVec_[i]->finish();
      }
    }

  private:
    tbq_rt* o_;
    QueueVec queueVec_;
    ThreadVec threadVec_;
    size_t q_;
    size_t queueSize_;
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
