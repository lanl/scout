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
  
  // ... void* fields
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

void* _runThread(void* t){
  Thread* thread = static_cast<Thread*>(t);
  thread->run();
  return 0;
};

struct Item{
  BlockLiteral* blockLiteral;
};

class Queue{
public:
  Queue(){
    queue_.push_back(Item());
    head_ = queue_.begin();
    tail_ = queue_.end();
  }
  
  void add(const Item& item){
    queue_.push_back(item);
    tail_ = queue_.end();
    queue_.erase(queue_.begin(), head_);
  }

  bool get(Item& item){
    Queue_::iterator next = head_;
    ++next;
    if(next != tail_){
      head_ = next;
      item = *head_;
      return true;
    }
    return false;
  }

private:
  typedef list<Item> Queue_;

  Queue_ queue_;
  Queue_::iterator head_;
  Queue_::iterator tail_;
};

typedef vector<Queue*> QueueVec;

class MeshThread : public Thread{
public:
  MeshThread(QueueVec& queueVec, size_t id)
    : queueVec_(queueVec),
      id_(id),
      currentId_(id),
      queueSize_(queueVec_.size()){
    
  }

  void run(){
    for(;;){
      Item item;
      if(queueVec_[currentId_ % queueSize_]->get(item)){
	item.blockLiteral->invoke(item.blockLiteral);
	free(item.blockLiteral);
      }
      else{
	++currentId_;
	if(currentId_ - id_ > queueSize_){
	  return;
	}
      }
    }
  }
  
private:
  QueueVec& queueVec_;
  size_t id_;
  size_t currentId_;
  size_t queueSize_;
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

      for(size_t i = 0; i < n; ++i){
	queueVec_.push_back(new Queue);
      }

      for(size_t i = 0; i < n; ++i){
	MeshThread* ti = new MeshThread(queueVec_, i);
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

    void queue(void* blockLiteral, int numFields){
      void* bp = malloc(sizeof(BlockLiteral) + numFields*sizeof(void*));
      BlockLiteral* bl = (BlockLiteral*)blockLiteral;
      
      *(BlockLiteral*)bp = *bl;
      for(size_t i = 0; i < numFields; ++i){
	size_t offset = bl->descriptor->size + i * sizeof(void*);
	*(void**)((char*)bp + offset) = *(void**)((char*)bl + offset);
      }
      
      Item item;
      item.blockLiteral = (BlockLiteral*)bp;
      queueVec_[q_ % queueSize_]->add(item);
      ++q_;
    }

    void run(){
      for(size_t i = 0; i < queueSize_; ++i){
	threadVec_[i]->start();
      }
      
      // run ...

      for(size_t i = 0; i < queueSize_; ++i){
	threadVec_[i]->await();
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

void tbq_rt::queue(void* blockLiteral, int numFields){
  x_->queue(blockLiteral, numFields);
}

void tbq_rt::run(){
  x_->run();
}
