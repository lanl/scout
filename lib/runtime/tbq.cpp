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

#include <vector>

#include "runtime/sysinfo_summary.h"

using namespace std;
using namespace scout;

namespace{

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
  void (^block)(index_t*,index_t*,index_t*,tbq_params_rt);
  range_t xRange;
  range_t yRange;
  range_t zRange;
  void* mesh;
  tbq_params_rt params;
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
	index_t xStart = item.xRange.lower_bound;
	index_t xEnd = item.xRange.upper_bound;
	index_t xStride = item.xRange.stride;

	if(item.yRange.stride == 0){
	  for(index_t i = xStart; i < xEnd; i += xStride){
	    item.block(&i, 0, 0, item.params); 
	  }
	}
	else{
	  index_t yStart = item.yRange.lower_bound;
	  index_t yEnd = item.yRange.upper_bound;
	  index_t yStride = item.yRange.stride;
	
	  if(item.zRange.stride == 0){
	    for(index_t i = xStart; i < xEnd; i += xStride){
	      for(index_t j = yStart; j < yEnd; j += yStride){
		item.block(&i, &j, 0, item.params);
	      } 
	    }
	  }
	  else{
	    index_t zStart = item.zRange.lower_bound;
	    index_t zEnd = item.zRange.upper_bound;
	    index_t zStride = item.zRange.stride;
	    
	    for(index_t i = xStart; i < xEnd; i += xStride){
	      for(index_t j = yStart; j < yEnd; j += yStride){
		for(index_t k = zStart; k < zEnd; k += zStride){
		  item.block(&i, &j, &k, item.params);
		}
	      } 
	    }
	  }
	}
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
      : o_(o){
      sysinfo_summary_rt sysinfo;

      size_t n = sysinfo.totalProcessingUnits();

      for(size_t i = 0; i < n; ++i){
	queueVec_.push_back(new Queue);
      }

      for(size_t i = 0; i < n; ++i){
	MeshThread* ti = new MeshThread(queueVec_, i);
	threadVec_.push_back(ti);
      }
    }

    ~tbq_rt_(){
      size_t n = queueVec_.size();
      for(size_t i = 0; i < n; ++i){
	delete queueVec_[i];
	delete threadVec_[i];
      }
    }

    void run(void (^block)(index_t*,index_t*,index_t*,tbq_params_rt),
	     range_t xRange, range_t yRange, range_t zRange){

      size_t queueSize = queueVec_.size();
      
      index_t xspan = 
	(xRange.upper_bound - xRange.lower_bound)/queueVec_.size();

      size_t q = 0;
      if(yRange.stride == 0){
	for(index_t i = 0; i <= xRange.upper_bound; i += xspan){
	  Item item;
	  item.xRange.lower_bound = i;
  
	  index_t end = i + xspan;
	  if(end > xRange.upper_bound){
	    end = xRange.upper_bound;
	  }

	  item.xRange.upper_bound = end;

	  item.xRange.stride = xRange.stride;
	  item.yRange.stride = 0;
	  item.zRange.stride = 0;

	  queueVec_[q % queueSize]->add(item);
	  ++q;
	}
      }
      else{
	index_t yspan = 
	  (yRange.upper_bound - yRange.lower_bound)/queueVec_.size();

	if(zRange.stride == 0){
	  for(index_t i = 0; i <= xRange.upper_bound; i += xspan){
	    for(index_t j = 0; j <= yRange.upper_bound; j += yspan){
	      Item item;
	      item.xRange.lower_bound = i;
	      item.yRange.lower_bound = j;

	      index_t end = i + xspan;
	      if(end > xRange.upper_bound){
		end = xRange.upper_bound;
	      }
	      item.xRange.upper_bound = end;

	      end = j + yspan;
	      if(end > yRange.upper_bound){
		end = yRange.upper_bound;
	      }

	      item.yRange.upper_bound = end;
	    
	      item.xRange.stride = xRange.stride;
	      item.yRange.stride = yRange.stride;
	      item.zRange.stride = 0;
	    
	      queueVec_[q % queueSize]->add(item);
	      ++q;
	    }
	  }
	}
	else{
	  index_t zspan = 
	    (zRange.upper_bound - zRange.lower_bound)/queueVec_.size();

	  for(index_t i = 0; i <= xRange.upper_bound; i += xspan){
	    for(index_t j = 0; j <= yRange.upper_bound; j += yspan){
	      for(index_t k = 0; k <= zRange.upper_bound; k += zspan){
		Item item;
		item.xRange.lower_bound = i;
		item.yRange.lower_bound = j;
		item.yRange.lower_bound = k;

		index_t end = i + xspan;
		if(end > xRange.upper_bound){
		  end = xRange.upper_bound;
		}
		item.xRange.upper_bound = end;

		end = j + yspan;
		if(end > yRange.upper_bound){
		  end = yRange.upper_bound;
		}
		item.yRange.upper_bound = end;

		end = k + zspan;
		if(end > zRange.upper_bound){
		  end = zRange.upper_bound;
		}

		item.zRange.upper_bound = end;
	    
		item.xRange.stride = xRange.stride;
		item.yRange.stride = yRange.stride;
		item.zRange.stride = zRange.stride;
	    
		queueVec_[q % queueSize]->add(item);
		++q;
	      }
	    }
	  }
	}
      }

      for(size_t i = 0; i < queueSize; ++i){
	threadVec_[i]->start();
      }

      // run ...

      for(size_t i = 0; i < queueSize; ++i){
	threadVec_[i]->await();
      }
    }

  private:
    tbq_rt* o_;
    QueueVec queueVec_;
    ThreadVec threadVec_;
  };

} // end namespace scout

tbq_rt::tbq_rt(){
  x_ = new tbq_rt_(this);
}

tbq_rt::~tbq_rt(){
  delete x_;
}

void tbq_rt::run(void (^block)(index_t*,index_t*,index_t*,tbq_params_rt),
		 range_t xRange, range_t yRange, range_t zRange){
  x_->run(block, xRange, yRange, zRange);
}

