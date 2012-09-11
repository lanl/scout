#include "scout/Runtime/system.h"
#include <unistd.h>

using namespace scout;

namespace scout{

class system_rt_{
public:
  system_rt_(system_rt* o){
  }

  ~system_rt_(){
  }

  size_t totalProcessingUnits() const{
    return sysconf( _SC_NPROCESSORS_ONLN );
  }
};

} // end namespace scout

system_rt::system_rt(){
  x_ = new system_rt_(this);
}

system_rt::~system_rt(){
  delete x_;
}

size_t system_rt::totalProcessingUnits() const{
  return x_->totalProcessingUnits();
}

