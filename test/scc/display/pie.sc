#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

double urand(double a, double b){
  return a + (b - a) * drand48();
}

frame MyFrame{
  choice: {type:Int32}
};

int32_t choice(){
  double r = urand(0.0, 1.0);

  if(r < 0.1){
    return 1;
  }
  else if(r < 0.3){
    return 2;
  }
  else if(r < 0.6){
    return 3;
  }
  else{
    return 4;
  }
}

int main(int argc, char** argv){
  MyFrame f;
  
  window win[1024, 1024];

  for(int32_t i = 0; i < 500; ++i){
    into f capture{
      choice: choice()
    }

    with f in win plot{
      pie: {proportion: [choice]}
    }
  }
  
  return 0;
}
