#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

double urand(double a, double b){
  return a + (b - a) * drand48();
}

double exponential(double x){
  return -1.0/x * log(1.0 - drand48());
}

frame MyFrame{
  temperature: {type:Temperature}
};

int main(int argc, char** argv){
  MyFrame f;
  
  window win[1024, 1024];

  for(int32_t i = 0; i < 3000; ++i){
    double t = exponential(0.5) + urand(0.0, 10.0);

    into f capture{
      temperature: t
    }
  }

  for(int32_t i = 0; i < 150; ++i){
    double t = exponential(0.5) + urand(0.0, 10.0);

    into f capture{
      temperature: t
    }

    with f in win plot{
    interval: {position: {bin:temperature, n:100},
                color: [0.2, 0.3, 0.4, 1.0]},

      axis: {dim:1, label:"Temperature"},

      axis: {dim:2, label:"Count"} 
    }
  }
  
  return 0;
}
