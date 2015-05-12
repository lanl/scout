#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

double urand(double a, double b){
  return a + (b - a) * drand48();
}

int main(int argc, char** argv){  
  window win[1024, 1024];

  double t = 0.0;

  for(int32_t i = 0; i < 500; ++i){
    t += urand(-0.9, 1.0);

    in win plot{
      lines: {position: [i, t + 100]},

      axis: {dim:1, label:"Timestep"},
      axis: {dim:2, label:"Temperature"}  
    }
  }
  
  return 0;
}
