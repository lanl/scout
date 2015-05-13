#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

double urand(double a, double b){
  return a + (b - a) * drand48();
}

frame MyFrame{
  timestep: {type:Timestep},
  temperature: {type:Temperature}
};

int main(int argc, char** argv){
  MyFrame f;
  
  window win[512, 512];

  double t = 0.0;

  for(int32_t i = 0; i < 500; ++i){
    t += urand(-0.9, 1.0);

    into f capture{
      timestep: i,
      temperature: t
    }

    with f in win plot{
      antialiased: false,
      
      lines: {position: [timestep, temperature],
               color: [0.7, 0.5, 0.3, 1.0],
               size: 1.0},

      axis: {dim:1, label:"Timestep"},
      axis: {dim:2, label:"Temperature"}  
    }
  }
  
  return 0;
}
