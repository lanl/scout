#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <stdio.h>

double urand(double a, double b){
  return a + (b - a) * drand48();
}

frame MyFrame{
  timestep: {type:Timestep},
  temperature: {type:Temperature},
  label: {type:String}
};

int main(int argc, char** argv){
  MyFrame f;
  
  window win[1024, 1024];

  double t = 0.0;

  for(int32_t i = 0; i < 50; ++i){
    t += urand(-0.9, 1.0);

    char buf[64];
    sprintf(buf, "label%d", i);

    into f capture{
      timestep: i,
      temperature: t,
      label: buf
    }
  }

  with f in win plot{        
    points: {position: [timestep, temperature], size: 5.0, label: label},

    axis: {dim:1, label:"Timestep"},
    axis: {dim:2, label:"Temperature"} 
  }

  sleep(3);

  return 0;
}
