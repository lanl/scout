#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

double urand(double a, double b){
  return a + (b - a) * drand48();
}

void nsleep(double dt){
  double sec = floor(dt);
  double fsec = dt - sec;
  
  struct timespec ts;
  ts.tv_sec = sec;
  ts.tv_nsec = fsec*1e9;
  
  nanosleep(&ts, 0);
}

frame MyFrame{
  timestep: {type:Timestep},
  temperature: {type:Temperature},
  size: {type: Double},
  hue: {type: Double} 
};

int main(int argc, char** argv){
  MyFrame f;
  
  window win[1024, 1024];

  double t = 0.0;

  for(int32_t i = 0; i < 500; ++i){
    t += urand(-0.9, 1.0);

    into f capture{
      timestep: i,
      temperature: t,
      size: urand(1.0, 10.0),
      hue: urand(0, 360.0)  
    }

    with f in win plot{

      lines: {position:[timestep, cos(temperature)], size: 3.0,
          color: [0.0, 0.0, 1.0, 1.0]},

      points: {position:[timestep, temperature], size: size,
              color: hsva(hue, 1.0, 1.0, 1.0)},

      axis: {dim:1, label:"Timestep"},
      axis: {dim:2, label:"Temperature"}  
    }
  }
  
  return 0;
}
