#include <stdlib.h>
#include <math.h>

double exponential(double x){
  return -1.0/x * log(1.0 - drand48());
}

uniform mesh MyMesh{
 cells:
  float h;
};

int main(int argc, char** argv){
  MyMesh m[64, 64];
  
  window win[512, 512];

  forall cells c in m {
    h = exponential(1.0);
  }

  for(size_t i = 0; i < 500; ++i){
    with m in win plot{
       interval: {position: {bin:h + 10, n:100},
          color: [0.2, 0.3, 0.4, 1.0]},

        axis: {dim:1, label:"Temperature"},

        axis: {dim:2, label:"Count"} 
    }

    forall cells c in m {
      h += 0.008*exponential(0.2);
    }
  }
  
  return 0;
}
