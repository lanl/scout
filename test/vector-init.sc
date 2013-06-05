#include <stdio.h>

int main(int argc, char** argv){
  double4 x = double4(0.0, 1.0, 2.0, 3.0);
 
  if(x.x*x.x > 1.e-10) return -1; 
  if((x.y-1)*(x.y-1) > 1e-10) return -2; 
  if((x.z-2)*(x.y-2) > 1e-10) return -3; 
  if((x.w-3)*(x.w-3) > 1e-10) return -4; 

  return 0;
}

