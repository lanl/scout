#include <iostream>

using namespace std;

int main(int argc, char** argv){

  uniform mesh MyMesh{
  cells:
    float i;
  };

  MyMesh m[512,512];

  forall cells c of m{
    i = position.x;
  } 

  for(float k = 0.0; k < 1.0; k += 0.01){  
    renderall cells c of m{
      color = hsva(i/512.0*360.0, i/512.0, k, 1.0);
    }

    renderall cells c of m{
      color = hsva(i/512.0*360.0, i/512.0, k, 1.0);
    }
  }
  

  return 0;
}

