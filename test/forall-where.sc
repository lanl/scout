#include <iostream>

using namespace std;

int main(int argc, char** argv){

  uniform mesh MyMesh{
  cells:
    float a;
    float b;
  vertices:
    float c;
    float d;
  };

  MyMesh m[512,512];
  
  forall cells c of m where(c.b > 2){
    c.a = 100;
  }

  return 0;
}
