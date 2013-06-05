#include <iostream>

using namespace std;

int main(int argc, char** argv){

  uniform mesh MyMesh{
  cells:
    float a;
    float b;
    float c;
  };

  MyMesh m[512,512];
  
  forall cells c of m{
    a = 2;
    b = a;
  }

  return 0;
}
