#include <iostream>

using namespace std;

int main(int argc, char** argv){

  float MAX_TEMP = 1000;

  uniform mesh MyMesh{
  cells:
    float a;
    float b;
  vertices:
    float c;
    float d;
  };

  MyMesh m[512,512];

  m.a[1 .. width-2][1 .. height-2] = MAX_TEMP;

  return 0;
}
