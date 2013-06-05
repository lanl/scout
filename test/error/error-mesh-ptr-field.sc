#include <iostream>

using namespace std;

int main(int argc, char** argv){

  uniform mesh MyMesh{
  cells:
    float* a;
    float b;
  };

  MyMesh myMesh[512,512];

  return 0;
}
