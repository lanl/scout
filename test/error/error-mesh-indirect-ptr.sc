#include <iostream>

using namespace std;

struct foo{
  int* a;
  int b;
};

int main(int argc, char** argv){

  uniform mesh MyMesh{
  cells:
    foo f;
    float b;
  };

  MyMesh myMesh[512,512];

  return 0;
}
