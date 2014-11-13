#include <stdio.h>

uniform mesh MyMesh{
cells:
  float a;
vertices:
  float b;
edges:
  float d;
};

void MyFunc(MyMesh *m) { 
  window mywin[512,512]; 

  forall cells c in *m {
    a = position().x;
  } 
  
  forall vertices v in *m {
    b = position().y;
  } 

  size_t j = 0;
  forall edges e in *m {
    d = j;
    ++j;
  }

  renderall vertices v in *m to mywin{
    color = hsva(b/16.0f*360.0f, 1.0, 1.0, 1.0);
  }

  renderall edges e in *m to mywin{
    color = hsva(d/j*360.0f, 1.0, 1.0, 1.0);
  }

  renderall cells c in *m to mywin{
    color = hsva(a/16.0f*360.0f, 0.5, 1.0, 1.0);
  }
    
  sleep(3);
}

int main(int argc, char** argv) {
  MyMesh m[16,16];
  MyFunc(&m);
  sleep(3);
  return 0;
}

