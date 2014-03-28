#include <stdio.h>
#include <assert.h>

uniform mesh MyMesh {
 cells:
  float a;
 vertices:
  float b;
};

int main(int argc, char** argv) {
  MyMesh m[2,2,2];

  forall cells c in m{
    a = position().x + position().y*10 + position().z*100;
  }

  forall cells c in m{
    forall vertices v in c{
      b = a;
    }
  }

  float expected[] = 
    {0.000000f, 1.000000f, 1.000000f, 10.000000f, 
     11.000000f, 11.000000f,10.000000f, 11.000000f,
     11.000000f, 100.000000f, 101.000000f, 101.000000f,
     110.000000f, 111.000000f, 111.000000f, 110.000000f,
     111.000000f, 111.000000f, 100.000000f, 101.000000f, 
     101.000000f, 110.000000f, 111.000000f, 111.000000f,
     110.000000f, 111.000000f, 111.000000};
  
  int i = 0;
  forall vertices v in m{
    assert(b == expected[i] && "unexpected value");
    ++i;
  }
  
  return 0;
}
