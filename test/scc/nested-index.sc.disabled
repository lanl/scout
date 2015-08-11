#include <assert.h>
#include <stdio.h>

uniform mesh MyMesh {
 cells:
  int a;
 vertices:
  int b;
};

int const m1size = 4;
int const m2size = 3;

int main(int argc, char** argv) {
  MyMesh m1[m1size];
  MyMesh m2[m2size, m2size];

  int out1[2*m1size];
  int exp1[] = {0,1,1,2,2,3,3,4};

  int i = 0;
  forall cells c in m1 {
    printf("out %d\n",gindexx());  
    forall vertices v in c {
      printf("in %d\n",gindexx());
      out1[i] = gindexx();  
      i++;
    }
  }

  for(int j = 0; j < 2*m1size; j++) {
    assert(out1[j] == exp1[j] && "bad value in rank=1");
  }

  int out2x[4*m2size*m2size], out2y[4*m2size*m2size];
  int exp2x[] = {0,1,0,1,1,2,1,2,2,3,2,3};
  int exp2y[] = {0,0,1,1,0,0,1,1,0,0,1,1,
                 1,1,2,2,1,1,2,2,1,1,2,2,
                 2,2,3,3,2,2,3,3,2,2,3,3};


  forall vertices v in m2 {
    b = position().x + 10*position().y;
  }

  i = 0;
  forall cells c in m2 {
    printf("out %d\n",gindexx());
    forall vertices v in c {
      printf("in %d %d v %d\n",gindexx(),gindexy(),v.b);
      out2x[i] = gindexx();  
      out2y[i] = gindexy();  
      i++;
    }
  }

 for(int j = 0; j < 4*m2size*m2size; j++) {
    assert(out2x[j] == exp2x[j%12] && "bad x value in rank=2");
    assert(out2y[j] == exp2y[j] && "bad y value in rank=2");
  }


}
