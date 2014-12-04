#include <stdio.h>
#include <assert.h>

uniform mesh AMeshType{
cells:
  int a;
  int b;
edges:
  int x;
  int y;
  int z;
  int w;
};


int main(int argc, char *argv[])
{

  AMeshType m1[3];
  AMeshType m2[2,3];

  forall edges e in m1 {
    e.x = head().x;
    e.y = head().y;
    e.z = head().z;
    e.w = head().w;
  }

  int count = 1;
  forall edges e in m1 {
    assert(e.x == count && "d1 bad x");
    assert(e.y == 0 && "d1 bad y");
    assert(e.z == 0 && "d1 bad z");
    assert(e.w == count && "d1 bad w");
    count++;
  } 

  forall edges e in m2 {
    e.x = head().x;
    e.y = head().y;
    e.z = head().z;
    e.w = head().w;
  }

  int expx[] = {0,1,2,0,1,2,0,1,2,1,2,1,2,1,2,1,2};
  int expy[] = {1,1,1,2,2,2,3,3,3,0,0,1,1,2,2,3,3};
 
  count = 0;
  forall edges e in m2 {
    printf("%d %d %d %d\n", e.x, e.y, e.z, e.w);
    assert(e.x == expx[count] && "d2 bad x");
    assert(e.y == expy[count] && "d2 bad y");
    assert(e.z == 0 && "d2 bad z");
    assert(e.w == count && "d2 bad w");
    count++;
  }
}
