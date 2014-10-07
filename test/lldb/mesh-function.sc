#include <stdio.h>

uniform mesh MyMesh {
 cells:
  float field;
};


task void MyTask(MyMesh *m) {

  forall cells c in *m {
    c.field = 9;
  }

  int x = 9;

  forall cells c in *m {
    printf("%f\n", c.field);
  }
}

int main(int argc, char** argv) {
  MyMesh m[3];
  MyTask(&m);
  return 0;
}
