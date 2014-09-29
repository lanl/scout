#include <stdio.h>

uniform mesh MyMesh {
 cells:
  float a;
  float b;
};

task void MyTask1(MyMesh *m) {
  forall cells c in *m {
    printf("%f\n", a);
  }
}

task void MyTask2(MyMesh *m) {
  forall cells c in *m {
    printf("%lx\n", (long)&a);
  }
}

int main(int argc, char** argv) {
  MyMesh m[10];

  MyTask1(&m);
  MyTask2(&m);

  return 0;
}
