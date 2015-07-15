#include <stdio.h>

uniform mesh vecMesh {
  cells:
    double a;
};

void func() {
  const int n = 10;
  vecMesh vec_mesh[n];
}

void dispMesh(vecMesh *m) {
  forall cells c in *m {
    printf("%f\n", c.a);
  }
}

int main() {
  return 0;
}
