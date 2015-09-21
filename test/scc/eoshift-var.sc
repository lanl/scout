# 1 "issue56.sc"
# 1 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 1 "<command-line>" 2
# 1 "issue56.sc"
uniform mesh MyMesh {
cells:
  float a;
  float b;
};


int main(int argc, char** argv) {
  MyMesh m[512];
  forall cells c in m {
    b = 1;
  }

  int x = 1;
  forall cells c in m {
    a = eoshift(b, -1.f, x);
  }
  return 0;
}
