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

  forall cells c in m {
    a = eoshift(b, 1, -1);
  }
  return 0;
}
