uniform mesh MyMesh {
 cells:
  float a;
  float b;
  float c;
  float d;
};

task void MyTask(MyMesh *m) {
  forall cells l in *m {
    b = 1;
    d = 3;
  }
}

int main(int argc, char** argv) {
  MyMesh m[10];

  MyTask(&m);

  return 0;
}
