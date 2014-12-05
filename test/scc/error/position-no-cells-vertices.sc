
uniform mesh MyMesh {
edges:
  int a;
};

int main(int argc, char *argv[]) {

  MyMesh m[3];

  forall edges e in m {
    e.a = position().x;
  }
}
