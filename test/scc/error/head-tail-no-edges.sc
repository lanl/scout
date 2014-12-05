
uniform mesh MyMesh {
cells:
  int a;
  int b;
};

int main(int argc, char *argv[]) {

  MyMesh m[3];

  forall cells c in m {
    c.a = head().x;
    c.b = tail().x;
  }
}
