
uniform mesh dataMesh {
  cells:
    double val;

};

int main () {
  dataMesh f[10];
  forall cells c in f {
    f.val=0.0;
  }
}

