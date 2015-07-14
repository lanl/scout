
uniform mesh MyMesh { cells: int a; int b; };

int main(int argc, char** argv) {

  MyMesh m[10]; int temp[30];

  int i = 0;
  forall cells c in m { 
    c.a = i++;
    c.b = i;
  }

  forall cells c in m { 
    temp[c.a + c.b ] = 3; 
  } 
}
