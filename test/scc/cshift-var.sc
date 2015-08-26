uniform mesh testmesh{

cells:
  double val_new;
  double val_old;
  int l_shift;

};//create a mesh with old and
//new vals and an int field for
//the right and left shifts


void setMesh(testmesh* tmesh){

  forall cells c in *tmesh{

    c.val_old=0.0;
    c.val_new=0.0;
    c.l_shift=-1;

  }
}//set the val values to 0.0
//l_shift is -1 and r_shift is 1

void test_cshift(testmesh* tmesh){
  int a=1;
  forall cells c in *tmesh{

    c.val_new=cshift(c.val_old,c.l_shift);
    c.val_new=cshift(c.val_old, a);
  }
}

int main(){

  testmesh AMesh[10];

  setMesh(&AMesh);
  test_cshift(&AMesh);

  return 0;
}
