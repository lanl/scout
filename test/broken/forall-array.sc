#include <iostream>

using namespace std;

int main(int argc, char** argv){
  
  int f[100];
  int g[100][100][100];

  forall i in [::]{
    f[i] = i * 2;
  }

  forall i in [10::]{
    f[i] = i * 2;
  }

  forall i in [10:20:]{
    f[i] = i * 2;
  }

  forall i in [10:20:3]{
    f[i] = i * 2;
  }

  forall i in [::9]{
    f[i] = i * 2;
  }

  forall i in [1::9]{
    f[i] = i * 2;
  }

  forall i in [:99:]{
    f[i] = i * 2;
  }

  forall i,j,k in [::,::,::]{
    g[i][j][k] = i * 2 + j * k;
  }

  forall i,j,k in [::,::9,10:20:3]{
    g[i][j][k] = i * 2 + j * k;
  }

  return 0;
}

