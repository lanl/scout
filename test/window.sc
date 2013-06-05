#include <iostream>

using namespace std;

int main(int argc, char** argv){
  window win[1024,1024] {
    background  = hsv(0.1, 0.2, 0.3);
    save_frames = true;
    filename    = "heat2d-####.png";
  };
  
  return 0;
}
