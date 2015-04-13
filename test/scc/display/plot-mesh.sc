#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <stdlib.h>

#define WIDTH 16
#define HEIGHT 16

uniform mesh MyMesh{
 cells:
  float a;
  float b;
};

int main(int argc, char** argv){
  MyMesh m[WIDTH, HEIGHT];

  window win[1024, 1024];

  float size = 5.0;
  
  forall cells c in m {
    a = size;
    b = 1;
    size += 0.1;
  }

  with m in win plot{
    points: {position:[rowIndex % WIDTH + 0.5, rowIndex / HEIGHT + 0.5],
                 size: a},

      line: {start:[rowIndex % WIDTH, rowIndex / HEIGHT],
               end:[rowIndex % WIDTH + 1.0, rowIndex / HEIGHT]
            },

      line: {start:[rowIndex % WIDTH, rowIndex / HEIGHT],
               end:[rowIndex % WIDTH, rowIndex / HEIGHT + 1.0]
            },

      line: {start:[WIDTH, 0], end:[WIDTH, HEIGHT]},

      line: {start:[0, HEIGHT], end:[WIDTH, HEIGHT]}
  } 

  sleep(2);
  
  return 0;
}
