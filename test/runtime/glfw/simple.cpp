
#include <iostream>
#include "scout/Runtime/opengl/glfw/glfwDevice.h"
//#include "scout/Runtime/opengl/glQuadRenderableVA.h"
#include <unistd.h>

using namespace std;
using namespace scout;

int main(int argv, char** argc) {
  
  glDevice* glDevice_ = glfwDevice::Instance();
  
  if (glDevice_) {
    glWindow* glWindow_ = glDevice_->createWindow(100, 50);
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // show empty buffer
    glWindow_->swapBuffers();
    
    sleep(1);
    
    bool done = glWindow_->processEvent();
    
    if (done) {
      glDevice_->deleteWindow(glWindow_);
    }
    
    
    printf("Done: %d\n", done?1:0);
  }
  
  return 0;
  
}

