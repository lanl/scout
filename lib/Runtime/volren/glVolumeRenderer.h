#ifndef GL_VOLUME_RENDERER_H
#define GL_VOLUME_RENDERER_H

#include "vector_types.h"
#include "glRenderer.h"
#include "scout/Runtime/opengl/opengl.h"

class DataMgr;
namespace scout{
  class RenderCallback{
  public:
    virtual void render_kernel(void* output) = 0;
  };

    class glVolumeRenderer: public glRenderer
    {
    public:
        glVolumeRenderer(size_t width, size_t height, size_t depth);

      void setRenderCallback(RenderCallback* callback){
        callback_ = callback;
      }

    protected:
        void initializeGL() override;

        void resizeGL(int width, int height) override;

        void draw(float modelview[16], float projection[16]) override;

        void cleanup() override;

        virtual void saveImage(const char* filename) override;

        virtual void saveMultiImage() override;

      void init_();

    private:
      RenderCallback* callback_;

      size_t width_;
      size_t height_;
      size_t depth_;
      bool ready_;

        /*****cuda*****/
        dim3 blockSize;
        dim3 gridSize;

        /*****shading*****/
        float density = 0.05f;
        float brightness = 1.0f;
        float transferOffset = 0.0f;
        float transferScale = 1.0f;
        bool linearFiltering = true;

        GLfloat invPVM[16];



        /****opengl rendering****/
        GLuint pbo = 0;     // OpenGL pixel buffer object
        GLuint tex = 0;     // OpenGL texture object
        struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

        void updatePixelBuffer();

        void cuda_render();
    };
}
#endif //GL_VOLUME_RENDERER_H
