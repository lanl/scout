#ifndef CUDA_VOLUME_TRACER_H
#define CUDA_VOLUME_TRACER_H

#include "vector_types.h"
#include "VolumeTracer.h"
//#include "scout/Runtime/opengl/opengl.h"



class DataMgr;
namespace scout{
    class CudaVolumeTracer: public VolumeTracer
    {
    public:
        CudaVolumeTracer(size_t width, size_t height, size_t depth)
            :VolumeTracer(width, height, depth) {}

//        uint* d_output = NULL;
    protected:
//        void initializeGL() override;

//        void resizeGL(int width, int height) override;

//        void draw(float modelview[16], float projection[16]) override;

//        void cleanup() override;

//        virtual void saveMultiImage() override;

        virtual void init() override;

        virtual void draw(float modelview[16], float projection[16]) override;

        virtual void cleanup() override;

        virtual void resize(int width, int height) override;


    private:
        /*****cuda*****/
        dim3 blockSize;
        dim3 gridSize;

//        /*****shading*****/
//        float density = 0.05f;
//        float brightness = 1.0f;
//        float transferOffset = 0.0f;
//        float transferScale = 1.0f;
//        bool linearFiltering = true;


//        //CUDA resource
//        struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)


//        void cuda_render(GLfloat invPVM[16]);
    };
}
#endif //CUDA_VOLUME_TRACER_H
