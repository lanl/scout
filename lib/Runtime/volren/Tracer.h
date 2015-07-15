#ifndef TRACER_H
#define TRACER_H
#include "QMatrix4x4"
#include <iostream>


#define USE_PBO 1
inline void GetInvPVM(float modelview[16], float projection[16], float invPVM[16])
{
    QMatrix4x4 q_modelview(modelview);
    q_modelview = q_modelview.transposed();

    QMatrix4x4 q_projection(projection);
    q_projection = q_projection.transposed();

    QMatrix4x4 q_invProjMulView = (q_projection * q_modelview).inverted();

    q_invProjMulView.copyDataTo(invPVM);
}

inline void GetNormalMatrix(float modelview[16], float NormalMatrix[9])
{
    QMatrix4x4 q_modelview(modelview);
    q_modelview = q_modelview.transposed();

    q_modelview.normalMatrix().copyDataTo(NormalMatrix);
}

class DataMgr;
namespace scout{
    class RenderCallback{
    public:
      virtual void render_kernel(void* output, float* invMat) = 0;
    };


    class Tracer
    {
    public:
        Tracer(size_t width, size_t height, size_t depth)
            :width_(width), height_(height), depth_(depth) {}
        ~Tracer();

        virtual void init() = 0;

        virtual void resize(int width, int height);

        virtual void draw(float modelview[16], float projection[16]) = 0;

        virtual void cleanup() = 0;

        virtual void saveImage(const char* filename){}

        virtual void saveMultiImage(){}

        void SetDataMgr(DataMgr* ptr) {dataMgr = (DataMgr*)ptr;}

        uint* GetHostImage(){ return h_output;}

        void SetDeviceImage(uint* p){ d_output = p;}

        void SetPBO(uint v) {pbo = v;}

        void GetDataDim(int &nx, int &ny, int &nz);

        void SetWindowSize(int w, int h) {winWidth = w; winHeight = h;}

        void setRenderCallback(RenderCallback* callback){
          callback_ = callback;
         }


    protected:
        DataMgr *dataMgr;
        RenderCallback* callback_;
        int winWidth, winHeight;
        size_t width_;
        size_t height_;
        size_t depth_;
        uint pbo;
        uint* d_output = NULL;
        uint* h_output = NULL;

        bool initialized = false;

    private:
        void AllocOutImage();
    };
}
#endif //TRACER_H
