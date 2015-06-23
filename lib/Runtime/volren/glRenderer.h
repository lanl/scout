#ifndef GL_RENDERER_H
#define GL_RENDERER_H

#include <vector>

//#include "scout/Runtime/opengl/opengl.h"

class DataMgr;
namespace scout{

    class glRenderer
    {
    public:
      glRenderer(size_t width, size_t height, size_t depth);
      ~glRenderer();

//        void SetDataDim(int nx, int ny, int nz);

        void GetDataDim(int &nx, int &ny, int &nz);

        void SetWindowSize(int w, int h) {winWidth = w; winHeight = h;}

        virtual void initializeGL() = 0;

        virtual void resizeGL(int width, int height) {winWidth = width; winHeight = height;}

        virtual void draw(float modelview[16], float projection[16]) = 0;

        virtual void cleanup() = 0;

        virtual void saveImage(const char* filename){}

        virtual void saveMultiImage(){}

        void SetDataMgr(DataMgr* ptr) {dataMgr = (DataMgr*)ptr;}

    protected:
//        int dataDim[3];
        DataMgr *dataMgr;

        int winWidth, winHeight;
      size_t width_;
      size_t height_;
      size_t depth_;
    };
}
#endif //GL_MESH_RENDERER_H
