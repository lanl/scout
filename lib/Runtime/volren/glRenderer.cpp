#include "glRenderer.h"

using namespace scout;
//void glRenderer::SetDataDim(int nx, int ny, int nz)
//{
//    dataDim[0] = nx;
//    dataDim[1] = ny;
//    dataDim[2] = nz;
//}

glRenderer::glRenderer(size_t width, size_t height, size_t depth)
  : width_(width),
    height_(height),
    depth_(depth){

}

glRenderer::~glRenderer()
{

}

void glRenderer::GetDataDim(int &nx, int &ny, int &nz)
{
  nx = width_;
  ny = height_;
  nz = depth_;
}
