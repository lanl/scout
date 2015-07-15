#include "Tracer.h"
//#include "DataMgr.h"

using namespace scout;

void Tracer::AllocOutImage() {
    if(h_output != NULL)
        delete [] h_output;

    h_output = new uint[winWidth * winHeight];
}

Tracer::~Tracer() {
    if(h_output != NULL)
        delete [] h_output;
}

void Tracer::resize(int width, int height) {
    winWidth = width;
    winHeight = height;
    AllocOutImage();
}

void Tracer::GetDataDim(int &nx, int &ny, int &nz) {
 //   dataMgr->GetDataDim(nx, ny, nz);
    nx = width_;
    ny = height_;
    nz = depth_;
}
