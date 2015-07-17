#ifndef VOLUME_TRACER_H
#define VOLUME_TRACER_H

#include "Tracer.h"

class DataMgr;
namespace scout{
    class VolumeTracer: public Tracer
    {
    public:
        VolumeTracer(size_t width, size_t height, size_t depth)
            :Tracer(width, height, depth) {}

    protected:
        /*****shading*****/
        float density = 0.05f;
        float brightness = 1.0f;
        float transferOffset = 0.0f;
        float transferScale = 1.0f;
        bool linearFiltering = true;

    };
}
#endif //GL_VOLUME_RENDERER_H
