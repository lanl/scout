#include <runtime/renderall.h>

#include <runtime/framebuffer.h>
#include <runtime/viewport.h>

#include <cmath>
#include <algorithm>

using namespace scout;

inline static float4 lerp(float x, const float4 &a, const float4 &b)
{
    float4 vecx = {x, x, x, x};
    float4 res;
    res.vec = a.vec + (vecx.vec * (b.vec - a.vec));
    return res;
}

inline float4 bilinearSample(const float4 *buffer, int wd, int ht, float x,
                             float y)
{
    int xl = x;
    int xu = xl + ((xl < wd - 1) ? 1 : 0);
    int yl = y;
    int yu = yl + ((yl < ht - 1) ? 1 : 0);

    x -= floor(x);
    y -= floor(y);

    return lerp(y, lerp(x, buffer[yl*wd + xl], buffer[yl*wd + xu]),
                   lerp(x, buffer[yu*wd + xl], buffer[yu*wd + xu]));
}

void mapToFrameBuffer(const float4 *colors, int dataw, int datah, 
                      framebuffer_rt &fb, const viewport_rt &vp, 
                      MapFilterType filter) 
{
    float wratio = vp.width/float(dataw);
    float hratio = vp.height/float(datah);
    float factor = std::min(wratio, hratio);

    float bboxx1 = floor((vp.width - (factor * float(dataw)))/2.0f);
    float bboxy1 = floor((vp.height - (factor * float(datah)))/2.0f);
    float bboxx2 = ceil(bboxx1 + (factor * float(dataw)));
    float bboxy2 = ceil(bboxy1 + (factor * float(datah)));

    // handle special case of linear minification filter
    if (filter == FILTER_LINEAR && factor < 1.0f) {
        for (float dy = bboxy1; dy < bboxy2; dy += 1.0f) {
            for (float dx = bboxx1; dx < bboxx2; dx += 1.0f) {
                float4 &dst = fb.pixels[int(dy)*fb.width + int(dx)];
           
                float wt = 1.0f/factor;
                float sx = (dx - bboxx1) * wt;
                float sy = (dy - bboxy1) * wt;

                float4 avg;
                avg.components[0] = avg.components[1] = avg.components[2] =
                avg.components[3] = 0.0f;
                float div = 0.0f;
                for (float y = sy; y < sy + wt; y += 1.0f) {
                    for (float x = sx; x < sx + wt; x += 1.0f, div += 1.0f) {
                        float4 samp = bilinearSample(colors, dataw, datah, x,
                                                     y);
                        avg.vec += samp.vec;
                    }
                }

                avg.components[0] /= div;
                avg.components[1] /= div;
                avg.components[2] /= div;
                avg.components[3] /= div;

                dst = avg;
            }
        }

        return;
    }
    // for all other cases
    for (float dy = bboxy1 + 0.5f; dy < bboxy2; dy += 1.0f) {
        for (float dx = bboxx1 + 0.5f; dx < bboxx2; dx += 1.0f) {
            float4 &dst = fb.pixels[int(dy)*fb.width + int(dx)];
           
            float sx = (dx - bboxx1)/factor;
            float sy = (dy - bboxy1)/factor;
            
            switch (filter) {
                case FILTER_NEAREST:
                    dst = colors[int(sy)*dataw + int(sx)];
                    break;
                case FILTER_LINEAR:
                    dst = bilinearSample(colors, dataw, datah, sx, sy);
                    break;
            }
        }
    }
}

