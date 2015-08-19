/*
 * ###########################################################################
 * Copyright (c) 2015, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2015. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 * ###########################################################################
 *
 * Notes
 *
 * #####
 */



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
