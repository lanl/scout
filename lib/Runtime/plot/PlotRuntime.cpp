/*
 * ###########################################################################
 * Copyright (c) 2015, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
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

#include <vector>
#include <cassert>

#include <Rcpp.h>
#include <RInside.h>

using namespace std;
using namespace Rcpp;

namespace{

  static const int ELEMENT_INT32 = 0;
  static const int ELEMENT_INT64 = 1;
  static const int ELEMENT_FLOAT = 2;
  static const int ELEMENT_DOUBLE = 3;

  class Plot{
  public:
    Plot()
      : r_(0, 0){

      r_.parseEvalQ("library(ggplot2)");

#ifdef __APPLE__
      r_.parseEvalQ("quartz()");
#else
      r_.parseEvalQ("x11()");
#endif
    }

    void plot(void* data, size_t count, int elementKind){
      switch(elementKind){
      case ELEMENT_INT32:{
        int32_t* d = (int32_t*)data;
        r_["values"] = vector<int32_t>(d, d + count);
        break;
      }
      case ELEMENT_INT64:{
        int64_t* d = (int64_t*)data;
        r_["values"] = vector<int64_t>(d, d + count);
        break;
      }
      case ELEMENT_FLOAT:{
        float* d = (float*)data;
        r_["values"] = vector<float>(d, d + count);
        break;
      }
      case ELEMENT_DOUBLE:{
        double* d = (double*)data;
        r_["values"] = vector<double>(d, d + count);
        break;
      }
      default:
        assert(false && "invalid element kind");
      }

      r_.parseEvalQ("qplot(values)");
      r_.parseEvalQ("print(last_plot())");
    }

  private:
    RInside r_;
  };
  
  Plot* _plot = 0;

} // end namespace

extern "C"
void __scrt_plot(void* data, size_t count, int elementKind){
  if(!_plot){
    _plot = new Plot;
  }

  _plot->plot(data, count, elementKind);
}
