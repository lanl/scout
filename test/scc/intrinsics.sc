/*
 * ###########################################################################
 * Copyright (c) 2013, Los Alamos National Security, LLC.
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
#include <assert.h>
#include <stdio.h>

#define W 3
#define H 4
#define D 5

uniform mesh MyMesh3 {
 cells:
  float a,b;
  int px, py, pz, pw;
  int w,h,d;
};

uniform mesh MyMesh2 {
 cells:
  float a,b;
  int px, py, pz, pw;
  int w,h,d;
};

int main(int argc, char** argv){

  MyMesh3 m[W,H,D];
  MyMesh2 n[W,H]; 
 
  forall cells c in m {
    a = 1.0f;
    b = 2.0f;
    px = positionx();
    py = positiony();
    pz = positionz();
    pw = positionw();
    w = width();
    h = height();
    d = depth();    
  }

  forall cells c in m {
    c.a += c.b;
    assert(px == position().x && "bad PositionX");
    assert(py == position().y && "bad PositionY");
    assert(pz == position().z && "bad PositionZ");
    assert(pw == position().w && "bad PositionW");
    assert(w == W && "bad width");
    assert(h == H && "bad height");
    assert(d == D && "bad depth");
    assert(w == width() && "bad width");
    assert(h == height() && "bad height");
    assert(d == depth() && "bad depth");
  }

  forall cells c in n {
    a = 1.0f;
    b = 2.0f;
    px = positionx();
    py = positiony();
    pz = positionz();
    pw = positionw();
    w = width();
    h = height();
    d = depth();
  }

  forall cells c in n {
    c.a += c.b;
    assert(px == position().x && "bad PositionX");
    assert(py == position().y && "bad PositionY");
    assert(pz == position().z && "bad PositionZ");
    assert(position().z == 0 && "bad PositionZ");
    assert(positionz() == 0 && "bad PositionZ");
    assert(pz == 0 && "bad PositionZ");
    assert(pw == position().w && "bad PositionW");
    assert(w == W && "bad width");
    assert(h == H && "bad height");
    assert(d == 1 && "bad depth");
    assert(w == width() && "bad width");
    assert(h == height() && "bad height");
    assert(d == depth() && "bad depth");
  }

  return 0;
}
