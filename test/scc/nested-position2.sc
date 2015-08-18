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

#include <stdio.h>
#include <assert.h>

uniform mesh MyMesh {
 cells:
  int a;
 vertices:
  int b;
};

int const m1size = 4;
int const m2size = 3;
int const m3size = 2;

int main(int argc, char** argv) {
  MyMesh m1[m1size];
  MyMesh m2[m2size, m2size];
  MyMesh m3[m3size, m3size, m3size];

  int out1[2*m1size];
  int exp1[] = {1,0,1,0,1,0,1,0};

  forall cells c in m1 {
    a = position().x;
  }

  int i = 0;
  forall vertices v in m1 {
      printf("out %d\n",positionx());  
    forall cells c in v {
      printf("in %d v %d\n",positionx(),a);  
      out1[i] = position().x;  
      i++;
    }
  }

  for(int j = 0; j < 2*m1size; j++) {
    assert(out1[j] == exp1[j] && "bad value in rank=1");
  }


  int const s2 = 4*m2size*m2size;
  int out2x[s2], out2y[s2];
  int exp2x[] = {1,0,1,0,1,0,
                 1,1,0,1,0,1,0,1,0,1,0,0,
                 1,1,0,1,0,1,0,1,0,1,0,0,
                 1,0,1,0,1,0};
  int exp2y[] = {1,1,1,1,1,1,
                 0,1,0,0,1,1,0,0,1,1,0,1,
                 0,1,0,0,1,1,0,0,1,1,0,1,
                 0,0,0,0,0,0};

  forall cells c in m2 {
    a = position().x + 10*position().y;
  }

  i = 0;
  forall vertices v in m2 {
    printf("out %d %d\n",positionx(),positiony());
    forall cells c in v{
      printf("in %d %d v %d\n",positionx(),positiony(),c.a);
      out2x[i] = position().x;  
      out2y[i] = position().y;  
      i++;
    }
  }

  for(int j = 0; j < s2; j++) {
    assert(out2x[j] == exp2x[j] && "bad x value in rank=2");
    assert(out2y[j] == exp2y[j] && "bad y value in rank=2");
  }

  int const s3 = 8*m3size*m3size*m3size;
  int out3x[s3], out3y[s3],out3z[s3];

  forall cells c in m3 {
    a = position().x + 10*position().y + 100*position().z;
  }

  i = 0;
  forall vertices v in m3 {
      printf("out %d %d %d\n",positionx(),positiony(),positionz());  
    forall cells c in v {
      printf("in %d %d %d %d v %d\n",positionx(),positiony(),positionz(),position().w,c.a); 
      out3x[i] = position().x;  
      out3y[i] = position().y;  
      out3z[i] = position().z;  
      i++; 
    }
  }

#if 0
  for(int j = 0; j < s3; j++) {
    assert(out3x[j] == exp3x[j] && "bad x value in rank=3");
    assert(out3y[j] == exp3y[j] && "bad y value in rank=3");
    assert(out3z[j] == exp3z[j] && "bad z value in rank=3");
  }
#endif
 return 0;
}


