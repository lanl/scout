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

int const m2size = 3;


int main(int argc, char** argv) {
  MyMesh m2[m2size, m2size];

  int const s2 = 4*m2size*m2size;
  int out2x[s2], out2y[s2];

  //initialize
  for (int i = 0; i < s2; i++) {
    out2x[i] = 0;
    out2y[i] = 0;
  }

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

  // I created this so there can be a conditional before the inner forall
  forall vertices v in m2 {
    b = position().x + 10*position().y;
  }

  int i = 0;

#if 0 
  // this forall works 
  forall vertices v in m2 {
    printf("out %d %d\n",positionx(),positiony());
      forall cells c in v{
        printf("in %d %d v %d\n",positionx(),positiony(),c.a);
        out2x[i] = position().x;  
        out2y[i] = position().y;  
        i++;
      }
    }
#endif

  // this forall doesn't work
  forall vertices v in m2 {
    printf("out %d %d\n",positionx(),positiony());

    // odds
    if (v.b % 2) {
      forall cells c in v{
        printf("in %d %d %d %d v %d\n",gindexx(),gindexy(),positionx(),positiony(),c.a);
        out2x[i] = position().x;  
        out2y[i] = position().y;  
        i++;
      }
    }
    // 0 and evens 
    if ((v.b+1) % 2) {
      forall cells c in v{
        printf("in2 %d %d %d %d v %d\n",gindexx(),gindexy(),positionx(),positiony(),c.a);
        out2x[i] = position().x;  
        out2y[i] = position().y;  
        i++;
      }
    }
  }

  // check for negatives
  for(int j = 0; j < s2; j++) {
    assert(out2x[j] >= 0 && "negative x value in rank=2");
    assert(out2y[j] >= 0 && "negative y value in rank=2");
  }

  // check for hitting all inner cells correctly
  for(int j = 0; j < s2; j++) {
    assert(out2x[j] == exp2x[j] && "bad x value in rank=2");
    assert(out2y[j] == exp2y[j] && "bad y value in rank=2");
  }

 return 0;
}


