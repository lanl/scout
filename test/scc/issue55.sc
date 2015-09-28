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
#include <assert.h>
#include <stdio.h>

uniform mesh MyMesh {
 cells:
  int b;
};

int main(int argc, char** argv) {
  MyMesh m[5];
  int coutm1[5] = {4,0,1,2,3};
  int coutp1[5] = {1,2,3,4,0};
  int eoutm1[5] = {-1,0,1,2,3};
  int eoutp1[5] = {1,2,3,4,-1};


  forall cells c in m {
    b = positionx();
  }

  int i = 0;
  forall cells c in m {
    if (positionx() == 0) {
      printf("cell position 0: %d, b: %d, cshift cell-1 b: %d, cell+1 b: %d, eoshift cell-1 b: %d, cell+1 b: %d\n",
          positionx(), c.b, cshift(c.b, -1), cshift (c.b, 1), eoshift(c.b, -1, -1), eoshift(c.b, -1, 1) );
      assert(cshift(c.b, -1) ==  coutm1[i] && "bad cshift -1");
      assert(cshift(c.b, 1) ==  coutp1[i] && "bad cshift 1");
      assert(eoshift(c.b, -1, -1) ==  eoutm1[i] && "bad eoshift -1");
      assert(eoshift(c.b, -1, 1) ==  eoutp1[i] && "bad eoshift 1");

    } else {
      printf("cell position: %d, b: %d, cshift cell-1 b: %d, cell+1 b: %d, eoshift cell-1 b: %d, cell+1 b: %d\n",
          positionx(), c.b, cshift(c.b, -1), cshift (c.b, 1), eoshift(c.b, -1, -1), eoshift(c.b, -1, 1) );
      assert(cshift(c.b, -1) ==  coutm1[i] && "bad cshift -1");
      assert(cshift(c.b, 1) ==  coutp1[i] && "bad cshift 1");
      assert(eoshift(c.b, -1, -1) ==  eoutm1[i] && "bad eoshift -1");
      assert(eoshift(c.b, -1, 1) ==  eoutp1[i] && "bad eoshift 1");
    }
    i++;
  }
  return 0;
}
