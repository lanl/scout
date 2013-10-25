/*
 * ###########################################################################
 * Copyright (c) 2010, Los Alamos National Security, LLC.
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
 *  Variable size mesh
 *
 * ##### 
 */ 

#include <stdio.h>

int main(int argc, char *argv[])
{
  int dim = 2;

  uniform mesh HeatMeshType{
  cells:
    float t1, t2;
  };

  printf("mesh dims: %d, %d, %d \n", dim+2, 3, dim);

  HeatMeshType heat_mesh[dim+2,3,dim];

  float outfield[(dim+2)*3*dim];
  int expected[] = {0,100,200,300,10,110,210,310,20,120,220,320,1,101,201,301,11,111,211,311,21,121,221,321};

  for (int i = 0; i < (dim+2)*3*dim; i++) {
      outfield[i] = -1.0;
  }

<<<<<<< HEAD
  forall cells c of heat_mesh {
=======
  forall cells c in heat_mesh {
>>>>>>> 331f45ad55fb625f198d765bff49b3d4fc0a6ce5
    float val = (float)(c.position.x*100 + c.position.y*10 + c.position.z);
    t1 = val;
    int index = (c.position.z*3 + c.position.y)*(dim+2) + c.position.x;
    outfield[index] = val;
  }

<<<<<<< HEAD
  forall cells c of heat_mesh {
=======
  forall cells c in heat_mesh {
>>>>>>> 331f45ad55fb625f198d765bff49b3d4fc0a6ce5
    t2 = t1;
  }
  
  for (int i = 0; i < (dim+2)*3*dim; i++) {
    printf("outfield[%d] = %f\n", i, outfield[i]);
    if((expected[i]-outfield[i])*(expected[i]-outfield[i]) > 1e-10) return -1;
  }

  return 0;
}

