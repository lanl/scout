/*
 * ###########################################################################
 * Copyright (c) 2014, Los Alamos National Security, LLC.
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
 * Simplistic 2D heat transfer...
 * Modified to include (quasi-)soluble bodies and constant advection term
 * Jamal Mohd-Yusof 10/20/11
 *
 * Test case which just checks that a single value is the expected value (within epsilon).
 *
 * ##### 
 */
#include <assert.h>
#include <stdio.h>

const int MESH_DIM = 512;
const int NTIME_STEPS = 100;

uniform mesh HeatMesh {
  cells: 
    float h;
    float h_next;
    float mask;
};

task void init(HeatMesh *m) {
  const int NBODIES = 5;
  const float MAX_TEMP = 100.0f;
  
  int c_x[NBODIES] = { 128, 128, 394, 394, 256 };
  int c_y[NBODIES] = { 128, 394, 128, 394, 256 };
  int r2cyl = MESH_DIM / 4;
  float u = 0.001;

  forall cells c in *m {
    h = 0.0f;
    h_next = 0.0f;
    mask = 1.0;

    if (position().y == 0 || position().y == width()-1) {
      h = MAX_TEMP;
      h_next = MAX_TEMP;
      mask = 0.0;
    } 

    for (int i = 0; i < NBODIES; i++) {
      float r2 = (position().x - c_x[i]) * (position().x - c_x[i])
          + (position().y - c_y[i]) * (position().y - c_y[i]);
      if (r2 < r2cyl) {
          mask = r2 / r2cyl;
  
	      h = MAX_TEMP;
	      h_next = MAX_TEMP;
	   }
    }
  }
}
task void work(HeatMesh *m) {
  const float dx = 10.0f / width(*m);
  const float dy = 10.0f / height(*m);
  const float alpha = 0.00001f;
  const float dt = 0.5f * (dx * dx + dy * dy) / 4.0f / alpha;
  const float u = 0.001;

  forall cells c in *m {
    float ddx = 0.5 * (cshift(h, 1, 0) - cshift(h, -1, 0)) / dx;
    float d2dx2 = cshift(h, 1, 0) - 2.0f * h + cshift(h, -1, 0);
    d2dx2 /= dx * dx;

    float d2dy2 = cshift(h, 0, 1) - 2.0f * h + cshift(h, 0, -1);
    d2dy2 /= dy * dy;

    h_next = mask * dt * (alpha * (d2dx2 + d2dy2) - mask * u * ddx)
          + h;
  }
  forall cells c in *m {
    h = h_next;
  }
}	
	
task void check(HeatMesh *m) {
  const float VALUE = 87.899254;
  forall cells c in *m {
    if (position().x == 260 && position().y == 260) {
       // if value does not match exit w/ error.
       if ((h - VALUE)*(h - VALUE) > 1e-10) {
         assert(false); 
       } 
    }
  }
}

int main(int argc, char *argv[]) {
  
  HeatMesh m[MESH_DIM, MESH_DIM];

  init(&m);
  
  // Time steps loop
  for (unsigned int t = 0; t < NTIME_STEPS; ++t) {
    work(&m);
    if(t == NTIME_STEPS-2) check(&m);
  }

  return 0;
}

