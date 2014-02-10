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
 * Simplistic 1D heat transfer...
 *
 * ##### 
 */ 

int main(int argc, char *argv[])
{
  const int NTIME_STEPS     = 300;
  const float MAX_TEMP      = 100.0f;
  
  uniform mesh HeatMeshType{
    cells:
      float t1, t2;
  };

  HeatMeshType heat_mesh[1024];

  // Set initial conditions.
  //
  // A nice shorthand for the forall construct above might be
  // something like this (stealing from Fortran):
  // 
  //  heat_mesh::cells.t1[0:1023:1023] = 0.0f;
  //  heat_mesh::cells.t1[1:1022] = 100.0f;
  forall cells c in heat_mesh {
    if (position().x > 0 && position().x < 1023)
      t1 = 0.0f;
    else
      t1 = MAX_TEMP;      
  }

  const float therm_conduct = 0.445f;
  const float spec_heat     = 0.113f;
  const float rho           = 7.8f;

  float mat_const = therm_conduct / (spec_heat * rho);

  // Time steps loop. 
  for(unsigned int t = 0; t < NTIME_STEPS; ++t) {

    // The 'position' attribute of a cell is automatically
    // provided to contain the coordinates of the current 
    // cell being processed. 
    //
    // cshift (circular shift) is a built-in function (part of the 
    // standard library) that we use for access to neighboring 
    // cells in the mesh.  For now we're only working with uniform 
    // meshes.  This is like F90 but we shift index values vs. 
    // array duplication (which is a horrid feature in most F90 
    // runtimes -- no wonder the data-parallel features are never
    // used).
    forall cells c in heat_mesh {
      if (position().x > 0 && position().x < 1023) {
        t2 = t1 + mat_const * (cshift(c.t1,1) + cshift(c.t1,-1) - 2.0f * t1);
      }
    }

    forall cells c in heat_mesh {
      t1 = t2;
    }

    renderall cells c in heat_mesh {
      // Normalize temperatures ranges into the 0...1.0 range and then map into
      // HSV color space with hue running from blue to red for cold to hot. 
      float norm_t1 = t1 / MAX_TEMP;
      float hue = 240.0f - 240.0f * norm_t1;
      color = hsv(hue, 1.0f, 1.0f);
    }
  }

  return 0;
}
