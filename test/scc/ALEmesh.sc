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

#include <stdio.h>
#include <assert.h>

ALE mesh AMeshType{
vertices:
  float foo;
};


int main(int argc, char *argv[])
{
  AMeshType m1[1];

  float sumvalx = 0;
  float sumvaly = 0;
  float sumvalz = 0;

  forall vertices v in m1{
    mpositionx(2.0); 
    sumvalx += mpositionx();
    mpositiony(3.0); 
    sumvaly += mpositiony();
    mpositionz(4.0);
    sumvalz += mpositionz();
  }

  assert(sumvalx == 4.0 && "bad mpositionx()");
  assert(sumvaly == 6.0 && "bad mpositiony()");
  assert(sumvalz == 8.0 && "bad mpositionz()");

  // could also do a test for these:
  //mpositionx('c');  //breaks, need to fix
  //mpositionx(5); //breaks, need to fix
  //mpositionx(positionx()); // breaks, can't do FP cast
  //float val = 43;
  //float valarray[3] = {2, 3, 4};
  //float* valptr = &val;
  //mpositionx(valarray[0]);  // ok
  //mpositionx(*valptr); //ok
  //mpositionx(val); //ok
  //mpositiony(val+5); //ok

  sumvalx = 0;
  sumvaly = 0;
  sumvalz = 0;

  float3 mposVector = {3, 4, 5}; 

  forall vertices v in m1{
    mposition(mposVector); 
    sumvalx += mpositionx();
    sumvaly += mpositiony();
    sumvalz += mpositionz();
  }

  assert(sumvalx == 6.0 && "bad mpositionx()");
  assert(sumvaly == 8.0 && "bad mpositiony()");
  assert(sumvalz == 10.0 && "bad mpositionz()");

  sumvalx = 0;
  sumvaly = 0;
  sumvalz = 0;

  mposVector.x = 4;
  mposVector.y = 5;
  mposVector.z = 6;

  forall vertices v in m1{
    mposition(mposVector); 
    float4 mposVector2 = mposition();  
    sumvalx += mposVector2.x;
    sumvaly += mposVector2.y;
    sumvalz += mposVector2.z;
  }

  assert(sumvalx == 8.0 && "bad mpositionx()");
  assert(sumvaly == 10.0 && "bad mpositiony()");
  assert(sumvalz == 12.0 && "bad mpositionz()");

  return 0;
}
