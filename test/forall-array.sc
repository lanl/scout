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
 *
 * ##### 
 */ 
 #include <iostream>

int main(int argc, char** argv) {
  
  int a[10];
  int b[10][10];
  int c[10][10][10];

  // default start and stride 
  forall i in [:10:] { // 0 to 9
    a[i] = i * 2;
  }

  // check default start and stride
  for(size_t i = 0; i < 10; i++) {
    if(a[i] != i*2 ) return -1; 
  }
  
  // default start and stride 3
  forall i in [:10:3] { // 0,3,6,9
    a[i] = i * 10;
  }
  
  // check default start and stride 3
  for(size_t i = 0; i < 10; i++) {	
    if (i % 3 == 0) {
      if(a[i] != i*10 ) return -1; 
    } else {
      if(a[i] != i*2 ) return -1; 
    }
    a[i] = 0;
  } 

  // start 2 and default stride
  forall i in [2:10:] { // 2 to 9
    a[i] = i * 3;
  }

  // check start 2 and default stride
  for(size_t i = 0; i < 10; i++) {
    if(i < 2) {
      if(a[i] != 0 ) return -1; 
    } else {
      if(a[i] != i*3 ) return -1; 
    }
    a[i] = 0;
 }

  // start 2 and stride 2
  forall i in [2:10:2] { // 2,4,6,8
    a[i] = i * 4;
  }

  // check start 2 and stride 2
  for(size_t i = 0; i < 10; i++) {	
    if (i % 2 == 0) {
      if(a[i] != i*4 ) return -1;
    } else {
      if(a[i] != 0 ) return -1; 
    }
    a[i] = 0;
  } 

  // default start and stride 2D
  forall i,j in [:10:,:10:] { // 0 to 9
    b[i][j] = i * j * 2;
  }
  
  // check default start and stride 2D
  for(size_t i = 0; i < 10; i++) {
    for(size_t j = 0; j < 10; j++) {
      if(b[i][j] != i*j*2 ) return -1;
    }
  }
 
   // default start and stride 3 2D
  forall i,j in [:10:3,:10:3] { // 0,3,6,9
    b[i][j] = i * j * 10;
  }
	
  // check default start and stride 3 2D
  for(size_t i = 0; i < 10; i++) {	
    for(size_t j = 0; j < 10; j++) {
      if (i % 3 == 0 && j % 3 == 0) {
        if(b[i][j] != i*j*10 ) return -1; 
      } else {
        if(b[i][j] != i*j*2 ) return -1; 
      }
      b[i][j] = 0;
    }
  }   

  // start 2 and stride 2 2D
  forall i,j in [2:10:2,2:10:2] { // 2,4,6,8
    b[i][j] = i * j * 4;
  }

  // check start 2 and stride 2 2D
  for(size_t i = 0; i < 10; i++) {	
    for(size_t j = 0; j < 10; j++) {
      if (i % 2 == 0 && j % 2  == 0) {
        if(b[i][j] != i*j*4 ) return -1;
      } else {
        if(b[i][j] != 0 ) return -1; 
      }
      b[i][j] = 0;
    }  
  } 
	
  // default start and stride 3D
  forall i,j,k in [:10:,:10:,:10:] { // 0 to 9
    c[i][j][k] = i * j * k * 2;
  }
  
  // check default start and stride 3D
  for(size_t i = 0; i < 10; i++) {
    for(size_t j = 0; j < 10; j++) {
      for(size_t k = 0; k < 10; k++) {
        if(c[i][j][k] != i*j*k*2 ) return -1;
      }  
    }
  }	
	
} 
