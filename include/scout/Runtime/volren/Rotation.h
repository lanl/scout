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

#ifndef _ROTATION_H_
#define _ROTATION_H_

#include "QuatVector.h" // For internal members

class Rotation
{
 public:
  Rotation();
  Rotation(const Rotation &rot);
  Rotation(float angle, float x, float y, float z);
  ~Rotation();

  //! Clear rotation.
  void clear();

  //! Set rotation according to angle and axis
  void set(float angle, float x, float y, float z);
  void set(float angle, QuatVector &axis);

  //! Accumulate rotation according to angle and axis
  void rotate(float angle, float x, float y, float z);
  void rotate(float angle, QuatVector &axis);

  //! Set rotation according to explicit quaternion specification 
  void setQuaternion(float x, float y, float z, float w);
  void setQuaternion(float vec[4]);
  void setQuaternion(const QuatVector &quat);

  //! Accumulate rotation according to explicit quaternion specification 
  void addQuaternion(float x, float y, float z, float w);
  void addQuaternion(float vec[4]);
  void addQuaternion(const QuatVector &quat);

  // Given two quaternions, add them together to get a third quaternion.
  // Adding quaternions to get a compound rotation is analagous to adding
  // translations to get a compound translation.  When incrementally
  // adding rotations, the first argument here should be the new
  // rotation, the second and third the total rotation (which will be
  // over-written with the resulting new total rotation).
  void add(const Rotation &rot);

  // ------------------- GET METHODS -----------------------

  //! Build a rotation matrix based on internal quaternion.
  void matrix(float m[16]);
  //! Build an inverse matrix based on internal quaternion.
  void invMatrix(float m[16]);

  //! Copy out internal quaternion vector
  void quaternion(float quat[4]);
  QuatVector quaternion();

  //! Apply to GL state
  void apply();
  void iapply();

 private:
  //! Normalize internal quaternion. Normalizes all 4 positions.
  void normalize();

  int m_renorm_count;
  int m_count;
  QuatVector m_quat;
  float angle;
  float axis[3];

}; // class Rotation

#endif // _ROTATION_H_
