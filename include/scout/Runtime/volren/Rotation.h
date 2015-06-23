//
//     _____                   __  
//    / ___/__________  __  __/ /_ 
//    \__ \/ ____/ __ \/ / /   __/
//   ___/   /___/ /_/ / /_/ / /__ 
//  /____/\____/\____/\____/\___/
//
//       Visualization Team 
//     Advanced Computing Lab 
//  Los Alamos National Laboratory
// --------------------------------
// 
// $Id: Rotation.h,v 1.3 2005/09/02 09:49:30 groth Exp $
// $Date: 2005/09/02 09:49:30 $
// $Revision: 1.3 $
//
// Authors: A. McPherson, P. McCormick, J. Kniss, G. Roth
//
//.........................................................................
// 
//! Abstraction representing rotations of geometry
//!
//! A thinly-disguised quaternion class. I refrained from naming it accordingly
//! with the idea that how the rotation is actually implemented shouldn't
//! matter to anyone using this class. It can spit out the quaternion, or the
//! corresponding transformation matrix.
//! Rotations can be set explicitly in every conceivable format. 
//! That includes angles and axes, explicit quaternions, or rotation matrices.
//! Rotations can also be accumulated. Setting a rotation will add that
//! to the currently-held rotation. This is the state that was formerly encoded
//! into Trackball. To avoid requiring each View to have its own trackball,
//! which threw a wrench in our organization, this was excised from that class,
//! leaving it rather desolate.
//!
//! NOTE: Everything is in radians!
//!
//
//.........................................................................
//

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
