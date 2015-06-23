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
// $Id: QuatVector.h,v 1.2 2005/09/02 09:49:30 groth Exp $
// $Date: 2005/09/02 09:49:30 $
// $Revision: 1.2 $
//
// Authors: A. McPherson, P. McCormick, J. Kniss, G. Roth
//
//.........................................................................
// 
//! This is NOT a quaternion class. Nor is it a really complete Vector class.
//! It is the subset of a proper 4-element vector class needed for quaternion
//! operations. It is not anticipated it will be used outside of the Rotation
//! class. It couldn't be Vector, because name was taken.
//!
//
//.........................................................................
//

#ifndef _QUAT_VECTOR_H_
#define _QUAT_VECTOR_H_

class QuatVector
{
 public:

  QuatVector();
  QuatVector(float x, float y, float z, float w = 1.0);
  QuatVector(float v[4]);
  ~QuatVector();

  void set(float x, float y, float z, float w = 1.0);
  void set(const float src[4]);

  void copy(const QuatVector &src);
  void add(const QuatVector &vec1, const QuatVector &vec2);
  void sub(const QuatVector &vec1, const QuatVector &vec2);
  float dot(const QuatVector &vec2) const;
  void cross(const QuatVector &vec1, const QuatVector &vec2);

  void scale(float factor);
  void normalize();

  float length();
  bool coincident(const QuatVector &vec);

  //! Return value located at passed index
  //! No error checking. There are only four values. Don't be stupid.
  inline float &operator[](int i)
    { return m_vec[i]; }

  inline float operator[](int i) const
    { return m_vec[i]; }

 private:
  float m_vec[4];

}; // class QuatVector

#endif // _QUAT_VECTOR_H_
