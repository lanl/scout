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
// $Id: QuatVector.cpp,v 1.2 2005/09/02 09:49:30 groth Exp $
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
//! and Trackball classes. It couldn't be Vector, because that name was taken.
//!
//
//.........................................................................
//

#include "QuatVector.h"  // Class definition
#include <cmath>         // for sqrt()

QuatVector::QuatVector()
{
  m_vec[0] = m_vec[1] = m_vec[2] = 0.0;
  m_vec[3] = 1.0;

} // QuatVector()

QuatVector::QuatVector(float v[4])
{
  set(v);

} // QuatVector()

QuatVector::QuatVector(float x, float y, float z, float w)
{
  set(x, y, z, w);

} // QuatVector()

QuatVector::~QuatVector()
{

} // QuatVector()

void QuatVector::set(float x, float y, float z, float w)
{
  m_vec[0] = x;
  m_vec[1] = y;
  m_vec[2] = z;
  m_vec[3] = w;

} // set()

void QuatVector::set(const float src[4])
{
  m_vec[0] = src[0];
  m_vec[1] = src[1];
  m_vec[2] = src[2];
  m_vec[3] = src[3];

} // set()

void QuatVector::copy(const QuatVector &src)
{
  set(src.m_vec);

} // copy()

void QuatVector::add(const QuatVector &vec1, const QuatVector &vec2)
{
  m_vec[0] = vec1[0] + vec2[0];
  m_vec[1] = vec1[1] + vec2[1];
  m_vec[2] = vec1[2] + vec2[2];

} // add()

void QuatVector::sub(const QuatVector &vec1, const QuatVector &vec2)
{
  m_vec[0] = vec1[0] - vec2[0];
  m_vec[1] = vec1[1] - vec2[1];
  m_vec[2] = vec1[2] - vec2[2];

} // sub()

//! Note only three components are used
float QuatVector::dot(const QuatVector &vec2) const
{
  return m_vec[0]*vec2[0] + m_vec[1]*vec2[1] + m_vec[2]*vec2[2];

} // dot()

void QuatVector::cross(const QuatVector &vec1, const QuatVector &vec2)
{
  m_vec[0] = (vec1[1] * vec2[2]) - (vec1[2] * vec2[1]);
  m_vec[1] = (vec1[2] * vec2[0]) - (vec1[0] * vec2[2]);
  m_vec[2] = (vec1[0] * vec2[1]) - (vec1[1] * vec2[0]);

} // cross()

float QuatVector::length()
{
  return sqrt(m_vec[0] * m_vec[0] + m_vec[1] * m_vec[1] + m_vec[2] * m_vec[2]);

} // length()

void QuatVector::scale(float factor)
{
  m_vec[0] *= factor;
  m_vec[1] *= factor;
  m_vec[2] *= factor;

} // scale()

void QuatVector::normalize()
{
  float len = length();
  if(len > 0)
    scale(1.0/len);
    
} // normalize()

//! Determines if vector is coincident to this
bool QuatVector::coincident(const QuatVector &vec2)
{
  return(m_vec[1] * vec2[2] == m_vec[2] * vec2[1] &&
	 m_vec[2] * vec2[0] == m_vec[0] * vec2[2] &&
	 m_vec[0] * vec2[1] == m_vec[1] * vec2[0]);

} // coincident()
