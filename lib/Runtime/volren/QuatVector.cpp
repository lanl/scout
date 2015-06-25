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
