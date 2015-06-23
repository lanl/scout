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
// $Id: Trackball.h,v 1.1 2005/07/22 18:27:30 groth Exp $
// $Date: 2005/07/22 18:27:30 $
// $Revision: 1.1 $
//
// Authors: A. McPherson, P. McCormick, J. Kniss, G. Roth
//
//.........................................................................
// 
//! Initially part of the ACL Visualization Library moved into 
//! scout to clean things up some and remove external dependencies.
//! Updated to the new 20-mule-team scout framework. Effectively gutted this
//! part of it to alter the behavior somewhat.
//! This class must service a number of different rotations so the internal
//! state it maintained became a problem. Rather than clear the quaternions
//! each time and start fresh, each update operations returns Rotations for
//! us to do with as we will. In fact, we will hand them off to the Transform
//! class which will composite them for the rotation part thereof.
//! 
//! Based on SGI's trackball.h---see copyright at end of this file.
//
//.........................................................................
//

#ifndef _TRACKBALL_H_
#define _TRACKBALL_H_

#include "Rotation.h" // For internal member

class Trackball
{
 public:
  Trackball();
  ~Trackball();

  //! ----------------------- INPUT METHODS ----------------------
  //! Pass the x and y coordinates of the last and current positions of
  //! the mouse, scaled so they are from (-1.0 ... 1.0).
  //! The resulting Rotation is returned.

  //! performs 3D rotations that include the z axis
  const Rotation &rotate(float fromX, float fromY, float toX, float toY);

  //! performs 2D rotations that disregard the z axis
  const Rotation &spin  (float fromX, float fromY, float toX, float toY);

 private:
  //! Given the x,y screen position and the radius of the trackball,
  //! this returns the missing z coordinate that places the point on the sphere
  //! or hyperbolic sheet if outside the sphere
  float project_to_sphere(float r, float x, float y);

  //! Perform actual rotation from one vector to the other regardless of dims
  const Rotation &rotate_vectors(QuatVector &fromVec, QuatVector &toVec);

  //! A pointer to this is always returned as the result of an input to avoid
  //! having to keep creating and destroying rotations.
  Rotation m_result;

  //! Radius of the trackball.
  static float m_trackball_size;

};

/*
 * (c) Copyright 1993, 1994, Silicon Graphics, Inc.
 * ALL RIGHTS RESERVED
 * Permission to use, copy, modify, and distribute this software for
 * any purpose and without fee is hereby granted, provided that the above
 * copyright notice appear in all copies and that both the copyright notice
 * and this permission notice appear in supporting documentation, and that
 * the name of Silicon Graphics, Inc. not be used in advertising
 * or publicity pertaining to distribution of the software without specific,
 * written prior permission.
 *
 * THE MATERIAL EMBODIED ON THIS SOFTWARE IS PROVIDED TO YOU "AS-IS"
 * AND WITHOUT WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR OTHERWISE,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTY OF MERCHANTABILITY OR
 * FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL SILICON
 * GRAPHICS, INC.  BE LIABLE TO YOU OR ANYONE ELSE FOR ANY DIRECT,
 * SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY
 * KIND, OR ANY DAMAGES WHATSOEVER, INCLUDING WITHOUT LIMITATION,
 * LOSS OF PROFIT, LOSS OF USE, SAVINGS OR REVENUE, OR THE CLAIMS OF
 * THIRD PARTIES, WHETHER OR NOT SILICON GRAPHICS, INC.  HAS BEEN
 * ADVISED OF THE POSSIBILITY OF SUCH LOSS, HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE
 * POSSESSION, USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 * US Government Users Restricted Rights
 * Use, duplication, or disclosure by the Government is subject to
 * restrictions set forth in FAR 52.227.19(c)(2) or subparagraph
 * (c)(1)(ii) of the Rights in Technical Data and Computer Software
 * clause at DFARS 252.227-7013 and/or in similar or successor
 * clauses in the FAR or the DOD or NASA FAR Supplement.
 * Unpublished-- rights reserved under the copyright laws of the
 * United States.  Contractor/manufacturer is Silicon Graphics,
 * Inc., 2011 N.  Shoreline Blvd., Mountain View, CA 94039-7311.
 *
 * OpenGL(TM) is a trademark of Silicon Graphics, Inc.
 */

#endif // _TRACKBALL_H_

