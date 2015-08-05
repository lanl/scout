/**
 * Ian Sohl - 2015
 * Copyright (c) 2015      Los Alamos National Security, LLC
 *                         All rights reserved.
 * Legion Image Composition - Qt Interaction Header
 */

#ifndef INTERACT_H
#define INTERACT_H

#include <cmath>

struct inArgs{
	int argc;
	char **argv;
}; /**< Application arguments for Qt */

struct Movement{
	float invPVM[16]; /**< Inverse PV Matrix for rendering */
	float xdat;		  /**< X[3] value for composition ordering */

	bool operator==( const Movement& rhs ) const {
		/**
		 * Manually check for invPVM equivalence
		 */
		for(int i = 0; i < 16; ++i){
			if(std::abs(invPVM[i]-rhs.invPVM[i])>0.000001){ // Floating point problems
				return false;
			}
		}
		return true;
	}

	Movement& operator =(const Movement& a){
		/**
		 * Manual assignment
		 */
		for(int i = 0; i < 16; ++i){
			invPVM[i] = a.invPVM[i];
		}
		xdat = a.xdat;
	    return *this;
	}
}; /**< Current data state */

void Interact(inArgs &);
void newImage(int *vals, Movement mov, int width, int height);
Movement getMovement();
bool getDone();

#endif
