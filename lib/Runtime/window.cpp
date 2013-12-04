/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 *
 *-----
 *
 */
#include <cassert>
#include <iostream>

#include "scout/scout.sch"
#include "scout/Runtime/window.h"
#include "scout/math.h"

using namespace std;
using namespace scout;

// ----- remove_all
//
static bool remove_all(viewport_rt_p vp)
{
  delete vp;
  return true;
}

// ----- ~window_rt
//
window_rt::~window_rt()
{
  viewports.remove_if(remove_all);
}

// ----- add_viewport
//
void window_rt::add_viewport(viewport_rt_p vp)
{
  assert(vp != 0);

  // Todo: It would probably be nice to have an option of turning on or
  // off these checks in the build configuration. --psm
  if (vp->xpos < 0.0f || vp->xpos > 1.0 ||
      vp->ypos < 0.0f || vp->ypos > 1.0) {
    cerr << "warning: viewport position is out of normalized bounds.\n";
    cerr << "   position: (" << vp->xpos << ", " << vp->ypos << ")\n";
    cerr << "   clamped into valid range\n";
    vp->xpos = clamp(vp->xpos, 0.0f, 1.0f);
    vp->ypos = clamp(vp->ypos, 0.0f, 1.0f);
  }

  if (vp->width < 0.0f || vp->width > 1.0 ||
      vp->height < 0.0f || vp->height > 1.0) {
    cerr << "warning: viewport dimensions are out of normalized bounds.\n";
    cerr << "   dimensions: " << vp->width << " x " << vp->height << endl;
    cerr << "   clamped into valid range.\n";
    vp->xpos = clamp(vp->xpos, 0.0f, 1.0f);
    vp->ypos = clamp(vp->ypos, 0.0f, 1.0f);
  }

  viewports.push_back(vp);
}

