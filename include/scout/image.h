/*
 *
 * ###########################################################################
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2013. Los Alamos National Security, LLC. This software was
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
 *
 */

#ifndef __SCOUT_IMAGE_H__
#define __SCOUT_IMAGE_H__

#if defined(__cplusplus)
extern "C" {
#endif

  // This is an opaque type used by the runtime to maintain handles on
  // image instances.  They correspond directly to instances of
  // images created within the language.  The implementation details
  // behind this void* are hidden from the compiler to give us a
  // better shot at portability and to reduce the complexity of code
  // generation.
  typedef void* __scout_img_t;


  //===-----------------------------------------------------------------===//
  // 
  // Image system (environment) support
  //
  
  /// Request that the runtime start/initialize the image system (that
  /// is used in place of having a windowing system (i.e. we want to
  /// render to images instead of to windows).  This should be made
  /// prior to the creation of any images (at startup). 
  int __scout_start_imgsys();

  /// Request that the runtime stop/disable the image system.  This
  /// call should really only be made at program exit to clean up state.
  /// If the system was not properly initialized this call is a no-op and
  /// it is therefore safe to call without tracking the success state of
  /// '__scout_start_imgsys()'.
  void __scout_stop_imgsys();


  //===-----------------------------------------------------------------===//
  // 
  // Image support
  //
  
  /// Create an image with the given width and height in pixels.  If
  /// the image system has not been successfully initialized this
  /// routine will abort and terminate the execution of the program;
  /// in other words, we consider runtime errors like this to be fatal
  /// and our fault vs. a mistake being made by the programmer.
  __scout_img_t __scout_create_image(unsigned width, unsigned height);

  /// Request that the runtime destroy the given image.  Ideally this
  /// should probably be called when any user defined image goes out
  /// of scope within a program -- the semantics of having an image
  /// survivie its scope is a messy thing to consider...
  void __scout_destroy_image(__scout_img_t img);

  /// Request that the runtime show the given image (map it onto the
  /// screen).  We currently do not expose this functionality via the
  /// high-level language so it is typically not dealt with from a
  /// code generation point of view.  This may change...
  void __scout_show_imagew(__scout_img_t img);

  /// Request that the runtime hide the given image (unmap it from the
  /// screen).  Note that this is different than destroying the image.
  /// We currently do not expose this functionality via the high-level
  /// language so it is typically not dealt with from a code
  /// generation point of view.  This may change...
  void __scout_hide_image(__scout_img_t img);

  /// Return the width and height (in pixels) of the given image.  We
  /// current do not expose this functionality via the high-level
  /// language so it is typically not dealt with from a code
  /// generation point of view.  This may change...
  void __scout_image_size(__scout_img_t img,
                          unsigned *width,
                          unsigned *height);

  /// Return the width (in pixels) of the given image.  We currently
  /// do not expose this functionality via the high-level language so
  /// it is typically not dealt with from a code generation point of
  /// view.  This may change...
  unsigned _scout_image_width(__scout_img_t img);

  /// Return the height (in pixels) of the given image.  We currently
  /// do not expose this functionality via the high-level language so
  /// it is typically not dealt with from a code generation point of
  /// view.  This may change...
  unsigned _scout_image_height(__scout_img_t img);

  /// Request that the runtime make the given img the current
  /// target for rendering commands.  This should be called prior to
  /// any of our rendering constructs (e.g. 'renderall') as it is
  /// required for the image to accept rendering commands.
  void __scout_make_image_current(__scout_img_t img);

  /// Request that the runtime swap the buffers of the given image.
  /// This should be called at the completion of any of our rendering
  /// constructs (e.g. renderall) as it is required for the rendered
  /// imagery to become visible (our runtime execution model assumes
  /// double buffering).
  void __scout_swap_buffers(__scout_img_t img);

#if defined(__cplusplus)  
}
#endif 

#endif
