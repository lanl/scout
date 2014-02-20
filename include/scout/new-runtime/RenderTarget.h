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

#ifndef __SCOUT_RENDER_TARGET_H__ 
#define __SCOUT_RENDER_TARGET_H__

namespace scout 
{
  typedef float float4 __attribute__((ext_vector_type(4)));

  /// Render targets are tightly coupled with the support of the
  /// language's 'renderall' and other visual constructs.  They have a
  /// base set of functionality but can be either purely software
  /// driven or hardware accelerated.  They can be thought of have
  /// having a direct connection to a framebuffer and the implicit
  /// 'color' variable within rendering blocks.
  ///
  /// This type has a strong connection to our opaque types witin the
  /// language for 'window' and 'image' -- therefore the functionality
  /// represented by the base class tries to provide a uniform
  /// interface and one that is of reduced complexity for code
  /// generation.  Note that we use a C langauge interface that masks
  /// these types via void pointers (these are our opaque types within
  /// the compiler's/Scout's type system.
  class RenderTarget {

   public:

    /// The kind of render target -- this have a direct correspondence
    /// to our subclasses.
    enum RTKind {
      RTK_image,     // software, off-screen frame buffer. 
      RTK_window,    // hardware, on-screen frame buffer.   
      RTK_viewport   // hardware, on-screen subset of a frame buffer. 
    };

   private:
    RTKind   Kind;

   public:
    /// Create a render target of the given kind with the given width
    /// and height in pixels. 
    RenderTarget(RTKind k, unsigned width, unsigned height);
    
    virtual ~RenderTarget() { /* currently a no-op */ }

    /// Return the kind of render target. 
    RTKind   kind() const   { return Kind;   }

    bool isImage() const    { return Kind == RTK_image;    }
    bool isWindow() const   { return Kind == RTK_window;   }
    bool isViewport() const { return Kind == RTK_viewport; }

    /// Return the width of the render target in pixels. 
    unsigned width() const  { return Width;  }

    /// Return the height of the render target in pixels.     
    unsigned height() const { return Height; }

    /// Make the render target the active target for rendering.
    virtual void    bind()           = 0;

    /// Release the render target from being the active target. 
    virtual void    release()        = 0;

    /// Clear the render target's buffers (can be color buffer,
    /// z-buffer, etc. 
    virtual void    clear()          = 0;

    /// Swap the render target's front and back buffers.  
    virtual void    swapBuffers()    = 0;

    /// Obtain a handle to the render target's color buffer.  In the
    /// case of a hardawre-accelerated target this call with return
    /// null (as it is impossible to get a pointer to GPU memory).  If
    /// you want a copy of the data stored in the GPU's memory use the
    /// 'readColorBuffer()' call below.
    virtual float4 *getColorBuffer()        = 0;

    /// Obtain a copy of the render target's color buffer.  Note that
    /// this memory is currently managed internally by the target and
    /// we currently only allow for read-only access (we may need to
    /// change this at a future date).
    virtual const float4 *readColorBuffer() = 0;

    /// Save the image data held by the render target as a PNG image
    /// with the given filename.
    virtual bool savePNG(const char *filename) = 0;

    /// The parent class maintains internal state that allows us to
    /// track the currently active render target (this is currently
    /// limiting and once we get a multi-threaded runtime implemented
    /// there is a potential for having many active targets at a
    /// time).  This call will return the active target, or null if
    /// there is not an active target. 
    static RenderTarget* getActiveTarget() { return ActiveTarget; }

    /// Set the active target. 
    static void setActiveTarget(RenderTarget *target) { ActiveTarget = target; }

   protected:
    float4   Background;      /// The background color of the target. 
    unsigned Width, Height;   /// The pixel dimensions of the target. 
    
   protected:
    static RenderTarget* ActiveTarget;
  };
}

#endif
