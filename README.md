# The Scout Programming Language

The Scout toolchain is still very much a *work-in-progress* and the
documentation and source are both under active development.  We use
64-bit Mac OS X 10.9.X and Linux systems (primarily the LTS releases
of Ubuntu / Linux Mint) as our primary development platforms.  In
addition to these primary platforms our nightly regression systems
test the following:

Scout is open-source software (released under LA-CC-13-015).  Los
Alamos National Security, LLC (LANS) owns the copyright to Scout.  The
license is BSD-like with a "modifications must be indicated"
clause. See the
[LICENSE](http://github.com/losalamos/scout/blob/master/License.md)
file for the full details.

**NOTE**: Scout is still very much a *work-in-progress* and the
documentation and source are both under active development.  We use
Linux-based systems and Mac OS X 10.8.X as our primary software
development platforms with a range of supported GPU architectures.
For CPU architectures we primarily support x86 systems but have
recently added support for ARM-based platforms. More details on
specific hardware support is included below.

## Requirements 

This section highlights the required software packages for a
successful build and installation of Scout.  At this point in time we
only develop under Linux and Mac OS X and therefore can not provide
support for building on Windows-based systems.

The capabilities of the Scout Language rely heavily on the
[LLVM](http://llvm.org) compiler infrastructure; including
[Clang](http://clang.llvm.org), [LLDB](http:://lldb.llvm.org) and the
[compiler-rt](http://compiler-rt.llvm.org) low-level runtime.  As we
modify each of these components they are included within our release.
*Please make sure to read the individual license agreements for each
of these packages.*

### Operating System Support

**Linux Systems**: As mentioned above, our primary development
environments are Mac OS X 10.8.X and several varieties of Linux-based
distributions.  A list of Linux releases that we monitor as part of
our continuous integration development process are listed below:

  * [Scientific Linux 6.2](https://www.scientificlinux.org)
  * [Fedora Core 17](http://fedoraproject.org)
  * [Ubuntu 12.04](http://www.ubuntu.com)

Each of these platforms requires the installation of additional
software development packages and libraries.  This can be achieved
used either the various software package management systems on these
platforms or by downloading and building them directly from source.
Further details on each required (and optional) package are provided
below.

**Mac OS X Systems**: For Mac OS X systems we primarily test using
10.8.X and 10.7.X and the latest version of Apple's Xcode (currently
4.6).  While there are several package distribution management systems
for Mac OS X (e.g. macports) we typically avoid them and instead favor
building and installing from source.  If you use of these systems you
may experience differences and/or conflicts with the details that
follow.

#### Required Software Packages 

The following software packages are required to produce a minimal
version of Scout that targets CPU (x86) architectures (both single and
multi-core):

  * [CMake](http:://cmake.org) -- version 2.8.9 or later. **TODO**:
    Check version specs in the source.
  
  * [Simple Direct Media Layer](http://www.libsdl.org) -- **TODO**
    verify version requirements.  (**NOTE** we are working on our own
    layer of support in place of SDL).
	
  * [OpenGL 2.X or higher](http://opengl.org) -- We **strongly**
    recommend having access to a system with a GPU capable of
    supporting hardware accelerated OpenGL.  We are currently
    exploring the use of software-based OpenGL support but can not
    recommend that as a viable option at this point in time.  **NOTE**
    that we have primarily worked with NVIDIA GPUs for OpenGL support
    and are actively hardening support for AMD GPUs as well as
    beginning the initial exploration of both Intel's and ARM's GPU
    architectures.

#### Optional Software Packages 

These additional packages expand Scout's target architectures and can
be used to generate code for a mixed set of platforms from a single
source file.  For details on the various architectures supported by
Scout see the Hardware section below.

  * [NVIDA's CUDA](http://developer.nvidia.com/) -- we support CUDA
    version 5.X and Fermi- and Kepler-class cards (*sorry, we haven't
    verified older GPU models are well supported*).
	
  * **OpenCL** -- on platforms with AMD GPUs we support OpenCL-based
    code generation.  **NOTE**: If our build configuration finds CUDA
    installed on your system OpenCL support will be disabled.
     
Our build configuration process will attempt to discover if you have
any of these GPU development environments installed and will add
support for them automatically. If you are using a non-standard
install location for these packages you can set the following
environment variables prior to invoking the build (more details
below):

  * `SDL_DIR=/path/to/sdl/install`
  * `CUDA_DIR=/path/to/cuda/install`
  
### Building From Source 

In the top-level directory of the source code there is a `Makefile`
that will automate the process of creating an *out-of-source* build
directory and running CMake.  After the configuration step is
completed, this approach will begin the compilation of the full
distribution.

This process is as simple as invoking 

	$ make
	
from the root of the distribution. 

To allow finer control of the build configuration the following
environment variables may be set prior to invoking `make`.

  * `SC_BUILD_NTHREADS` -- Controls the use of a parallel build of the
    source.  Set this variable to the number of make processes/threads
    to be executed.  This is extremely useful in speeding up the build
    process.  It not only helps to have a system with several process
    cores but also a lot of installed RAM -- a systems with several
    cores and limited memory can struggle through a parallel build.
	
	$ export SC_BUILD_NTHREADS=8      (compile with 8 build threads) 

  * `SC_BUILD_TYPE` -- Control whether a debug or release (optimized)
    build is used:

	$ export SC_BUILD_TYPE=DEBUG|RELEASE       (defaults to DEBUG)
                
  * `SC_BUILD_DIR` -- Controls both the name and location of the build
    directory:

	$ export SC_BUILD_DIR=/the/path/to/the/build
                
  * `SC_BUILD_CMAKE_FLAGS` -- (*Advanced*) This should primarily be
    used by those who have a detailed understanding of CMake and the
    configuration parameters within both Scout and LLVM.  The set of
    provided flags will be passed to CMake as part of the
    configuration run.  Full details of this using this option are
    currently beyond the scope of this document.



  


 
