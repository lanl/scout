.. _build:

======================
Building Scout
======================

Scout is open source software.  For details on the terms of the
release please see the license agreement.

.. todo:: Need to add a link to the ``license`` above. 

The capabilities of the Scout language rely heavily on the 
`LLVM compiler infrastructure <http://llvm.org/>`_, which includes not 
only LLVM itself but also the `Clang <http://clang.llvm.org>`_ front-end,
the `compiler-rt <http://compiler-rt.llvm.org>`_ library, and the
the `lldb debugger <http://lldb.llvm.org>`_.  As we have modified portions
of these projects they are included within the Scout source
distribution.  Please note that all these packages also come with
their own licenses.

This section provides the basic set of instructions for building the
Scout compiler and its supporting infrastructure from the source code
distribution.  In addition, it contains an advanced section on how the
build system is designed and how to go about making additions and
changes.


Prerequisites 
====================

This section provides a list of the set of software packages that are
required as well as aditional packages that can be used to expand the
capabilities of Scout.

Supported Platforms
--------------------------

Our primary development platforms are Mac OS X 10.7-10.9 and several 
varieties of Linux distributions.  A list of Linux releases we use as 
part of our continuous integration development process are listed below:

   * **Fedora 17** -- See http://fedoraproject.org
   * **Ubuntu 12.04** -- See http://www.ubuntu.com 
   * **Ubuntu 14.04** -- See http://www.ubuntu.com 

The use of these Linux-based platforms will require the installation
of additional software development packages and libraries.  This can
either be achieved with via package management systems or downloading
and building from source.  Further details on these packages are
provided below. 

For Mac OS X based systems we primarily run and test using Mac OS X
10.8 and 10.9 with the latest release of Xcode (currently 5.1).
While there are several package distribution management systems for
Mac OS (e.g. macports) we typically do not use them in favor of
building and installing from source.

Ubuntu 12.04 Requirements
--------------------------

A number of other required packages need to be installed:
 
    $ apt-get install build-essential git libsdl1.2dev freeglut3 freeglut3-dev xorg xorg-dev libxcursor-dev

LLVM requires gcc 4.7 which does not ship with Ubuntu 12.04 but can
be added via:

    $ add-apt-repository ppa:ubuntu-toolchain-r/test

    $ apt-get update

    $ apt-get install gcc-4.7 g++-4.7

    $ update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-4.6 

    $ update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.7 40 --slave /usr/bin/g++ g++ /usr/bin/g++-4.7 

    $ update-alternatives --config gcc

    $ > select gcc-4.7 (option 2)

Other required packages need to be build from source, these include CMake and GLFW3

Ubuntu 14.04 Requirements
--------------------------

A number of other required packages need to be installed:
    $ apt-get install build-essential git libsdl1.2-dev freeglut3 freeglut3-dev xorg-dev libxcursor-dev cmake

CMake
-------------

The build configuration of Scout is based on `CMake
<http://cmake.org>`_ and requires version 2.8.9 or later.  Note that
CMake is not a build system but is instead a cross-platform build
system generator.  Our default build system is handled by creating
Unix makefiles.  Other options exist via CMake but the details are
beyond the scope of this document.

While CMake does provide support Microsoft Windows, we have not tested
our code in this enviornment.  We welcome contributions in this area.

Libraries
---------------------

Scout is currently dependent on a few external projects that must be
installed on your system prior to building.  These required, and 
strongly recommended, packages include:

  * **GLFW3**. Library for OpenGL windowing.  See http://www.glfw.org/

  * **OpenGL 2.X or higher**.  We *strongly* recommend having access
    to a system with a GPU capable of supporting hardware accelerated
    rendering.  It might be possible to use a software-only
    implementation of OpenGL (such as `Mesa <http://www.mesa3d.org>`_)
    but we have not yet tested this approach.
    
    We are working on other approaches to rendering but they are not
    available as part of the current release.
  
In addition, several other packages can be installed that provide
extended support for the language.  These include:

  * **CUDA** -- If CUDA is installed on your system it will 
    enable code generation for NVIDIA GPUs.  We currently
    require that you have CUDA 5.X installed and have either 
    a Fermi or Kepler card installed in your system.  
   
    On Linux some versions of the cuda driver require that 
    you set __GL_NO_DSO_FINALIZER=1 or programs may segfault 
    on exit.
  
    CUDA is available from [NVIDIA's `developer's web 
    site <http://developer.nvidia.com/>`_.

  * **Thrust** -- If CUDA is installed on your system you also
    need the thrust library v1.7.0 which is available from 
    `GitHub <https://github.com/thrust/thrust.git>`_.  
    (git clone -b 1.7.0 https://github.com/thrust/thrust.git)
    Replace your current thrust library (e.g. /usr/local/cuda/include/thrust) 
    with the thrust sub-directory that is inside the git repository.

  * **MPI** -- We currently use MPI for 
    implementation.  Scout does not currently support 
    distributed memory applications.

.. todo:: Need to add a link to ``hwloc`` in the list above. 

The CMake build system for Scout checks for all of these packages
installed in reasonably standard locations (.e.g. /usr, /usr/local,
/opt,) but if they are installed in a non-standard location you can
use the following environment variables to help CMake find them during
the configuration stages:

   * .. envvar:: GLFW_DIR=/path/to/glfw3/install
   * .. envvar:: CUDA_DIR=/path/to/cuda/install
   * .. envvar:: MPI_HOME=/path/to/mpi/install 

.. _documentation-system-label:

Documentation System
---------------------------

Our on-line documentation is created using the Sphinx Documentation
Generator.  For more information see the `Sphinx Overview 
<http://sphinx.pocoo.org/index.html>`_.

Build
=====================

In the top-level directory of the source code there is a ``Makefile``
that will automate the process of running CMake and creating an
out-of-source build directory.  After this is step is completed, the
``Makefile`` will begin the compilation of the libraries and programs
that make up the toolchain.  

This process is as simple as invoking

    $ make 
    
at the command prompt.  By default the process will create a *build*
directory at the top-level of the source that contains the compiled
files and libraries. To allow finer control of the build configuration
the following environment variables may be set prior to invoking
``make``.

*  .. envvar:: SC_BUILD_NTHREADS 

  Controls the number of make processes/threads executed as part of
  the final, after configuration is complete, build.  This is
  equivalent to executing::
    
    $ make -j $(SC_BUILD_NTHREADS)

* .. envvar:: SC_BUILD_TYPE 

  Control whether a debug or release (optimized) build is used::

  $ export SC_BUILD_TYPE=DEBUG|RELEASE       (defaults to DEBUG)
        
* .. envvar:: SC_BUILD_DIR 

  Controls both the name and location of the build directory::
  
       $ export SC_BUILD_DIR=/the/path/to/the/build
        
* .. envvar:: SC_BUILD_CMAKE_FLAGS

  This should primarily be used by those who have a detailed
  understanding of CMake and the configuration parameters within
  both Scout and LLMV.  The set of provided flags will be passed to
  CMake as part of the configuration run.  Full details of this
  process are currently beyond the scope of this document.

* .. envvar:: SC_BUILD_LLDB

  Controls if lldb with scout suppport is built. On Linux this requires 
  the additional pacakges gcc-4.8, swig, python-dev, and libedit-dev. On mac
  requires xcode, swig and prce.

.. ifconfig:: lanl==True

  .. include:: lanl-only/build.rst
