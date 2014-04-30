.. _lanl_only_build:

Building on Darwin
---------------------------

If you are building on the Darwin cluster use the following steps::

    $ module load cuda
    $ export CUDA_DIR=$(CUDA_INSTALL_PATH)
    $ module load cmake/2.8.11.1
    $ module load compilers/gcc/4.7.2
    $ module load mpi

In addition, build SDL-1.2.15 from source and install within your home
directory and then set __SDL_DIR__ correspondingly. 

    $ curl https://www.libsdl.org/release/SDL-1.2.15.tar.gz > SDL-1.2.15.tar.gz
    $ tar xfz SDL-1.2.15.tar.gz
    $ cd SDL-1.2.15
    $ ./configure -prefix=HOMEDIR
    $ make; make install

Build glfw from source and install within your home
directory and then set __GLFW_DIR__ correspondingly. 

    $ git clone https://github.com/glfw/glfw.git
    $ cmake -DCMAKE_INSTALL_PREFIX:PATH=HOMEDIR .
    $ make; make install

Finally, you
should unset your __DISPLAY__ environment variable to avoid issues
with OpenGL contexts across the ssh connection (this skips the
compilation checks of OpenGL shaders but will still build the
libraries correctly -- assuming you are not editing the runtime
shaders).

