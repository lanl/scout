.. _lanl_only_build:

Building on Darwin
---------------------------

If you are building on the Darwin cluster use the following steps::

    $ module load cuda
    $ export CUDA_DIR=$(CUDA_INSTALL_PATH)
    $ module load mpi

In addition, build SDL from source and install within your home
directory and then set __SDL_DIR__ correspondingly. You will also
need to build cmake and add its location to the beginning of your path,
as the version on Darwin is out of date.  Finally, you
should unset your __DISPLAY__ environment variable to avoid issues
with OpenGL contexts across the ssh connection (this skips the
compilation checks of OpenGL shaders but will still build the
libraries correctly -- assuming you are not editing the runtime
shaders).

