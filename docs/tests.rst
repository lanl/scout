.. _tests:

Tests
=====================

Building and running the optional scout tests is done by invoking

        $ make test

at the command prompt. The following enviroment variables control 
which tests are run.

*  .. envvar:: DISPLAY

  if __DISPLAY__ is set then the build system will assume you have
  full access to the computer display, and will run various test which
  open windows on your display.
   
*  .. envvar:: SC_NVIDIA

  Controls whether tests requiring a Nvidia graphics card are run.
 
.. ifconfig:: lanl==True

  .. include:: lanl-only/tests.rst

