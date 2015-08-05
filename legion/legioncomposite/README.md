## Legion Compositor
A Legion-based Image Compositor that performs interactive rendering in an HPC environment.

## Dependencies
 * Stanford Legion (https://github.com/StanfordLegion/legion) master branch
 * If building the Qt visualizer, Qt 4 or greater.
 * Eigen linear algebra package (http://eigen.tuxfamily.org)
 * If using GPU rendering, NVIDIA CUDA

## Compilation Instructions
Compilation of the compositor itself simply requires running the Makefile in the main project directory. Some components may require you to manually edit the Makefile itself to specify paths.
In order to specify where Legion is, you need to set the environment variable 'LG_RT_DIR' to point to the runtime directory of Legion.

Compiling the viewer requires you to be on a 64-bit system for the moment, and can be built in the CompositeViewer folder using the command
```bash
g++ -O2 -o viewer viewer.cc `Magick++-config --cppflags --cxxflags --ldflags --libs`
```
