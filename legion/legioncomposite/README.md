## Legion Compositor
A Legion-based Image Compositor that performs interactive rendering in an HPC environment.

## Dependencies
 * GNU C Compiler 4.7 or greater (Tested with 4.9.2) or equivalent compiler
 * Stanford Legion (https://github.com/StanfordLegion/legion) master branch
 * If building the Qt visualizer, Qt 4.0 or greater.
 * If using GPU rendering, NVIDIA CUDA (Only tested with 7.0)

## Compilation Instructions
Compilation of the compositor itself simply requires running the Makefile in the main project directory. Some components may require you to manually edit the Makefile itself to specify paths.
In order to specify where Legion is, you need to set the environment variable 'LG_RT_DIR' to point to the runtime directory of Legion.

 * If using the Qt Visualizer, compile the contents of the QtViewer folder first using 'make'. Also set the environment variable 'QT_HOME' to the location of the QtViewer folder.
 * Compile the main program with 'make' in the main legioncomposite folder. Make sure to set GASNET_ROOT and OPTIX_DIR if needed.

Compiling the viewer requires you to be on a 64-bit system for the moment, and can be built in the CompositeViewer folder using the command
```bash
g++ -O2 -o viewer viewer.cc `Magick++-config --cppflags --cxxflags --ldflags --libs`
```
