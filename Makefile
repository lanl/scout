#
# ----- The Scout Programming Language
#
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file LICENSE.txt for details. 
# 

.PHONY : default
.PHONY : build 
.PHONY : test

arch        := $(shell uname -s)
date        := $(shell /bin/date "+%m-%d-%Y")
cmake_flags := -DCMAKE_BUILD_TYPE=DEBUG
build_dir   := build

# Attempt to figure out how many processors are available for a parallel build.
# Note that this could be fragile across platforms/operating systems.
arch := $(shell uname -s)

ifeq ($(arch),Darwin) 
  sysprof = /usr/sbin/system_profiler -detailLevel mini SPHardwareDataType
  nprocs := $(shell ($(sysprof) | /usr/bin/grep -i cores | /usr/bin/awk '{print $$ NF}'))
else 
  nprocs := $(shell /bin/cat /proc/cpuinfo | /bin/grep processor | /usr/bin/wc -l)
endif 

# Since we're likely I/O bound try and sneak in twice as many build
# threads as we have cores...
nprocs := $(shell expr $(nprocs) \* 2)



all: $(build_dir)/Makefile compile
.PHONY: all 

$(build_dir)/Makefile: CMakeLists.txt
	@((test -d $(build_dir)) || (mkdir $(build_dir)))
	@(cd $(build_dir); cmake $(cmake_flags) ..;)

.PHONY: compile
compile: $(build_dir)/Makefile 
	@(cd $(build_dir)/llvm; make -j $(nprocs))
	@(cd $(build_dir); make -j $(nprocs))

.PHONY: xcode
xcode:;
	@((test -d xcode) || (mkdir xcode))
	@(cd xcode; cmake -G Xcode ..)

.PHONY: clean
clean:
	-@/bin/rm -rf $(build_dir)
	-@/usr/bin/find . -name '*~' -exec rm -f {} \{\} \;
	-@/usr/bin/find . -name '._*' -exec rm -f {} \{\} \;
	-@/usr/bin/find . -name '.DS_Store' -exec rm -f {} \{\} \;

