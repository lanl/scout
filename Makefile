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

nprocs := $(shell expr $(nprocs) \* 2)

# Since we're likely I/O bound try and sneak in twice as many build
# threads as we have cores...

default: $(build_dir)
	echo "building using $(nprocs) processors."
	(cd $(build_dir); make -j $(nprocs))

$(build_dir):
	@((test -d $(build_dir)) || (mkdir $(build_dir)))
	(cd $(build_dir); cmake $(cmake_flags) ..)

clean:
	-@/bin/rm -rf $(build_dir)
	-@/usr/bin/find . -name '*~' -exec rm -f {} \{\} \;
	-@/usr/bin/find . -name '._*' -exec rm -f {} \{\} \;
	-@/usr/bin/find . -name '.DS_Store' -exec rm -f {} \{\} \;

