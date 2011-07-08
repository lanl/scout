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
cmake_flags := -DCMAKE_BUILD_TYPE=Debug
build_dir   := build

# Attempt to figure out how many processors are available for a parallel build.
# Note that this could be fragile across platforms/operating systems.
arch := $(shell uname -s)

ifeq ($(arch),Darwin) 
  sysprof = /usr/sbin/system_profiler -detailLevel mini SPHardwareDataType
  nprocs := -j $(shell $(sysprof) | /usr/bin/grep -i cores | /usr/bin/awk '{print $$ NF}')
else 
  nprocs := -j $(shell /bin/cat /proc/cpuinfo | /bin/grep processor | /usr/bin/wc -l)
endif 


default: $(build_dir)
	echo "building using $(nprocs) processors."
	(cd $(build_dir); make -j $(nprocs))

$(build_dir):
	@((test -d $(build_dir)) || (mkdir $(build_dir) || (mkdir $(build_dir)/doxygen)))
	(cd $(build_dir); cmake $(cmake_flags) ..)

doc: $(build_dir)
	(cd $(build_dir); make doc)

test: test/frontend test/backend

test/frontend: 
	make -C build/test/scfe test

test/backend: 
	make -C build/test/scc test

clean:
	-@/bin/rm -rf $(build_dir)
	-@/bin/rm -rf codeblock[0-9]*.* *.dot 
	-@/usr/bin/find . -name '*~' -exec rm -f {} \{\} \;
	-@/usr/bin/find . -name '._*' -exec rm -f {} \{\} \;
	-@/usr/bin/find . -name '.DS_Store' -exec rm -f {} \{\} \;

