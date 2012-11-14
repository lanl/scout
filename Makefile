#
###########################################################################
# Copyright (c) 2010, Los Alamos National Security, LLC.
# All rights reserved.
# 
#  Copyright 2010. Los Alamos National Security, LLC. This software was
#  produced under U.S. Government contract DE-AC52-06NA25396 for Los
#  Alamos National Laboratory (LANL), which is operated by Los Alamos
#  National Security, LLC for the U.S. Department of Energy. The
#  U.S. Government has rights to use, reproduce, and distribute this
#  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
#  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
#  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
#  derivative works, such modified software should be clearly marked,
#  so as not to confuse it with the version available from LANL.
#
#  Additionally, redistribution and use in source and binary forms,
#  with or without modification, are permitted provided that the
#  following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided 
#      with the distribution.  
#
#    * Neither the name of Los Alamos National Security, LLC, Los
#      Alamos National Laboratory, LANL, the U.S. Government, nor the
#      names of its contributors may be used to endorse or promote
#      products derived from this software without specific prior
#      written permission.
#
#  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
#  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
#  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
#  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
#  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGE.
#
###########################################################################
#
# Notes
# 
#   - The primary job of this Makefile is to run CMake for you.  If you
#     prefer to run CMake manually you are free to do so (see details in
#     the ReadMe.txt file).
#
# Enviornment variables:
# 
#   - SC_BUILD_NTHREADS: Controls the number of threads in a parallel
#     build (enabled if set).
#
#   - SC_BUILD_TYPE: Controls the build type (e.g. debug or release):
#
#	export SC_BUILD_TYPE=[DEBUG|RELEASE] (default is debug)
#   
#####

.PHONY : default
.PHONY : build 
.PHONY : test


##### PARALLEL BUILD CONFIGURATION 
#
ifdef SC_BUILD_NTHREADS
  make_flags := --jobs=$(SC_BUILD_NTHREADS)
endif
#
#####


##### BUILD TYPE SELECTION 
#
ifdef SC_BUILD_TYPE
  build_type  := $(SC_BUILD_TYPE)
else
  build_type  := DEBUG
endif

#
#####


##### SOURCE LOCATION 
# 
src_dir := ${CURDIR}

##### BUILD LOCATION 
# 
# We strongly advice against doing an in-source build with CMake...
# 
ifdef SC_BUILD_DIR
  build_dir := $(SC_BUILD_DIR)
else
  build_dir := $(CURDIR)/build
endif

stdlib_build_dir := $(build_dir)/lib/Standard
stdlib_flags := -DCMAKE_SCC_BOOTSTRAP

#
#####

cmake_flags := -DCMAKE_BUILD_TYPE=$(build_type) -DCMAKE_INSTALL_PREFIX=$(build_dir) $(SC_BUILD_CMAKE_FLAGS)

all: $(build_dir)/Makefile compile 
.PHONY: all 

$(build_dir)/Makefile: CMakeLists.txt
	@echo "*** Scout source directory: $(src_dir)"
	@((test -d $(build_dir)) || (mkdir $(build_dir)))
	@echo "*** Creating Scout build directory: $(build_dir)"
	@(cd $(build_dir); cmake $(cmake_flags) ..;)
	@echo "*** Creating standard library build directory: $(stdlib_build_dir)"
	@((test -d $(stdlib_build_dir)) || (mkdir $(stdlib_build_dir)))

.PHONY: compile
compile: $(build_dir)/Makefile 
	@(cd $(build_dir); make $(make_flags) install)
	@(cd $(stdlib_build_dir); cmake $(cmake_flags) -DCMAKE_SCC_BOOTSTRAP=ON $(src_dir)/lib/Standard)
	@(cd $(stdlib_build_dir); make $(make_flags) install)

.PHONY: scc-only
scc-only: $(build_dir)/Makefile
	@(cd $(build_dir); make $(make_flags))
	cp $(build_dir)/tools/clang/scc/scc $(build_dir)/scout/bin/scc
	cp $(build_dir)/tools/clang/scc-rewrite/scc-rewrite $(build_dir)/scout/bin/scc-rewrite

.PHONY: test
test: 
	@((test -d $(build_dir)/test) || (mkdir $(build_dir)/test))
	@(cd $(build_dir)/test; cmake $(cmake_flags) $(src_dir)/test)
	@(cd $(build_dir)/test; make $(make_flags))
	@(cd $(build_dir)/test; ARGS="-D ExperimentalTest --no-compress-output" make test; cp Testing/`head -n 1 Testing/TAG`/Test.xml ./CTestResults.xml)

.PHONY: testclean
testclean: 
	-@/bin/rm -rf $(build_dir)/test
	@(cd $(build_dir); cmake $(cmake_flags) ..;)

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
