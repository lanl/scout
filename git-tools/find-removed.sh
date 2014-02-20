#!/bin/sh
#
# find files that exist in our repo but not in llvm/clang
# you need to run current-merged-llvm.sh first
# use git reset --hard HEAD to undo

#make sure current-merged-llvm.sh has run
if [ -d "/tmp/llvm/tools/clang" ]; then

  # i'm not sure why we need LibraryDependencies.inc but we do.
  #exclude /llvm/project and tools/lldb for now
  diff -rq -x .git llvm /tmp/llvm | grep "Only in llvm" | grep -iv Scout | grep -v  "llvm/projects" | grep -v "LibraryDependencies.inc" | grep -v "tools/lldb" | awk '{print $3 $4}' | sed 's/:/\//' > /tmp/filelist

  #make sure none of these files have "Scout" in them
  while read f; do
    grep -L "Scout" $f
  done < /tmp/filelist > /tmp/filelist2

  #clobber them all
  while read f; do
    ls $f
    #use at your own risk...
    #git rm -r $f
  done < /tmp/filelist2
fi
