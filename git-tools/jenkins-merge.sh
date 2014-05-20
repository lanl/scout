#!/bin/sh 
#
# script for jenkins to try and merge w/ llvm/clang/lldb
# and generate a list of conflicts

ret=0

# try llvm merge
mkdir -p llvmlog
echo "<pre>" > llvmlog/index.html
./git-tools/update-llvm-subtree.sh >> llvmlog/index.html 2>&1
if [ $? -ne 0 ]
then
  # get list of conflicts
  git diff --name-only --diff-filter=U > /tmp/conflicts
  git commit -am "commit w/ conflicts"
  while read f; do
    git diff HEAD HEAD~1 $f
  done < /tmp/conflicts
  ret=$((ret+1))
fi

#try and merge w/ clang
mkdir -p clanglog
echo "<pre>" > clanglog/index.html
./git-tools/update-clang-subtree.sh >> clanglog/index.html 2>&1
if [ $? -ne 0 ]
then
  # get list of conflicts
  git diff --name-only --diff-filter=U > /tmp/conflicts
  git commit -am "commit w/ conflicts"
  while read f; do
    git diff HEAD HEAD~1 $f
  done < /tmp/conflicts
  ret=$((ret+1))
fi

#try and merge w/ lldb
mkdir -p lldblog
echo "<pre>" > lldblog/index.html
./git-tools/update-lldb-subtree.sh >> lldblog/index.html 2>&1
if [ $? -ne 0 ]
then
  # get list of conflicts
  git diff --name-only --diff-filter=U > /tmp/conflicts
  git commit -am "commit w/ conflicts"
  while read f; do
    git diff HEAD HEAD~1 $f
  done < /tmp/conflicts
  ret=$((ret+1))
fi

return $ret
#
