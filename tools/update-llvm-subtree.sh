#!/bin/bash 
#
# Update Scout's llvm subtree with the main llvm trunk.  Note that
# llvm's primary version control is handled with subversion but is
# mirrored with a git snapshot...
#
# This script should be run from the root directory of the Scout
# source.
#
. tools/update-subtree-funcs.sh 

# Make sure we're in the devel branch -- at "home" we lock this down
# so people can't muck with the master branch (as getting there means
# you pass compile and regression checks).  However, we'll leave this
# as a sanity check...
branch=$(active_git_branch)
if [ $branch != 'devel' ]; then 
    echo "This script must be run with the 'devel' branch checked out..."
    exit 1;
fi

# Check for 'dirty' status... 
dirty=$(is_git_dirty)
if [ "$dirty" = "yes" ]; then 
    echo "Your repository has uncommitted changes.  Aborting..."
    exit 1;
fi 


if [ ! -d ./llvm ]; then 
    echo "Unable to find llvm/ in current working directory."
    exit 1;
fi 

# Note: if you run this by hand make sure you do not place a slash
# after the llvm prefix below!  It will cause the subtree merge to
# fail (not yet sure what's going on in this case).
git subtree pull --squash -d -P llvm http://llvm.org/git/llvm.git master 

