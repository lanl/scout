#!/bin/sh
#
# find the current versions of llvm/clang/lldb
# that scout has merged w/ and checkout into /tmp
#

# stolen from git-subtree.sh
find_latest_squash()
{
        #debug "Looking for latest squash ($dir)..."
        dir="$1"
        sq=
        main=
        sub=
        git log --grep="^git-subtree-dir: $dir/*\$" \
                --pretty=format:'START %H%n%s%n%n%b%nEND%n' HEAD |
        while read a b junk; do
                #debug "$a $b $junk"
                #debug "{{$sq/$main/$sub}}"
                case "$a" in
                        START) sq="$b" ;;
                        git-subtree-mainline:) main="$b" ;;
                        git-subtree-split:) sub="$b" ;;
                        END)
                                if [ -n "$sub" ]; then
                                        if [ -n "$main" ]; then
                                                # a rejoin commit?
                                                # Pretend its sub was a squash.
                                                sq="$sub"
                                        fi
                                        #debug "Squash found: $sq $sub"
                                        #echo "$sq" "$sub"
                                        echo "$sub"
                                        break
                                fi
                                sq=
                                main=
                                sub=
                                ;;
                esac
        done
}

#find the hashes for last merge
llvmhash="$(find_latest_squash "llvm")"
echo "llvm hash: $llvmhash"
clanghash="$(find_latest_squash "llvm/tools/clang")"
echo "clang hash: $clanghash"
lldbhash="$(find_latest_squash "llvm/tools/lldb")"
echo "lldb hash: $lldbhash"

#get the repos and checkout the correct versions
export http_proxy="http://proxyout.lanl.gov:8080"
rm -rf /tmp/llvm
cd /tmp
git clone http://llvm.org/git/llvm.git
cd llvm
git checkout master
git checkout $llvmhash
cd tools
git clone http://llvm.org/git/clang.git
cd clang 
git checkout master
git checkout $clanghash
cd ..
git clone http://llvm.org/git/lldb.git 
cd lldb
git checkout master
git checkout $lldbhash



