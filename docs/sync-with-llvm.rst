.. _llvm_sync:

=================================
Sync'ing Scout with LLVM's Trunk
=================================

As of August 2012 we have transitioned from using a subversion checkout of LLVM within our repository to a git-subtree approach (note the hypen).  That the concepts conventions about *subtrees* in git can be confusing; this short document aims to help sort through the most common issues and describe how we have configured the Scout repository.


Submodules, Subtrees, and `git-subtree`
==========================================

In order to deal with an external project as part of a git repository the most common approach you'll see is typically `submodules <http://git-scm.com/book/en/Git-Tools-Submodules>`_.  While this works well for portions of your project that you do not change, it does not work well for us as we make direct modifications/additions to LLVM source (including Clang).  The next common approach is `subtree merging <http://git-scm.com/book/ch6-7.html>`_.  Unfortunately, our experience with this approach was also problematic and error prone (especially when pulling subproject updates and dealing with the mix of local and subproject history).  Instead we have adopted `git-subtree` which has recently become part of the mainstream *git* (although it is now in the development branch, it is likely not yet included with the git installed on your system).  

Instead of walking through the details `this post <http://ruleant.blogspot.com/2011/04/git-subtree-module.html>`_ provides a nice quick review of the details of using the `git subtree module`.  You can follow up by reading the man page for the original module `here <https://github.com/apenwarr/git-subtree/blob/master/git-subtree.txt>`_ -- remember this has been rolled into the `git` trunk and therefore might be out-of-date in the near future.


Subtree Structure in the Scout Repository 
=========================================

Assuming you have cloned the Scout repository into the driectory $(SC_SOURCE_DIR) there are three subtrees that correspond directly to the directory hierarchy:

  * `$(SC_SOURCE_DIR)/llvm/` -- the master (trunk) of `llvm <http://llvm.org>`_.
  * `$(SC_SOURCE_DIR)/llvm/tools/clang/` -- the master (trunk) of `clang <http://clang.llvm.org>`_. 
  * `$(SC_SOURCE_DIR)/llvm/projects/compiler-rt/` -- the master (trunk) of the `compiler-rt <http://compiler-rt.llvm.org>`_  project.

For simplicity, this layout matches that expected by LLVM's configuration/build system -- which we then incorporate directly into Scout's repository structure. 

Pull/Merging with the "trunks"
-----------------------------------------

The primary challenge is now take advantage of these subtrees and overall structure to keep our development up-to-date with the activity of the main LLVM/Clang/compiler-rt developers.  The Scout repository includes a set of shell scripts to help out with this process.  You can find them in the `$(SC_SOURCE_DIR)/utils` directory.  Each of the scripts corresponds to each of the subtrees and there are a few key points to keep in mind:

  * Each script uses the `--squash` flag to minimize the version history pulled into the Scout repository (which can cause a flurry of email to the developer's list).
  * You are still responsible for handling any conflicts that arise during the merge.
  * You are expected to be working in the `devel` branch of Scout when using these scripts.
  * The scripts must be invoked from the top-level of the repository to function correctly.  For example::

            $ utils/update-llvm-subtree.sh

Those wanting more details of the 'pull' process should look at the command(s) used within the scripts.  In addition to some error checking everything boils down to the following command::

            $ git subtree pull --squash -P llvm/ http://llvm.org/git/llvm.git master

These *squashed* operations will leave a message in the commit log so you can see the details in the form of commit messages from the subtree's developers.  For example::

            $ git log 
        commit c1b50ddcbc678471a8ef00a412c41843624038ad
        Author: Patrick McCormick <pat@lanl.gov>
        Date:   Mon Jul 30 12:27:53 2012 -0600

        Squashed 'llvm/projects/compiler-rt/' changes from 853733e..b0bb7fb
    
               b0bb7fb [ASan] fix cmake build warning
               8f88dd2 [TSan] delete trailing spaces
               9d150bd tsan: add ReleaseStore() function that merely copies vector clock rather than combines two clocks... (Go runtime)
               715c746 tsan: add missing include
               6b2804f tsan: change event handling from single HandleEvent() to a set of separate functions (Go runtime)
               c1527fd tasn: do not remember stack traces for sync objects for Go (they are not reported anyway)
               93ec440 tsan: remove unnecessary and wrong include
               8648e77 [asan] ensure that asan_init is called in str[n]cmp. Bug found by Nick Kralevich (thanks)
               37f52ab tsan: make the runtime library name tsan-neutral for Go upstream
               87dbdf5 tsan: allow environment to override OnReport() and OverrideFlags()
               8f1104c tsan: suport for Go finalizers
               8971f0c tsan: expect that Go symbolizer can return NULLs
               4d57f44 cmake for compiler-rt: add a function to set output dirs for compiler runtimes equal to directory used by Clang.
               b831086 [asan] don't return from a never-return function. fix a test that had a chain of bugs instead of just one
               b750c4c [ASan] fixup for r160712: provide a default definition for weak __asan_default_options()
               8a1dd56 Make __asan_default_options a weak function that returns a const char*. Users may define it to override the ... 
    
            git-subtree-dir: llvm/projects/compiler-rt


