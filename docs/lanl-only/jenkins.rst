.. _git_jenkins:

=================================
Git-Jenkins Integration
=================================

As of December 2012 we have transitioned to a system where all changesets are built and tested
before being merged into the master branch. Only changesets which build and pass all tests are
merged into the master branch. We have two major branches in the repository: "devel" and "master"
all development work should occur in the "devel" branch. To get a copy of the devel branch run the
following at the command prompt::
    
    $ git clone git@elmo:scout scout
    $ cd scout
    $ git checkout devel

When you are ready to push a new changeset to the server only push to the devel branch, e.g.::

    $ git push origin devel

If the build is successful on all platforms and all the tests pass, then the Jenkins continuous 
integration server will merge this changeset into the master branch. In this way the master branch
should never have a broken build.



