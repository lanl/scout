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

project(scout_test_scpp_error)

set(SRC_PATH ${CMAKE_SOURCE_DIR}/sc++/error)

set(name forall-mesh-unknown)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "error: unknown or invalid mesh instance")
set_property(TEST ${tname} PROPERTY WILL_FAIL) 

set(name forall-mesh-shadow)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "error: forall loop variable 'c' shadows mesh member")
set_property(TEST ${tname} PROPERTY WILL_FAIL) 

set(name lhs-rhs)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "error: mesh field appears on LHS then on subsequent RHS")
set_property(TEST ${tname} PROPERTY WILL_FAIL) 

set(name forall-array-4d)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "error: forall missing.*expected.*keyword: 'in'")
set_property(TEST ${tname} PROPERTY WILL_FAIL) 

set(name forall-array-coloncolon)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "error: invalid end in forall")
set_property(TEST ${tname} PROPERTY WILL_FAIL) 

set(name forall-array-comma)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "error: forall missing.*expected.*comma or keyword: 'in'")
set_property(TEST ${tname} PROPERTY WILL_FAIL) 

set(name forall-array-end)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "error: invalid end in forall")
set_property(TEST ${tname} PROPERTY WILL_FAIL) 
 
set(name forall-array-ident)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "error: expected identifier")
set_property(TEST ${tname} PROPERTY WILL_FAIL) 

set(name forall-array-square)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "error: expected '\\['")
set_property(TEST ${tname} PROPERTY WILL_FAIL) 

set(name forall-array-groups)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "error: start/end/stride groups misformated")
set_property(TEST ${tname} PROPERTY WILL_FAIL) 

set(name eoshift-args)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "error: invalid number of eoshift")
set_property(TEST ${tname} PROPERTY WILL_FAIL) 

set(name eoshift-nonmesh)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "error: first argument to eoshift")
set_property(TEST ${tname} PROPERTY WILL_FAIL) 

set(name renderall-color)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "error: color components {r,g,b,a} were not assigned in renderall")
set_property(TEST ${tname} PROPERTY WILL_FAIL) 

#SC_TODO
#set(name mesh-param-byvalue)
#set(tname sc++_error_${name})
#add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
#set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
#    "error: mesh parameters must be declared as pointers in scc")
#set_property(TEST ${tname} PROPERTY WILL_FAIL) 

set(name mesh-ptr-field)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "mesh field element must be non-pointer type")
set_property(TEST ${tname} PROPERTY WILL_FAIL) 

set(name mesh-indirect-ptr)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "mesh field element must be non-pointer type")
set_property(TEST ${tname} PROPERTY WILL_FAIL) 

set(name builtin-args)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION
    "error: invalid number of args")
set_property(TEST ${tname} PROPERTY WILL_FAIL)

set(name builtin-nonmesh)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION
    "error: argument to rank")
set_property(TEST ${tname} PROPERTY WILL_FAIL)

set(name mesh-access)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "error: illegal mesh element access -- outside of a supporting mesh construct")
set_property(TEST ${tname} PROPERTY WILL_FAIL) 

set(name non-pure-task)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "error: non-pure task function")
set_property(TEST ${tname} PROPERTY WILL_FAIL) 

set(name non-pure-task2)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "error: non-pure task function")
set_property(TEST ${tname} PROPERTY WILL_FAIL) 

set(name forall-nontask)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} -sclegion  ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "error: Forall outside of task in legion mode not allowed")
set_property(TEST ${tname} PROPERTY WILL_FAIL)

# make this a warning for now
#set(name task-io)
#set(tname sc++_error_${name})
#add_test(${tname} ${CMAKE_SCC_COMPILER} -sclegion ${SRC_PATH}/${name}.scpp)
#set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
#    "error: I/O function called within task body")
#set_property(TEST ${tname} PROPERTY WILL_FAIL)

set(name no-mesh-access)
set(tname sc++_error_${name})
add_test(${tname} ${CMAKE_SCC_COMPILER} -sclegion ${SRC_PATH}/${name}.scpp)
set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
    "error: no mesh access in task function")
set_property(TEST ${tname} PROPERTY WILL_FAIL)

# disable for now, as Jamal needs this
#set(name cshift-expr)
#set(tname sc++_error_${name})
#add_test(${tname} ${CMAKE_SCC_COMPILER} ${SRC_PATH}/${name}.scpp)
#set_property(TEST ${tname} PROPERTY PASS_REGULAR_EXPRESSION 
#    "error: Non integer shift in")
#set_property(TEST ${tname} PROPERTY WILL_FAIL) 
