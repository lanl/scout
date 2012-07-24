<<<<<<< HEAD
#
# Create a library called "CompilerRT" which includes the source files.

#INCLUDE_DIRECTORIES(
# ${CMAKE_CURRENT_BINARY_DIR}
#)

# Generic functions needed for each architecture
# libcompiler_rt.Generic.a    libcompiler_rt.Optimized.a

# Generic
SET( Generic_SRCS
 absvdi2.c absvsi2.c addvdi3.c addvsi3.c ashldi3.c ashrdi3.c
 clzdi2.c clzsi2.c cmpdi2.c ctzdi2.c ctzsi2.c
 divdc3.c divdi3.c divsc3.c ffsdi2.c
 fixdfdi.c fixsfdi.c fixunsdfdi.c fixunsdfsi.c fixunssfdi.c
 fixunssfsi.c floatdidf.c floatdisf.c floatundidf.c floatundisf.c
 gcc_personality_v0.c lshrdi3.c moddi3.c muldc3.c muldi3.c
 mulsc3.c mulvdi3.c mulvsi3.c negdi2.c negvdi2.c negvsi2.c
 paritydi2.c paritysi2.c popcountdi2.c popcountsi2.c powidf2.c
 powisf2.c subvdi3.c subvsi3.c ucmpdi2.c udivdi3.c
 udivmoddi4.c umoddi3.c apple_versioning.c eprintf.c
 )

# Optimized functions for each architecture

# Commenting out for the min until the basics are working first.
# ADD_SUBDIRECTORY( ppc )
# ADD_SUBDIRECTORY( x86_64 )
# ADD_SUBDIRECTORY( i386 )
# ADD_SUBDIRECTORY( arm )

# List of functions needed for each architecture.
SET( i386_Functions
 divxc3.c fixunsxfdi.c fixunsxfsi.c fixxfdi.c floatdixf.c
 floatundixf.c mulxc3.c powixf2.c clear_cache.c enable_execute_stack.c 
 )

SET( x86_64_Functions
 absvti2.c addvti3.c ashlti3.c ashrti3.c clzti2.c cmpti2.c
 ctzti2.c divti3.c divxc3.c ffsti2.c fixdfti.c fixsfti.c
 fixunsdfti.c fixunssfti.c fixunsxfdi.c fixunsxfsi.c
 fixunsxfti.c fixxfdi.c fixxfti.c floatdixf.c floattidf.c
 floattisf.c floattixf.c floatundixf.c floatuntidf.c
 floatuntisf.c floatuntixf.c lshrti3.c modti3.c multi3.c
 mulvti3.c mulxc3.c negti2.c negvti2.c parityti2.c
 popcountti2.c powixf2.c subvti3.c ucmpti2.c udivmodti4.c
 udivti3.c umodti3.c clear_cache.c enable_execute_stack.c
 )

SET( PPC_Functions
 divtc3.c fixtfdi.c fixunstfdi.c floatditf.c floatunditf.c
 gcc_qadd.c gcc_qdiv.c gcc_qmul.c gcc_qsub.c multc3.c
 powitf2.c restFP.c saveFP.c trampoline_setup.c
 clear_cache.c enable_execute_stack.c
 )

SET( ARM_Functions
 adddf3vfp.c addsf3vfp.c bswapdi2.c bswapsi2.c divdf3vfp.c
 divsf3vfp.c eqdf2vfp.c eqsf2vfp.c extendsfdf2vfp.c
 fixdfsivfp.c fixsfsivfp.c fixunsdfsivfp.c fixunssfsivfp.c
 floatsidfvfp.c floatsisfvfp.c floatunssidfvfp.c floatunssisfvfp.c
 gedf2vfp.c gesf2vfp.c gtdf2vfp.c gtsf2vfp.c
 ledf2vfp.c lesf2vfp.c ltdf2vfp.c ltsf2vfp.c
 muldf3vfp.c mulsf3vfp.c
 nedf2vfp.c negdf2vfp.c negsf2vfp.c nesf2vfp.c
 subdf3vfp.c subsf3vfp.c truncdfsf2vfp.c unorddf2vfp.c unordsf2vfp.c
 modsi3.c umodsi3.c udivsi3.c divsi3.c switch.c
 )

#FOREACH( LOOP_VAR ${Achitectures} )
# See ARCHIVE_OUTPUT_DIRECTORY docs.
#${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${LOOP_VAR}
#ENDFOREACH

ADD_LIBRARY( ${PROJECT_NAME}-Generic STATIC ${Generic_SRCS} )
#ADD_LIBRARY( ${PROJECT_NAME}-i386 STATIC ${i386_Functions} )

# [[debug|optimized|general]
#TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${PROJECT_NAME}-Common optimized ${PROJECT_NAME}-i386 )

=======
# First, add the subdirectories which contain feature-based runtime libraries
# and several convenience helper libraries.
add_subdirectory(asan)
add_subdirectory(interception)
add_subdirectory(sanitizer_common)

# FIXME: Add support for the profile library.


# The top-level lib directory contains a large amount of C code which provides
# generic implementations of the core runtime library along with optimized
# architecture-specific code in various subdirectories.

set(GENERIC_SOURCES
  absvdi2.c
  absvsi2.c
  absvti2.c
  adddf3.c
  addsf3.c
  addvdi3.c
  addvsi3.c
  addvti3.c
  apple_versioning.c
  ashldi3.c
  ashlti3.c
  ashrdi3.c
  ashrti3.c
  clear_cache.c
  clzdi2.c
  clzsi2.c
  clzti2.c
  cmpdi2.c
  cmpti2.c
  comparedf2.c
  comparesf2.c
  ctzdi2.c
  ctzsi2.c
  ctzti2.c
  divdc3.c
  divdf3.c
  divdi3.c
  divmoddi4.c
  divmodsi4.c
  divsc3.c
  divsf3.c
  divsi3.c
  divti3.c
  divxc3.c
  enable_execute_stack.c
  eprintf.c
  extendsfdf2.c
  ffsdi2.c
  ffsti2.c
  fixdfdi.c
  fixdfsi.c
  fixdfti.c
  fixsfdi.c
  fixsfsi.c
  fixsfti.c
  fixunsdfdi.c
  fixunsdfsi.c
  fixunsdfti.c
  fixunssfdi.c
  fixunssfsi.c
  fixunssfti.c
  fixunsxfdi.c
  fixunsxfsi.c
  fixunsxfti.c
  fixxfdi.c
  fixxfti.c
  floatdidf.c
  floatdisf.c
  floatdixf.c
  floatsidf.c
  floatsisf.c
  floattidf.c
  floattisf.c
  floattixf.c
  floatundidf.c
  floatundisf.c
  floatundixf.c
  floatunsidf.c
  floatunsisf.c
  floatuntidf.c
  floatuntisf.c
  floatuntixf.c
  gcc_personality_v0.c
  int_util.c
  lshrdi3.c
  lshrti3.c
  moddi3.c
  modsi3.c
  modti3.c
  muldc3.c
  muldf3.c
  muldi3.c
  mulodi4.c
  mulosi4.c
  muloti4.c
  mulsc3.c
  mulsf3.c
  multi3.c
  mulvdi3.c
  mulvsi3.c
  mulvti3.c
  mulxc3.c
  negdf2.c
  negdi2.c
  negsf2.c
  negti2.c
  negvdi2.c
  negvsi2.c
  negvti2.c
  paritydi2.c
  paritysi2.c
  parityti2.c
  popcountdi2.c
  popcountsi2.c
  popcountti2.c
  powidf2.c
  powisf2.c
  powitf2.c
  powixf2.c
  subdf3.c
  subsf3.c
  subvdi3.c
  subvsi3.c
  subvti3.c
  trampoline_setup.c
  truncdfsf2.c
  ucmpdi2.c
  ucmpti2.c
  udivdi3.c
  udivmoddi4.c
  udivmodsi4.c
  udivmodti4.c
  udivsi3.c
  udivti3.c
  umoddi3.c
  umodsi3.c
  umodti3.c
  )

if(CAN_TARGET_X86_64)
  add_library(clang_rt.x86_64 STATIC
    x86_64/floatdidf.c
    x86_64/floatdisf.c
    x86_64/floatdixf.c
    x86_64/floatundidf.S
    x86_64/floatundisf.S
    x86_64/floatundixf.S
    ${GENERIC_SOURCES}
    )
  set_target_properties(clang_rt.x86_64 PROPERTIES COMPILE_FLAGS "-std=c99 ${TARGET_X86_64_CFLAGS}")
endif()
if(CAN_TARGET_I386)
  add_library(clang_rt.i386 STATIC
    i386/ashldi3.S
    i386/ashrdi3.S
    i386/divdi3.S
    i386/floatdidf.S
    i386/floatdisf.S
    i386/floatdixf.S
    i386/floatundidf.S
    i386/floatundisf.S
    i386/floatundixf.S
    i386/lshrdi3.S
    i386/moddi3.S
    i386/muldi3.S
    i386/udivdi3.S
    i386/umoddi3.S
    ${GENERIC_SOURCES}
    )
  set_target_properties(clang_rt.i386 PROPERTIES COMPILE_FLAGS "-std=c99 ${TARGET_I386_CFLAGS}")
endif()
>>>>>>> 853733e772b2885d93fdf994dedc4a1b5dc1369e
