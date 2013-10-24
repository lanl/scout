//===----------------------------------------------------------------------===//
//
// This file defines properties of all Scout intrinsics.
//
//===----------------------------------------------------------------------===//

// Reminder for memory properties:
// 
//   Memory properties.  An intrinsic is allowed to have at most one of
//   these properties set.  They are listed from the most aggressive (best to
//   use if correct) to the least aggressive.  If no property is set, the worst //   case is assumed (it may read and write any memory it can get access to and
//   it may have other side effects).
// 
//   * IntrNoMem - The intrinsic does not access memory or have any other side
//     effects.  It may be CSE'd deleted if dead, etc.
//   * IntrReadArgMem - This intrinsic reads only from memory that one of its
//     pointer-typed arguments points to, but may read an unspecified amount.
//   * IntrReadMem - This intrinsic reads from unspecified memory, so it cannot //     be moved across stores.  However, it can be reordered otherwise and can 
//     be deleted if dead.
//   * IntrReadWriteArgMem - This intrinsic reads and writes only from memory
//     that one of its arguments points to, but may access an unspecified 
//     amount.  The reads and writes may be volatile, but except for this it has //     no other side effects.
//   * No properties 


// ----- heap memory related operations
// 
//   NOTE: memory allocators may do some funky stuff with 
//   memory so we will simply make them worst case (they 
//   may read and write memory anywhere and also have side 
//   effects).

// memalloc -- allocate heap memory.
//   NOTE: allocators may do some funky stuff with memory so 
//   we will simply make them worst case (they may read and 
//   write memory anywhere and also have side effects).
def int_memalloc   : Intrinsic< [llvm_anyptr_ty],
                                [llvm_i64_ty],
                                [] >;

// memalloca -- allocate aligned heap memory.
def int_memalloca  : Intrinsic< [llvm_anyptr_ty],
                                [llvm_i64_ty, llvm_i64_ty],
                                [] >;

// memfree -- free allocated memory 
def int_memfree    : Intrinsic< [],
                                [llvm_anyptr_ty],
                                [NoCapture<0>] >;


def int_position    : Intrinsic< [llvm_v4i32_ty],
                                 [],
                                 [] >; 

def int_positionx    : Intrinsic< [llvm_i32_ty],
                                 [],
                                 [] >; 
                                 
def int_positiony    : Intrinsic< [llvm_i32_ty],
                                 [],
                                 [] >; 
                                 
def int_positionz    : Intrinsic< [llvm_i32_ty],
                                 [],
                                 [] >; 

def int_positionl    : Intrinsic< [llvm_i32_ty],
                                 [],
                                 [] >;
                                                                   