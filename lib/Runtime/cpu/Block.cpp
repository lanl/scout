#include <stdlib.h>
#include <string.h>
#include <cassert>
#include "scout/Runtime/cpu/Block.h"

namespace scout {
  namespace cpu {

    // find total extent (product of dimensions)
    int findExtent(BlockLiteral *bl, int numDimensions) {
      size_t extent = 1;

      switch (numDimensions) {
      case 3:
        extent *= (*bl->zEnd - *bl->zStart);
      case 2:
        extent *= (*bl->yEnd - *bl->yStart);
      case 1:
        extent *= (*bl->xEnd - *bl->xStart);
      }
      return extent;
    }

    // create a BlockLiteral with just the required number of start, end pairs,
    // induction variables, and captured fields
    // numFields = captured vars + 3*Dim (start, end, and induction var)
    // layout of Fields is (start, end) pairs, then induction vars (dim_x, etc),
    // and last captured vars.
    BlockLiteral *createBlockLiteral(BlockLiteral *bl,
        int numDimensions, int numFields) {

      // numFields also includes start, end pairs
      // so need to subtract off the 6 entries from BlockLiteral
      // (xStart, xEnd, yStart, yEnd, zStart, zEnd)
      BlockLiteral *bp = (BlockLiteral *)malloc(sizeof(BlockLiteral)
          + (numFields - 6) * sizeof(void *));

      assert(bp != NULL);

      bp->isa = bl->isa;
      bp->flags = bl->flags;
      bp->reserved = bl->reserved;
      bp->invoke = bl->invoke;
      bp->descriptor = bl->descriptor;

      // offset to start of void* captured fields from Block.h
      int offset =
          bl->descriptor->size + 2 * numDimensions * sizeof(void *);

      // allocate the space for the (start, end) pairs
      switch (numDimensions) {
      case 3:
        bp->zStart = new uint32_t;
        bp->zEnd = new uint32_t;
      case 2:
        bp->yStart = new uint32_t;
        bp->yEnd = new uint32_t;
      case 1:
        bp->xStart = new uint32_t;
        bp->xEnd = new uint32_t;
      }

      // copy the ptrs to the other captured fields
      // induction vars (dim_x etc) and captured vars but not start/end pairs
      memcpy((char *) bp + offset, (char *) bl + offset,
          (numFields - 2 * numDimensions) * sizeof(void *));

      return bp;
    }

    Block::Block(BlockLiteral * bl, int numDimensions, int numFields, int start,
        int end) {
      int x, y;
      nDimensions_ = numDimensions;
      nFields_ = numFields;

      // build the blockLiteral
      blockLiteral_ = createBlockLiteral(bl, numDimensions, numFields);

      // populate the (start, end) pairs
      // split up so each thread gets a chuck that is contiguous in memory
      switch (numDimensions) {
      case 1:
        *blockLiteral_->xStart = start;
        *blockLiteral_->xEnd = end;
        break;
      case 2:
        x = (*bl->xEnd - *bl->xStart);

        *blockLiteral_->xStart = start % x;
        *blockLiteral_->xEnd = end % x;
        *blockLiteral_->yStart = start / x;
        *blockLiteral_->yEnd = end / x;
        break;
      case 3:
        x = (*bl->xEnd - *bl->xStart);
        y = (*bl->yEnd - *bl->yStart);
        *blockLiteral_->xStart = (start % (x * y)) % x;
        *blockLiteral_->xEnd = (end % (x * y)) % x;
        *blockLiteral_->yStart = (start % (x * y)) / x;
        *blockLiteral_->yEnd = end % (x * y) / x;
        *blockLiteral_->zStart = start / (x * y);
        *blockLiteral_->zEnd = end / (x * y);
        break;
      }
    }


    Block::~Block() {
      // free the space for the (start, end) pairs
      switch (nDimensions_) {
      case 3:
        delete blockLiteral_->zStart;
        delete blockLiteral_->zEnd;
      case 2:
        delete blockLiteral_->yStart;
        delete blockLiteral_->yEnd;
      case 1:
        delete blockLiteral_->xStart;
        delete blockLiteral_->xEnd;
      }

      free(blockLiteral_); // this was malloc'ed by createBlockLiteral()
    }


    BlockLiteral* Block::getBlockLiteral() {
      return (BlockLiteral *)blockLiteral_;
    }
  }
}
