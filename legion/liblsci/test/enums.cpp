#include "lsci.h"
#include "legion_types.h"

using namespace LegionRuntime::HighLevel; 

int main(int argc, char** argv) {
  //PrivilegeMode
  assert (NO_ACCESS == LSCI_NO_ACCESS && "bad NO_ACCESS enum");
  assert (READ_ONLY == LSCI_READ_ONLY && "bad READ_ONLY enum");
  assert (READ_WRITE == LSCI_READ_WRITE && "bad READ_WRITE enum");
  assert (WRITE_ONLY == LSCI_WRITE_ONLY && "bad WRITE_ONLY enum");
  assert (WRITE_DISCARD == LSCI_WRITE_DISCARD && "bad WRITE_DISCARD enum");
  assert (REDUCE == LSCI_REDUCE && "bad REDUCE enum");
  assert (PROMOTED == LSCI_PROMOTED && "bad PROMOTED enum");

  //CoherenceProperty
  assert (EXCLUSIVE == LSCI_EXCLUSIVE && "bad EXCLUSIVE enum");
  assert (ATOMIC == LSCI_ATOMIC && "bad ATOMIC enum");
  assert (SIMULTANEOUS == LSCI_SIMULTANEOUS && "bad SIMULTANEOUS enum");
  assert (RELAXED == LSCI_RELAXED && "bad RELAXED enum");
}

