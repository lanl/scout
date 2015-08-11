#include "stdio.h"
#include "assert.h"


uniform mesh aMesh
{
  cells:
    int c_index;
  vertices:
    int v_index;
};// end mesh definition


int main()
{
  aMesh M[3, 4]; // 3 x 4 cells, 4 x 5 vertices
  
  // Enumerate indices
  
  forall cells c in M
  {
    c.c_index = positionx() + 3 * positiony();     // Cells have values, 0-11
  }//end forall cells loop
  
  forall vertices v in M
  {
    v.v_index = positionx() + (3+1) * positiony(); // Vertices have values 0-19
  }// end forall vertices loop
  
  
  int numCells = 3*4;
  int numVerts = 4*5;
  
  int exp[numVerts];
  int out[numVerts];
  
 /* Here is how the cells and vertices are enumerated
  *
       16------------------17------------------18------------------19
  *    |                   |                   |                   |
  *    |                   |                   |                   |
  *    |                   |                   |                   |
  *    |         9         |         10        |         11        |
  *    |                   |                   |                   |
  *    |                   |                   |                   |
  *    |                   |                   |                   |
  *    12------------------13------------------14------------------15
  *    |                   |                   |                   |
  *    |                   |                   |                   |
  *    |                   |                   |                   |
  *    |         6         |         7         |         8         |
  *    |                   |                   |                   |
  *    |                   |                   |                   |
  *    |                   |                   |                   |
  *    8-------------------9-------------------10------------------11
  *    |                   |                   |                   |
  *    |                   |                   |                   |
  *    |                   |                   |                   |
  *    |         3         |         4         |         5         |
  *    |                   |                   |                   |
  *    |                   |                   |                   |
  *    |                   |                   |                   |
  *    4-------------------5-------------------6-------------------7
  *    |                   |                   |                   |
  *    |                   |                   |                   |
  *    |                   |                   |                   |
  *    |         0         |         1         |         2         |
  *    |                   |                   |                   |
  *    |                   |                   |                   |
  *    |                   |                   |                   |
  *    0-------------------1-------------------2-------------------3
  *
  */
  
  // Do vertex shift to the east.
  int j = 0;
  forall vertices v in M
  {
    out[j++] = cshift(v.v_index, 1, 0);
  }// end forall vertices loop
  
  for (int i = 0; i < numVerts; i++)
    exp[i] = 4 * (i/4) + (i+1) % 4;
  
  for (int i = 0; i < numVerts; i++)
    assert(exp[i] == out[i] && "bad value in test 1");
  // Output: 7/29/15
  // exp = {1 2 3 0   5 6 7 4   9  10 11 8    13 14 15 12   17 18 19 16}
  // out = {1 2 3 0   6 7 8 5   11 12 13 10   16 17 18 15   [garbage]  }
  
  
  // Do vertex shift to the north. Loop occurs on north exterior vertices
  j = 0;
  forall vertices v in M
  {
    out[j++] = cshift(v.v_index, 0, 1);
  }// end forall vertices loop
  
  for (int i = 0; i < numVerts; i++)
    exp[i] = 4 * ((i/4 + 1) % 5) + i % 4;
  
  for (int i = 0; i < numVerts; i++)
    assert(exp[i] == out[i] && "bad value in test 2");
  // Output: 7/29/15
  // exp = {4 5 6 7   8  9  10 11   12 13 14 15   16 17 18 19   0 1 2 3}
  // out = {5 6 7 8   10 11 12 13   15 16 17 18   [garbage]     0 1 2 3}
  
  
  
  // Do vertex shift to the west. Loop occurs on west exterior vertices
  j = 0;
  forall vertices v in M
  {
    out[j++] = cshift(v.v_index, -1, 0);
  }
  
  for (int i = 0; i < numVerts; i++)
    exp[i] = 4 * (i/4) + (i-1+4) % 4;
  
  for (int i = 0; i < numVerts; i++)
    assert(exp[i] == out[i] && "bad value in test 3");
  // Output 7/29/15
  // exp = {3 0 1 2   7 4 5 6   11  8  9 10   15 12 13 14   19 16 17 18}
  // out = {3 0 1 2   8 5 6 7   13 10 11 12   18 15 16 17   [garbage]}
  
  // Do vertex shift to the south. Loop occurs on south exterior vertices
  j = 0;
  forall vertices v in M
  {
    out[j++] = cshift(v.v_index, 0, -1);
  }// end forall vertices loop
  
  for (int i = 0; i < numVerts; i++)
    exp[i] = 4 * ((i/4 -1+5) % 5) + i % 4;
  
  for (int i = 0; i < numVerts; i++)
    assert(exp[i] == out[i] && "bad value in test 4");
  // Output: 7/29/15
  // exp = {16 17 18 19   0 1 2 3   4 5 6 7    8  9 10 11   12 13 14 15}
  // out = {[garbage]     0 1 2 3   5 6 7 8   10 11 12 13   15 16 17 18}
  for (int i = 0; i < numVerts; i++)
    printf("%i ", exp[i]);
  printf("\n\n");
  for (int i = 0; i < numVerts; i++)
    printf("%i ", out[i]);
  
  return 0;
}// end main() method
