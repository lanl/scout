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
  aMesh M[3, 3]; // 3x3 cells

  // Enumerate indices

  forall cells c in M
  {
    c.c_index = positionx() + 3 * positiony();            // Cells have positive values, 0 - 8
  }//end forall cells loop

  forall vertices v in M
  {
    v.v_index = -1 * (positionx() + (3+1) * positiony()); // Vertices have negative values, 0 - (-15)
  }// end forall vertices loop


  int numCells = 9;
  int numVerts = 16;

  int exp1[numCells];
  int out1[numCells];

 /* Here is how the cells and vertices are enumerated
  *
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
  * ******Although, vertices are negative values.********
  */

  // Do cell shift to the east. Loop occurs on east exterior cells.
  int j = 0;
  forall cells c in M
  {
    out1[j++] = cshift(c.c_index, 1, 0);
  }// end forall cells loop

  for (int i = 0; i < numCells; i++)
    exp1[i] = 3 * (i/3) + (i+1) % 3;
  //exp1 = {1, 2, 0,  4, 5, 3,  7, 8, 6}

  for (int i = 0; i < numCells; i++)
    assert(exp1[i] == out1[i] && "bad value in rank=1");




  // Do cell shift to the north. Loop occurs on north exterior cells.
  j = 0;
  forall cells c in M
  {
    out1[j++] = cshift(c.c_index, 0, 1);
  }// end forall cells loop

  for (int i = 0; i < numCells; i++)
    exp1[i] = 3 * ((i/3 + 1) % 3) + i % 3;
  //exp1 = {3, 4, 5,  6, 7, 8,  0, 1, 2}

  for (int i = 0; i < numCells; i++)
    assert(exp1[i] == out1[i] && "bad value in rank=2 north");




  int exp2[numVerts];
  int out2[numVerts];


  // Do vertex shift to the east. Loop occurs on east exterior vertices.
  j = 0;
  forall vertices v in M
  {
    out2[j++] = cshift(v.v_index, 1, 0);
  }// end forall vertices loop

  for (int i = 0; i < numVerts; i++)
    exp2[i] = -1*(4 * (i/4) + (i+1) % 4);
  //exp2 = {-1, -2, -3,  0,  -5, -6, -7, -4,  -9, -10, -11, -8,  -13, -14, -15, -12}

  for (int i = 0; i < numVerts; i++) {
    printf("%d %d\n", exp2[i], out2[i]);
    assert(exp2[i] == out2[i] && "bad value in rank=2 east"); 
  }
  return 0;
}// end main() method
