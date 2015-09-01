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
    out[j++] = eoshift(v.v_index, 49, 1, 0);
  }// end forall vertices loop

  int exp1[] = {1,2,3,49,5,6,7,49,9,10,11,49,13,14,15,49,17,18,19,49};
 
  for (int i = 0; i < numVerts; i++) {
    assert(exp1[i] == out[i] && "bad value in test 1");
  }

  // Do vertex shift to the north. Loop occurs on north exterior vertices
  j = 0;
  forall vertices v in M
  {
    out[j++] = eoshift(v.v_index, 49, 0, 1);
  }// end forall vertices loop
  
  int exp2[] = {4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,49,49,49,49,49};
  for (int i = 0; i < numVerts; i++) {
    //assert(exp2[i] == out[i] && "bad value in test 2");
  }
  
  // Do vertex shift to the west. Loop occurs on west exterior vertices
  j = 0;
  forall vertices v in M
  {
    out[j++] = eoshift(v.v_index, 49, -1, 0);
  }
 
  int exp3[] = {49,0,1,2,49,4,5,6,49,8,9,10,49,12,13,14,49,16,17,18};
 
  for (int i = 0; i < numVerts; i++) {
    assert(exp3[i] == out[i] && "bad value in test 3");
  }
  
  // Do vertex shift to the south. Loop occurs on south exterior vertices
  j = 0;
  forall vertices v in M
  {
    out[j++] = eoshift(v.v_index, 49,  0, -1);
  }// end forall vertices loop
 
  int exp4[] = {49,49,49,49,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
 
  for (int i = 0; i < numVerts; i++) {
    assert(exp4[i] == out[i] && "bad value in test 4");
  }
  
  return 0;
}// end main() method
