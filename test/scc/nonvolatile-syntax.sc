#include <stdio.h>

int main(int argc, char *argv[])
{
  nonvolatile int foobar = 5;
  printf("hello nonvolatile scout! (%d) \n", foobar);
  return 0;
}
