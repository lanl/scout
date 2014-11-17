#include <stdio.h>

int main(int argc, char *argv[])
{
  persistent int foobar = 5;
  printf("hello persistent scout! (%d) \n", foobar);
  return 0;
}
