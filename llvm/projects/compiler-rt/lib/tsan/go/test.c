#include <stdio.h>

void __tsan_init();
void __tsan_fini();
void __tsan_event(int typ, int tid, void *pc, void *addr, int info);

<<<<<<< HEAD
int goCallbackCommentPc(void *pc, char **img, char **rtn, char **file, int *l) {
=======
int __tsan_symbolize(void *pc, char **img, char **rtn, char **file, int *l) {
>>>>>>> 853733e772b2885d93fdf994dedc4a1b5dc1369e
  return 0;
}

int main(void) {
  __tsan_init();
  __tsan_event(1, 0, 0, &main, 0);  // READ
  __tsan_event(11, 1, 0, 0, 0);  // THR_START
  __tsan_event(11, 0, 0, &main, 0);  // READ
  __tsan_event(13, 1, 0, 0, 0);  // THR_END
  printf("OK\n");
  __tsan_fini();
  return 0;
}
