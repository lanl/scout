#include <stdio.h>

extern "C" void __scrt_legion_setup_mesh(void *p, int w, int h, int d) {
	printf("in setup mesh w=%d h=%d d=%d\n", w, h, d);
}

extern "C" void __scrt_legion_add_field(void *p, char *n, int t) {
	printf("in add field n=%s t=%d\n", n, t);
}

