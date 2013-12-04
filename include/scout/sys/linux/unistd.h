//workaround linux unistd.h having __block defined
#ifdef __linux__
    #ifdef __block
       #undef __block
       #include_next "unistd.h"
       #define __block __attribute__((__blocks__(byref)))
    #else
       #include_next "unistd.h"
    #endif
#else
       #include_next "unistd.h"
#endif
