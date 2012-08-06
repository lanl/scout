/**
 * hpgv_util.c
 *
 * Copyright (c) 2008 Hongfeng Yu
 *
 * Contact:
 * Hongfeng Yu
 * hfstudio@gmail.com
 * 
 * 
 * All rights reserved.  May not be used, modified, or copied 
 * without permission.
 *
 */

#include "scout/Runtime/volren/hpgv/hpgv_util.h"

namespace scout {

/**
 * hpgv_msg
 *
 */
void 
hpgv_msg(const char *fmt, ...)
{   
    va_list ap;
    char buf[MAXLINE];
    
    va_start(ap, fmt);
    vsnprintf(buf, MAXLINE, fmt, ap);          
    fputs(buf, stderr);
}



/**
 * hpgv_msg_p
 *
 */
void 
hpgv_msg_p(int id, int root, const char *fmt, ...)
{   
    va_list ap;
    char buf[MAXLINE];
    
    if (id != root) {
        return;
    }
    
    va_start(ap, fmt);
    vsnprintf(buf, MAXLINE, fmt, ap);          
    fputs(buf, stderr);
}
 
}
