/**
 * hpgv_socket.h
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

#ifndef HPGV_SOCKET_H
#define HPGV_SOCKET_H

namespace scout {

  extern "C" {

#define SOCKET_UNKNOW       0x0000
#define SOCKET_SERVER       0x0001
#define SOCKET_CLIENT       0x0002

#define SOCKET_PORT         7888

#define SOCKET_TAG_QUIT     0x00
#define SOCKET_TAG_QUERY    0x01
#define SOCKET_TAG_VIZ      0x02
#define SOCKET_TAG_IMAGE    0x03
#define SOCKET_TAG_SIZE     0x04
#define SOCKET_TAG_SUCCESS  0x05

#ifndef MY_ERROR
#define MY_ERROR    0
#endif

#ifndef MY_SUCCESS
#define MY_SUCCESS  1
#endif


    int hpgv_socket_init(int mode, int port);

    int hpgv_socket_get_port();

    int hpgv_socket_send(void *buffer, int size);

    int hpgv_socket_recv(void *buffer, int size);

  }

} // end namespace scout

#endif


