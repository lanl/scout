/**
 * hpgv_socket.c
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include "scout/Runtime/volren/hpgv/hpgv_socket.h"

namespace scout {

int theSocket = -1;
int thePort = -1;
int theMode = SOCKET_UNKNOW;

/**
 * hpgv_socket_init
 *
 */
int hpgv_socket_init(int mode, int port)
{
    theMode = mode;

    if (port == -1) { 
        thePort = SOCKET_PORT;
    } else {
        thePort = port;
    }
    
    if (mode == SOCKET_SERVER) {
        
        theSocket = socket(AF_INET, SOCK_STREAM, 0);
    
        if (theSocket < 0) {
            perror("ERROR opening socket");
            return MY_ERROR;
        }
        
        struct sockaddr_in serv_addr;
        
        bzero((char *) &serv_addr, sizeof(serv_addr));
        
        serv_addr.sin_family      = AF_INET;
        serv_addr.sin_addr.s_addr = INADDR_ANY;
        serv_addr.sin_port        = htons(thePort);
        
        if (bind(theSocket, (struct sockaddr *) &serv_addr, 
                 sizeof(serv_addr)) < 0)
        {
            perror("ERROR on binding");
            fprintf(stderr, "%d\n", thePort);
            return MY_ERROR;
        }
        
        listen(theSocket, 1);
        
        return MY_SUCCESS;
    }

    return MY_SUCCESS;
}


/**
 * hpgv_socket_get_port
 *
 */
int hpgv_socket_get_port()
{
    return thePort;
}


/**
 * socket_open
 *
 */
int socket_open()
{
    if (theMode == SOCKET_CLIENT) {
        
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        
        if (sockfd < 0) { 
            perror("ERROR opening socket");
            return -1;
        }
        
        
        struct hostent * server = gethostbyname("localhost");
        
        if (server == NULL) {
            perror("ERROR no such host\n");
            return -1;
        }
        
        struct sockaddr_in serv_addr;
        
        bzero((char *) &serv_addr, sizeof(serv_addr));
        
        bcopy((char *)server->h_addr, 
              (char *)&serv_addr.sin_addr.s_addr,
              server->h_length);
        
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port   = htons(thePort);
        
        if (connect(sockfd, 
                    (const struct sockaddr *)&serv_addr, 
                    sizeof(serv_addr)) < 0)
        {
            perror("ERROR connecting");
            return -1;
        }
        
        return sockfd;
        
    } else if ( theMode == SOCKET_SERVER) {
        
        struct sockaddr_in cli_addr;
        
        int clilen = sizeof(cli_addr);
        
        int newsockfd = accept(theSocket,
                              (struct sockaddr *) &cli_addr,
                              (socklen_t*)&clilen);
        
        if (newsockfd < 0) {
            perror("ERROR on accept");
            return -1;
        }
        
        return newsockfd;
    }
    
    return -1;
}

/**
 * socket_close
 *
 */
void socket_close(int sockfd)
{
    if (sockfd >= 0) {
        close(sockfd);
    }
}

/**
 * hpgv_socket_send
 *
 */
int hpgv_socket_send(void *buffer, int size)
{
    int n = 0;

    if (theMode == SOCKET_UNKNOW) {
        fprintf(stderr, "ERROR no initialization\n");
        return MY_ERROR;
    }
    
    int sockfd = socket_open();
    
    if (sockfd < 0) {
        fprintf(stderr, "ERROR no socket\n");
        return MY_ERROR;
    }
    
    n = write(sockfd, buffer, size);
    
    if (n != size) {
        fprintf(stderr, "supposed %d; actual %d\n", size, n);
        perror("ERROR writing to socket");
        return MY_ERROR;
    }
  
    char reply;

    n = read(sockfd, &reply, sizeof(char));

    socket_close(sockfd);

    if (n != sizeof(char)) {
        fprintf(stderr, "supposed %d; actual %d\n", (int)sizeof(char),  n);
        perror("ERROR writing to socket");
        return MY_ERROR;
    }

    if (reply != SOCKET_TAG_SUCCESS) {
        perror("ERROR receiving the reply.");
        return MY_ERROR;
    } 

    return MY_SUCCESS;
}


/**
 * hpgv_socket_recv
 *
 */
int hpgv_socket_recv(void *buffer, int size)
{ 
    int n = 0;

    if (theMode == SOCKET_UNKNOW) {
        fprintf(stderr, "ERROR no initialization\n");
        return MY_ERROR;
    }
    
    int sockfd = socket_open();
    
    if (sockfd < 0) {
        fprintf(stderr, "ERROR no socket\n");
        return MY_ERROR;
    }
    
    int count = 0;
    while (count < size) {
        n = read(sockfd, &(((char*)buffer)[count]), size - count);
        count += n;
    }
    
    if (count != size) {
        fprintf(stderr, "supposed %d; actual %d\n", size, count);
        perror("ERROR reading to socket");
        return MY_ERROR;
    }

    char reply = SOCKET_TAG_SUCCESS;

    n = write(sockfd, &reply, sizeof(char));

    socket_close(sockfd);

    if (n != sizeof(char)) {
        fprintf(stderr, "supposed %d; actual %d\n", (int)sizeof(char), n);
        perror("ERROR reading to socket");
        return MY_ERROR;
    }
    

    return MY_SUCCESS;
}


}
