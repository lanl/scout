/**
 * Ian Sohl - 2015
 * Copyright (c) 2015      Los Alamos National Security, LLC
 *                         All rights reserved.
 * Legion Image Composition - Qt Interaction File
 */

#include <QApplication>
#include "imagecomposer.h"
#include <pthread.h>
#include "interact.h"


ImageConnector *conn; // Keep a global reference to the connection object

void* interactThread(void *threadArg){
	/**
	 * Starts the Qt window in a separate thread on the same core as mainloop
	 */
	inArgs *args = (struct inArgs*) threadArg;			// Interpret thread arguments (future usability)
    Q_INIT_RESOURCE(imagecomposition);					// Tell Qt what class to source
    QApplication app(args->argc, args->argv);			// Create the QApplication object
	conn = new ImageConnector();						// Create the connection object and store the reference
	ImageComposer *composer = new ImageComposer(conn);	// Create the Qt windowing object and pass it the connection object reference
    composer->show();									// Display main window
	app.exec();											// Allow Qt to start execution (hangs here)
	pthread_exit(NULL);
}

void Interact(inArgs &args){
	/**
	 * Start a Qt Windowing Application in the same context
	 */
	pthread_t thread1; // Create a thread to launch in for concurrency
	pthread_create(&thread1,NULL,interactThread, (void*)&args); // Pass arguments along
}



void newImage(int *vals, Movement mov, int width, int height){
	/**
	 * Send a reference to an image to Qt. Must be in the same memory space as the application
	 */
	ImageConnector* conn2 = (ImageConnector*)conn; 	// Access the communicator reference
	conn2->sendImage(vals,mov,width,height);		// Activate the communicator slot
}

Movement getMovement(){
	/**
	 * Return current data state from the communicator
	 */
	ImageConnector* conn2 = (ImageConnector*)conn;
	Movement mov = conn2->mov;	// Get copy of data
	return mov;
}

bool getDone(){
	/**
	 * Check if Qt Application is closed
	 */
	ImageConnector* conn2 = (ImageConnector*)conn;
	return conn2->done;
}
