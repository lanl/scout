/**
 * Ian Sohl - 2015
 * Copyright (c) 2015      Los Alamos National Security, LLC
 *                         All rights reserved.
 * Legion Image Composition - Qt Window File
 */

#include <QtGui>
#include <iostream>
#include <sys/time.h>
#include "imagecomposer.h"


double getTime(){
	/**
	 * Get current system time
	 */
	struct timeval curtime;
	gettimeofday(&curtime,NULL);
	return curtime.tv_sec+(curtime.tv_usec/1000000.0);
}

static const QSize resultSize(1000, 1000); // Window Size

ImageComposer::ImageComposer(ImageConnector *conn){
	/**
	 * Create a Qt Window for displaying images and transmitting user commands
	 */
	qRegisterMetaType<Movement>("Movement"); 	// Register Movement struct with Qt
	setFocusPolicy(Qt::ClickFocus);				// Setup window focus behavior
	setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);	// Qt window size behavior
	QObject::connect(conn, SIGNAL(transmitImage(int *,Movement,int,int)), this, SLOT(loadImage(int *,Movement,int,int))); // Connect the image loading slots and signals
	QObject::connect(this, SIGNAL(transmitMovement(Movement)), conn, SLOT(receiveMovement(Movement)));	// Connect the user interaction slots and signals
	QObject::connect( qApp, SIGNAL(lastWindowClosed()), conn, SLOT(receiveDone()) ); // Connect the window close slots and signals

	QLabel *resultLabel = new QLabel();	// Place to put images
	resultLabel->setMinimumWidth(resultSize.width());
	resultLabel->setMinimumHeight(resultSize.height());

	QGridLayout *mainLayout = new QGridLayout; 	// Standard adaptable grid windowing system
	mainLayout->addWidget(resultLabel, 0, 0);	// Only one thing to add
	mainLayout->setSizeConstraint(QLayout::SetFixedSize); // Fill screen
	setLayout(mainLayout);						// Place layout

	this->setAutoFillBackground(true);			// Autofill background between frames (no smearing)
	QPalette p( this->palette());				// Define background color
	p.setColor( QPalette::Window, QColor(Qt::black)); // Black
	this->setPalette(p);						// Set background color

	setWindowTitle(tr("Image Composition"));	// Set window head text
	QTimer *timer = new QTimer(this);			// Prepare frame counter
	connect(timer, SIGNAL(timeout()), this, SLOT(update())); // Connect framerate timer with update function
	timer->start(50);							// Run every 50 ms

//	PVMMatrix.perspective(60,45,0.0001,1000);
	Perspective.ortho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);	// Orthographic projection for renderer
	Perspective.scale(0.002);							// Rescale the entire scene (should be in model projection)
	Perspective.translate(250,250,0);					// Move to center of screen
	Perspective.rotate(-90,0,0,1);						// Flip so upright

	lrAmount = 0;	// Initialize left-right rotation angle (degrees)
	udAmount = 0;	//		up-down rotation angle

	updatePVM();	// Initial modelview update

	this->counter = 0;	// Frame count
	this->moved = true;	// Creation is movement...
	sendMovement();		// Tell Legion to create an image
}

void ImageComposer::updatePVM(){
	/**
	 * Populate the MovelView Matrix with current values
	 */
	PVMMatrix.setToIdentity(); 	// Reset to identity matrix
	PVMMatrix *= Perspective;  	// Add in perspective transformation

	udRotate.setToIdentity();  	// Reset rotation matrices
	udAmount = udAmount % 360;	// Degrees
	udRotate.rotate(udAmount,0,1,0); // Set rotation matrix
	PVMMatrix *= udRotate;		// Add rotation to modelview

	lrRotate.setToIdentity();	// Left-Right after Up-Down
	lrAmount = lrAmount % 360;
	lrRotate.rotate(lrAmount,1,0,0);
	PVMMatrix *= lrRotate;

	PVMMatrix.translate(-150,-150,-150);	// Slide the origin to the center of the volume instead of the corner
}

void ImageComposer::keyPressEvent(QKeyEvent * event){
	/**
	 * Handle a user-input keyboard event
	 */
	if(event->key()==Qt::Key_Up){
//		udRotate.rotate(10,0,1,0);
		udAmount -= 10; // Change by 10 degrees
		updatePVM();	// Update the modelview
		this->moved = true;	// Indicate a new frame is needed
		sendMovement();	// Tell Legion
	}
	else if(event->key()==Qt::Key_Down){
//		udRotate.rotate(-10,0,1,0);
		udAmount += 10;
		updatePVM();
		this->moved = true;
		sendMovement();
	}
	else if(event->key()==Qt::Key_Left){
//		lrRotate.rotate(-10,1,0,0);
		lrAmount -= 10;
		updatePVM();
		this->moved = true;
		sendMovement();
	}
	else if(event->key()==Qt::Key_Right){
		lrAmount += 10;
//		lrRotate.rotate(10,1,0,0);
		updatePVM();
		this->moved = true;
		sendMovement();
	}
//	else if(event->key()==Qt::Key_Plus){
//		this->moved = true;
//		sendMovement();
//	}
//	else if(event->key()==Qt::Key_Minus){
//		this->moved = true;
//		sendMovement();
//	}
}

void ImageComposer::mousePressEvent(QMouseEvent * event){
	/**
	 * Handle a user input through mouse click
	 */
	// FILL IN
	this->moved = true;
	sendMovement();
}

void ImageComposer::paintEvent(QPaintEvent* evt) {
	/**
	 * Repaint the screen with new or old data
	 */
	QPainter painter(this);	// Get reference to paint context
	if(this->moved){		// Only check if there's been movement
		float* invPVM = PVMMatrix.transposed().inverted().data(); // Construct and inverse PV Matrix for validation
		Movement mov; // Place invPVM in Movement struct
		for(int i = 0; i < 16; ++i){
			mov.invPVM[i] = (float)invPVM[i]; // Copy individual members manually
		}
		for(unsigned int i = 0; i<imgs.size(); ++i){	// Check through list of images for the right one
			if(imgs[i].mov==mov){	// If the correct one is found
				std::cout << "Painting" << std::endl;
				this->img = imgs[i].img.scaled(resultSize); // Scale to the correct size
				this->painttime = true;	// Tell painter to begin painting the first time
				this->moved = false;	// Reset movement variable
				break;
			}
			if(imgs[i].receivedCount<this->counter-1000){ // If images have been cached for too long, delete them
				std::cout << "Erased Something" << std::endl;
				imgs.erase(imgs.begin()+i--);	// Remove and continue checking
			}
		}
	}
	if(this->painttime){ // If we have an image to display
		painter.drawImage(QPoint(0,0), this->img); // Draw into label
	}
}

QSize ImageComposer::sizeHint() const {
	/**
	 * Get total size
	 */
	return m_size;
}

void ImageComposer::sendMovement(){

	/**
	 * Send current image metadata to the interface
	 */
	float* invPVM = PVMMatrix.transposed().inverted().data(); // Construct an inverse PV Matrix

	Movement mov;
	for(int i = 0; i < 16; ++i){
		mov.invPVM[i] = (float)invPVM[i]; // Copy manually into Movement struct
	}
	mov.xdat = PVMMatrix.transposed().data()[8]; // Get the composition order element
	for(unsigned int i = 0; i<imgs.size(); ++i){
		if(imgs[i].mov==mov){	// Check through cache first
			std::cout << "Already Have" << std::endl;
			return;
		}
	}
	std::cout << "Transmitting" << std::endl;
	ImageComposer::transmitMovement(mov);	// Else send to the connection object SLOT
}


void ImageComposer::loadImage(int *vals, Movement mov, int width, int height){
	/**
	 * Receive a new image from Legion and add to cache
	 */
	std::cout << "Received" << std::endl;
	this->counter++;	// Image received counter
	QImage resultImage = QImage(QSize(width,height), QImage::Format_ARGB32_Premultiplied); // QImage construct to be placed in
	int xpix = 0;
	int ypix = 0;
	for(int i = 0; i < width*height*4; i+=4){ // Step through new pixel data
		QRgb val = qRgba(vals[i+0],vals[i+1],vals[i+2],vals[i+3]); // Create Qt color type
		resultImage.setPixel(xpix,ypix,val);	// Add to QImage
		++xpix;
		if(xpix>=width){ // De-linearize
			++ypix;
			xpix = 0;
		}
	}
	ImageCounter nimg = {resultImage,mov,this->counter}; // Create a metadata object for the image
	this->imgs.push_back(nimg); // Add to cache
}

QPoint ImageComposer::imagePos(const QImage &image) const{
	/**
	 * Get image position
	 */
	return QPoint((resultSize.width() - image.width()) / 2,
			(resultSize.height() - image.height()) / 2);
}

ImageConnector::ImageConnector(){
	/**
	 * Interface object for connection between Qt and Legion
	 */
	this->done = false;
}

void ImageConnector::sendImage(int *vals, Movement mov, int width, int height){
	/**
	 * Send a new image to Qt
	 */
	ImageConnector::transmitImage(vals, mov,width,height);
}

void ImageConnector::receiveMovement(Movement mov){
	/**
	 * Get new input data from Qt
	 */
	this->mov = mov;
}

void ImageConnector::receiveDone(){
	/**
	 * Mark the window as closed
	 */
	this->done = true;
	std::cout << "All Done" << std::endl;
}
