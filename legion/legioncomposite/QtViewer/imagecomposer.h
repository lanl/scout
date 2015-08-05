/**
 * Ian Sohl - 2015
 * Copyright (c) 2015      Los Alamos National Security, LLC
 *                         All rights reserved.
 * Legion Image Composition - Qt Window Header
 */

#ifndef IMAGECOMPOSER_H
#define IMAGECOMPOSER_H

#include <QPainter>
#include <QWidget>
#include <QtGui>
#include <QtCore>
#include <QElapsedTimer>
#include <vector>
#include "interact.h"
#include <QtWidgets/QLabel>
#include <QtWidgets/QGridLayout>

QT_BEGIN_NAMESPACE
class QLabel;
QT_END_NAMESPACE

struct ImageCounter{
	QImage img;
	Movement mov;
	int receivedCount;
}; /**< Image Metadata for cache */



class ImageConnector : public QThread{
	Q_OBJECT

public:
	ImageConnector();
	void sendImage(int *vals, Movement mov, int width, int height);
	Movement mov;
	bool done;

	signals:
	void transmitImage(int *vals, Movement mov, int width, int height);

	public slots:
	void receiveMovement(Movement mov);
	void receiveDone();
};


class ImageComposer : public QWidget{
	Q_OBJECT

public:
	ImageComposer(ImageConnector *conn);
	void sendMovement();
	void paintEvent(QPaintEvent* evt);
	void keyPressEvent(QKeyEvent * event);
	void mousePressEvent(QMouseEvent * event);
	QSize sizeHint() const;

	signals:
	void transmitMovement(Movement mov);

	public slots:
	void loadImage(int *vals, Movement mov, int width, int height);


	private:
	QPoint imagePos(const QImage &image) const;
	void updatePVM();
	bool painttime;
	float invPVM[16];
	int counter;
	bool moved;
	QImage img;
	QMatrix4x4 PVMMatrix;
	QMatrix4x4 Perspective;
	QMatrix4x4 lrRotate;
	QMatrix4x4 udRotate;
	int lrAmount;
	int udAmount;
	std::vector<ImageCounter> imgs;
	QSize m_size;
};

#endif
