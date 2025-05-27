#include "CursorShape.h"
#include <QPixmap>

QCursor getCursorShape(const CursorShapeType& cursor_shape){
	if (cursor_shape==CURSOR_PENCIL) {
		QPixmap pixmap(QString("./Cursor/pencil.png"));
		QCursor cursor(pixmap.scaled(35,35),2,33);
		return cursor;
	} else if (cursor_shape==CURSOR_ERASER) {
		QPixmap pixmap(QString("./Cursor/eraser.png"));
		QCursor cursor(pixmap.scaled(35,35),2,33);
		return cursor;
	} else if (cursor_shape==CURSOR_BRUSH) {
		QPixmap pixmap(QString("./Cursor/brush.png"));
		QCursor cursor(pixmap.scaled(35,35),2,33);
		return cursor;
	}

	QCursor cursor;
	switch(cursor_shape){
			case CURSOR_ARROW: 
				cursor.setShape(Qt::ArrowCursor); 
				break;
			case CURSOR_OPEN_HAND: 
				cursor.setShape(Qt::OpenHandCursor); 
				break;
			case CURSOR_CLOSED_HAND: 
				cursor.setShape(Qt::ClosedHandCursor); 
				break;
	}

	return cursor;
}