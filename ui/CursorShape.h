#ifndef CURSOR_SHAPE_H
#define CURSOR_SHAPE_H

#include <QCursor>

typedef enum{
	CURSOR_ARROW = 0,
	CURSOR_OPEN_HAND,
	CURSOR_CLOSED_HAND,
	CURSOR_PENCIL,
	CURSOR_ERASER,
	CURSOR_BRUSH
} CursorShapeType;

QCursor getCursorShape(const CursorShapeType& cursor_shape);

#endif //CURSOR_SHAPE_H