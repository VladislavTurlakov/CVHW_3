# Задание 4.1
# Написать программу, которая читает кадр с камеры, корректирует перспективу, распознает QR код.
# Провести оценки максимального угла на котором распознается QR код без коррекции и с коррекцией перспективы

import cv2
import numpy as np
import math

def order_points(points):
    """Упорядочивание 4 точки прямоугольника в строгом порядке:
    [верхний-левый, верхний-правый, нижний-правый, нижний-левый]."""

    points = np.array(points, dtype="float32")

    s = points.sum(axis=1)
    rect = np.zeros((4, 2), dtype="float32")

    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect  # возвращаем упорядоченные точки

def correct_perspective(frame, rect):
    """Коррекция перспективы."""

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    size = 300

    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (size, size))

    return warped

def estimate_angle(rect):
    """Оценка угла наклона четырехугольникаа относительно горизонтали."""
    (tl, tr, _, _) = rect
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    return math.degrees(math.atan2(dy, dx))

def smooth_bbox(prev, current, alpha=0.7):
    """Сглаживание координат ограничивающего прямоугольника между кадрами."""
    if prev is None:
        return current
    return alpha * prev + (1 - alpha) * current

def main():
    """Главная функция программы"""
    cap = cv2.VideoCapture(0)
    detector = cv2.QRCodeDetector()

    prev_bbox = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data, bbox, _ = detector.detectAndDecode(frame)

        cv2.putText(frame, "Mode: perspective correction", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        if bbox is not None:
            bbox = bbox.astype(np.float32)

            # сглаживание
            bbox = smooth_bbox(prev_bbox, bbox)
            prev_bbox = bbox

            rect = order_points(bbox[0])

            # рисуем рамку
            for i in range(4):
                pt1 = tuple(rect[i].astype(int))
                pt2 = tuple(rect[(i + 1) % 4].astype(int))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

            # коррекция для отображения
            corrected = correct_perspective(frame, rect)

            angle = estimate_angle(rect)

            text = f"Data: {data}" if data else "Data: None"

            cv2.putText(frame, text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"Angle: {angle:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Corrected QR", corrected)

        cv2.imshow("QR Scanner", frame)

        if cv2.waitKey(1) == 27: # ESCAPE
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
