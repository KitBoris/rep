import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('0%_3_ml_min_200_mm_35_gr.MOV')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
min_width_height = 20
max_width_height = 70
frame_data = []
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)
    edges = cv2.Canny(cl1, 200, 250)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = frame.copy()
    centers_set = set()

    for i in range(len(contours)):
        if len(contours[i]) >= 5:
            x, y, w, h = cv2.boundingRect(contours[i])
            if min_width_height <= w <= max_width_height and min_width_height <= h <= max_width_height:
                center_x = x + w // 2
                center_y = y + h // 2
                center_tuple = (center_x, center_y)
                centers_set.add(center_tuple)

                cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                frame_data.append((frame_number, center_x, center_y, w, h))

                # if i == 0:  # Внешний контур
                #    cv2.drawContours(contour_image, contours, i, (0, 255, 0), 3)  # Зеленый цвет
                # else:  # Внутренние контуры
                #    cv2.drawContours(contour_image, contours, i, (255, 0, 0), 1)  # Синий цвет

    contour_image = cv2.resize(contour_image, None, fx=0.7, fy=0.7)
    cv2.imshow('Contours and Rectangles', contour_image)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()

with open('frame_data.txt', 'w') as f:
    for frame_info in frame_data:
        frame_num = frame_info[0]
        center_x = frame_info[1]
        center_y = frame_info[2]
        width = frame_info[3]
        height = frame_info[4]

        f.write(f"{frame_num}\t\t{center_x}\t\t{center_y}\t\t{width}\t\t{height}\n")

all_centers_x = []
all_centers_y = []

for _, cx, cy, _, _ in frame_data:
    all_centers_x.append(cx)
    all_centers_y.append(cy)

plt.figure(figsize=(10, 6))
plt.scatter(all_centers_x, all_centers_y, s=1)
plt.xlabel('Ширина кадра')
plt.ylabel('Высота кадра')
plt.xlim(0, frame_width)
plt.ylim(frame_height, 0)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# if i == 0:  # Внешний контур
#    cv2.drawContours(contour_image, contours, i, (0, 255, 0), 3)  # Зеленый цвет
# else:  # Внутренние контуры
#    cv2.drawContours(contour_image, contours, i, (255, 0, 0), 1)  # Синий цвет