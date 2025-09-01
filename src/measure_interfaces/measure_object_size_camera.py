import sys
sys.dont_write_bytecode = True

import cv2
from object_detector import *
import numpy as np

# load aruco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

# load object detector
detector = HomogeneousBgDetector()

# buffer para média móvel das medições
BUFFER_SIZE = 10
width_buffer = []
height_buffer = []

# load cap
cap = cv2.VideoCapture(1)  # usando a webcam externa
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, img = cap.read()

    # get aruco marker
    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    
    # adicionar texto indicando se detectou o marcador
    cv2.putText(img, "Status: " + ("Marcador detectado!" if corners else "Procurando marcador..."), 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if corners else (0, 0, 255), 2)
    
    if corners:
        # mostrar qual ID foi detectado
        if ids is not None:
            cv2.putText(img, f"ID detectado: {ids[0][0]}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # draw polygon arround the marker
        int_corners = np.int32(corners)
        cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

        # aruco perimeter
        aruco_perimeter = cv2.arcLength(corners[0], True)

        pixel_cm_ratio = aruco_perimeter / 94

        marker_points = corners[0][0]
        marker_width = np.linalg.norm(marker_points[0] - marker_points[1]) / pixel_cm_ratio
        marker_height = np.linalg.norm(marker_points[1] - marker_points[2]) / pixel_cm_ratio
        
        # adicionar medidas do marcador ao buffer
        width_buffer.append(marker_width)
        height_buffer.append(marker_height)
        if len(width_buffer) > BUFFER_SIZE:
            width_buffer.pop(0)
            height_buffer.pop(0)
        
        # calcular média das últimas medições
        avg_width = sum(width_buffer) / len(width_buffer)
        avg_height = sum(height_buffer) / len(height_buffer)
        
        # mostrar tamanho do marcador
        cv2.putText(img, f"Marcador: {avg_width:.1f}x{avg_height:.1f}cm", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        contours = detector.detect_objects(img)

        # draw objects boundaries
        for cnt in contours:

            # get rect
            rect = cv2.minAreaRect(cnt)
            (x, y),(w, h), angle = rect

            # get width and height of the objects by applying the ratio pixel to cm
            object_width = w / pixel_cm_ratio
            object_height = h / pixel_cm_ratio
            
            # usar a média das últimas medições do marcador para ajustar as medidas
            correction_factor = 23.5 / avg_width  # ajusta baseado no tamanho real do marcador
            object_width *= correction_factor
            object_height *= correction_factor

            # display rectangle
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.polylines(img, [box], True, (255, 0, 0), 2)
            cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
            cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()