import sys
sys.dont_write_bytecode = True

import cv2
from object_detector import *
import numpy as np

# Verificar se foi passado o nome da imagem
if len(sys.argv) < 2:
    print("Por favor, forneça o nome da imagem.")
    print("Uso: python measure_object_size.py nome_da_imagem.jpg")
    sys.exit(1)

# load aruco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

# load object detector
detector = HomogeneousBgDetector()

# load image
img = cv2.imread(sys.argv[1])
if img is None:
    print(f"Erro: Não foi possível carregar a imagem {sys.argv[1]}")
    sys.exit(1)

# fazer uma cópia da imagem original
output_img = img.copy()

# get aruco marker
corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

# adicionar texto indicando se detectou o marcador
cv2.putText(output_img, "Status: " + ("Marcador detectado!" if corners else "Marcador não encontrado"), 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if corners else (0, 0, 255), 2)

if not corners:
    print("Erro: Marcador ArUco não encontrado na imagem")
    cv2.imshow("Image", output_img)
    cv2.waitKey(0)
    sys.exit(1)

# mostrar qual ID foi detectado
if ids is not None:
    cv2.putText(output_img, f"ID detectado: {ids[0][0]}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# draw polygon arround the marker
int_corners = np.int32(corners)
cv2.polylines(output_img, int_corners, True, (0, 255, 0), 5)

# aruco perimeter
aruco_perimeter = cv2.arcLength(corners[0], True)

# pixel to cm ratio (perímetro real do marcador = 94cm (23.5cm * 4 lados))
pixel_cm_ratio = aruco_perimeter / 94


contours = detector.detect_objects(img)

# draw objects boundaries
for cnt in contours:

    # get rect
    rect = cv2.minAreaRect(cnt)
    (x, y),(w, h), angle = rect

    # get width and height of the objects by applying the ratio pixel to cm
    object_width = w / pixel_cm_ratio
    object_height = h / pixel_cm_ratio

    # display rectangle
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.circle(output_img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.polylines(output_img, [box], True, (255, 0, 0), 2)
    cv2.putText(output_img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
    cv2.putText(output_img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

# Mostrar o tamanho do marcador para referência
marker_size = 23.5
cv2.putText(output_img, f"Marcador: {marker_size}x{marker_size}cm", (10, 90), 
           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Salvar a imagem com as medições
output_filename = "medidas_" + sys.argv[1]
cv2.imwrite(output_filename, output_img)
print(f"Imagem com medições salva como: {output_filename}")

# Mostrar a imagem
cv2.imshow("Image", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()