import sys
sys.dont_write_bytecode = True

import cv2
from object_detector import *
import numpy as np

IMAGE_FILE = "capaCelular.jpeg"  

parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

detector = HomogeneousBgDetector()
img = cv2.imread(IMAGE_FILE)
if img is None:
    print(f"Erro: Não foi possível carregar a imagem {IMAGE_FILE}")
    sys.exit(1)

height, width = img.shape[:2]
print(f"\nTamanho original da imagem: {width}x{height} pixels")

max_dimension = 1200
scale = 1.0
if width > max_dimension or height > max_dimension:
    scale = max_dimension / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    img = cv2.resize(img, (new_width, new_height))
    print(f"Imagem redimensionada para: {new_width}x{new_height} pixels")
    print(f"Fator de escala: {scale:.2f}")

output_img = img.copy()

corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

cv2.putText(output_img, "Status: " + ("Marcador detectado!" if corners else "Marcador não encontrado"), 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if corners else (0, 0, 255), 2)

if not corners:
    print("Erro: Marcador ArUco não encontrado na imagem")
    cv2.imshow("Image", output_img)
    cv2.waitKey(0)
    sys.exit(1)

if ids is not None:
    cv2.putText(output_img, f"ID detectado: {ids[0][0]}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

int_corners = np.int32(corners)
cv2.polylines(output_img, int_corners, True, (0, 255, 0), 5)

aruco_perimeter = cv2.arcLength(corners[0], True)

pixel_cm_ratio = aruco_perimeter / 94

print("\nIniciando detecção de objetos...")
contours = detector.detect_objects(img)

if not contours:
    print("Nenhum objeto detectado! Possíveis causas:")
    print("1. Contraste insuficiente entre o objeto e o fundo")
    print("2. Objeto muito pequeno ou muito grande")
    print("3. Iluminação irregular")
else:
    print(f"Detectados {len(contours)} objetos!")

for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect

    object_width = w / pixel_cm_ratio
    object_height = h / pixel_cm_ratio

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.circle(output_img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.polylines(output_img, [box], True, (255, 0, 0), 2)
    cv2.putText(output_img, "L: {:.1f} cm".format(object_width), (int(x - 100), int(y - 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
    cv2.putText(output_img, "A: {:.1f} cm".format(object_height), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

marker_size = 23.5
cv2.putText(output_img, f"Marcador: {marker_size}x{marker_size}cm", (10, 90), 
           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

output_filename = "medidas_" + IMAGE_FILE
cv2.imwrite(output_filename, output_img)
print(f"Imagem com medições salva como: {output_filename}")

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", min(width, 1200), min(height, 800))

cv2.imshow("Image", output_img)
print("\nControles:")
print("- Use o mouse para redimensionar a janela")
print("- Pressione qualquer tecla para fechar")
cv2.waitKey(0)
cv2.destroyAllWindows()
