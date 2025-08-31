import sys
sys.dont_write_bytecode = True

import cv2
from object_detector import *
import numpy as np

# Configure o nome da sua imagem aqui
IMAGE_FILE = "capaCelular.jpeg"  # <-- Altere para o nome da sua imagem

# load aruco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

# load object detector
detector = HomogeneousBgDetector()
# load image
img = cv2.imread(IMAGE_FILE)
if img is None:
    print(f"Erro: Não foi possível carregar a imagem {IMAGE_FILE}")
    sys.exit(1)

# Mostrar dimensões originais
height, width = img.shape[:2]
print(f"\nTamanho original da imagem: {width}x{height} pixels")

# Redimensionar se a imagem for muito grande
max_dimension = 1200
scale = 1.0
if width > max_dimension or height > max_dimension:
    scale = max_dimension / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    img = cv2.resize(img, (new_width, new_height))
    print(f"Imagem redimensionada para: {new_width}x{new_height} pixels")
    print(f"Fator de escala: {scale:.2f}")

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

print("\nIniciando detecção de objetos...")
contours = detector.detect_objects(img)

if not contours:
    print("Nenhum objeto detectado! Possíveis causas:")
    print("1. Contraste insuficiente entre o objeto e o fundo")
    print("2. Objeto muito pequeno ou muito grande")
    print("3. Iluminação irregular")
else:
    print(f"Detectados {len(contours)} objetos!")

# draw objects boundaries
for cnt in contours:
    # get rect
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect

    # get width and height of the objects by applying the ratio pixel to cm
    object_width = w / pixel_cm_ratio
    object_height = h / pixel_cm_ratio

    # display rectangle
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.circle(output_img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.polylines(output_img, [box], True, (255, 0, 0), 2)
    cv2.putText(output_img, "L: {:.1f} cm".format(object_width), (int(x - 100), int(y - 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
    cv2.putText(output_img, "A: {:.1f} cm".format(object_height), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

# Mostrar o tamanho do marcador para referência
marker_size = 23.5
cv2.putText(output_img, f"Marcador: {marker_size}x{marker_size}cm", (10, 90), 
           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Salvar a imagem com as medições
output_filename = "medidas_" + IMAGE_FILE
cv2.imwrite(output_filename, output_img)
print(f"Imagem com medições salva como: {output_filename}")

# Configurar a janela de visualização
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", min(width, 1200), min(height, 800))

# Mostrar a imagem
cv2.imshow("Image", output_img)
print("\nControles:")
print("- Use o mouse para redimensionar a janela")
print("- Pressione qualquer tecla para fechar")
cv2.waitKey(0)
cv2.destroyAllWindows()
