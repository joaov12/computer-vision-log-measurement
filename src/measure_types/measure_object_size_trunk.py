import sys
sys.dont_write_bytecode = True

import cv2
import numpy as np

IMAGE_FILE = "tronco1.jpeg"  

def detect_tree_trunk(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 21, 2)
    
    kernel = np.ones((7,7), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if area > 5000 and circularity > 0.4:  
            valid_contours.append(cnt)
    
    valid_contours.sort(key=cv2.contourArea, reverse=True)
    
    return valid_contours, binary, cleaned

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

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary", binary)

parameters = cv2.aruco.DetectorParameters()
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10
parameters.adaptiveThreshConstant = 7

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

cv2.putText(output_img, "Status: " + ("Marcador detectado!" if corners else "Marcador não encontrado"), 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if corners else (0, 0, 255), 2)

print("\nTentando detectar marcador ArUco...")
print(f"Encontrados: {len(corners) if corners else 0} marcadores")
if ids is not None:
    print(f"IDs detectados: {ids.flatten()}")

if not corners:
    print("Erro: Marcador ArUco não encontrado na imagem")
    print("Dicas:")
    print("1. Verifique se o marcador está bem visível na imagem")
    print("2. Certifique-se de que o marcador está bem iluminado")
    print("3. Evite ângulos muito inclinados")
    print("4. O marcador deve ser do tipo 5x5 (DICT_5X5_50)")

pixel_cm_ratio = None
if corners and len(corners) > 0:
    if ids is not None:
        cv2.putText(output_img, f"ID detectado: {ids[0][0]}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    int_corners = np.int32(corners)
    cv2.polylines(output_img, int_corners, True, (0, 255, 0), 5)

    aruco_perimeter = cv2.arcLength(corners[0], True)
    pixel_cm_ratio = aruco_perimeter / 94
else:
    cv2.putText(output_img, "Usando escala aproximada!", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

debug_img = img.copy()
trunk_contours, binary, cleaned = detect_tree_trunk(img)

print(f"\nDetecção do tronco:")
print(f"Encontrados {len(trunk_contours)} contornos possíveis")

debug_binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
debug_cleaned = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

cv2.drawContours(debug_binary, trunk_contours, -1, (0, 255, 0), 2)
cv2.drawContours(debug_cleaned, trunk_contours, -1, (0, 255, 0), 2)

debug_row1 = np.hstack([img, debug_binary])
debug_row2 = np.hstack([debug_cleaned, debug_img])
debug_view = np.vstack([debug_row1, debug_row2])
debug_view = cv2.resize(debug_view, (0,0), fx=0.5, fy=0.5)
cv2.imshow("Etapas de Processamento", debug_view)

for i, cnt in enumerate(trunk_contours):
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    
    area = cv2.contourArea(cnt)
    print(f"\nContorno {i+1}:")
    print(f"- Área: {area:.0f} pixels²")
    print(f"- Raio: {radius} pixels")
    
    diameter_cm = (radius * 2) / pixel_cm_ratio
    
    cv2.circle(output_img, center, radius, (255, 0, 0), 3)
    
    cv2.line(output_img, 
             (int(x - radius), int(y)), 
             (int(x + radius), int(y)), 
             (0, 0, 255), 2)
    
    cv2.putText(output_img, 
                f"Diametro: {diameter_cm:.1f} cm", 
                (int(x - radius), int(y - 20)), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

marker_size = 23.5
cv2.putText(output_img, f"Marcador: {marker_size}x{marker_size}cm", (10, 90), 
           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

output_filename = "medidas_" + IMAGE_FILE
cv2.imwrite(output_filename, output_img)
print(f"Imagem com medições salva como: {output_filename}")

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", min(width, 1200), min(height, 800))

cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)
debug_img = np.hstack([gray, binary])
cv2.imshow("Debug", debug_img)

cv2.imshow("Image", output_img)

print("\nControles:")
print("- Use o mouse para redimensionar as janelas")
print("- Janela 'Debug' mostra etapas intermediárias da detecção")
print("- Pressione qualquer tecla para fechar")
cv2.waitKey(0)
cv2.destroyAllWindows()
