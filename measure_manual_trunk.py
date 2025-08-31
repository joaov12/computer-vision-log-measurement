import sys
sys.dont_write_bytecode = True

import cv2
import numpy as np

IMAGE_FILE = "troncoSemSombra.jpeg"  

points = []
img_copy = None

def click_event(event, x, y, flags, param):
    global points, img_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img_copy, (x, y), 3, (0, 0, 255), -1)
        points.append((x, y))
        
        if len(points) == 2:
            cv2.line(img_copy, points[0], points[1], (0, 255, 0), 2)
            
        cv2.imshow("Image", img_copy)

def try_detect_marker(image):
    print("\n=== INICIANDO DETECÇÃO DO MARCADOR ===")
    print("Dimensões da imagem de entrada:")
    print(f"- Altura: {image.shape[0]} pixels")
    print(f"- Largura: {image.shape[1]} pixels")
    print(f"- Canais: {image.shape[2]}")
    print("\nParâmetros do detector:")
    
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.adaptiveThreshConstant = 7
    
    print(f"- Tamanho mínimo da janela adaptativa: {parameters.adaptiveThreshWinSizeMin}")
    print(f"- Tamanho máximo da janela adaptativa: {parameters.adaptiveThreshWinSizeMax}")
    print(f"- Passo da janela adaptativa: {parameters.adaptiveThreshWinSizeStep}")
    print(f"- Constante adaptativa: {parameters.adaptiveThreshConstant}")
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)
    
    print("\nResultados da detecção:")
    print(f"- Marcadores encontrados: {len(corners) if corners else 0}")
    if corners:
        for i, corner in enumerate(corners):
            print(f"\nMarcador {i+1}:")
            print(f"- ID: {ids[i][0] if ids is not None else 'Desconhecido'}")
            print(f"- Coordenadas dos cantos:")
            corners_array = corner[0]
            for j, point in enumerate(corners_array):
                print(f"  Canto {j+1}: ({point[0]:.1f}, {point[1]:.1f})")
            
            width = np.linalg.norm(corners_array[0] - corners_array[1])
            height = np.linalg.norm(corners_array[1] - corners_array[2])
            print(f"- Largura em pixels: {width:.1f}")
            print(f"- Altura em pixels: {height:.1f}")
            print(f"- Área em pixels²: {width * height:.1f}")
            
            # Calcular perímetro
            perimeter = cv2.arcLength(corner, True)
            print(f"- Perímetro em pixels: {perimeter:.1f}")
    else:
        print("Nenhum marcador encontrado!")
        
    print("\n=== FIM DA DETECÇÃO DO MARCADOR ===")
    return corners, ids

def main():
    global img_copy
    
    # Carregar imagem
    print("\n=== INICIANDO PROCESSAMENTO ===")
    print(f"Arquivo de entrada: {IMAGE_FILE}")
    
    img = cv2.imread(IMAGE_FILE)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem {IMAGE_FILE}")
        sys.exit(1)

    height, width = img.shape[:2]
    print("\nDimensões da imagem:")
    print(f"- Largura original: {width} pixels")
    print(f"- Altura original: {height} pixels")
    print(f"- Resolução: {width * height} pixels")
    print(f"- Proporção: {width/height:.2f}")

    max_dimension = 1200
    scale = 1.0
    if width > max_dimension or height > max_dimension:
        scale = max_dimension / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height))
        print("\nRedimensionamento:")
        print(f"- Dimensão máxima permitida: {max_dimension} pixels")
        print(f"- Nova largura: {new_width} pixels")
        print(f"- Nova altura: {new_height} pixels")
        print(f"- Nova resolução: {new_width * new_height} pixels")
        print(f"- Fator de escala: {scale:.3f}")
        print(f"- Redução de tamanho: {100 - (scale * 100):.1f}%")

    img_copy = img.copy()

    corners, ids = try_detect_marker(img)
    
    pixel_cm_ratio = None
    
    if corners and len(corners) > 0:
        print("\n=== ANÁLISE DO MARCADOR ENCONTRADO ===")
        print("Status: Marcador ArUco detectado com sucesso!")
        if ids is not None:
            print(f"ID do marcador: {ids[0][0]}")
        
        int_corners = np.int32(corners)
        cv2.polylines(img_copy, int_corners, True, (0, 255, 0), 2)
        
        corners_array = corners[0][0]
        width_pixels = np.linalg.norm(corners_array[0] - corners_array[1])
        height_pixels = np.linalg.norm(corners_array[1] - corners_array[2])
        
        print("\nDimensões do marcador:")
        print(f"- Largura em pixels: {width_pixels:.1f}")
        print(f"- Altura em pixels: {height_pixels:.1f}")
        print(f"- Proporção do marcador: {width_pixels/height_pixels:.2f}")
        
        # Calcular a razão pixel/cm (perímetro = 94cm (23.5cm * 4))
        aruco_perimeter = cv2.arcLength(corners[0], True)
        pixel_cm_ratio = aruco_perimeter / 94
        
        print("\nCálculos de escala:")
        print(f"- Perímetro em pixels: {aruco_perimeter:.1f}")
        print(f"- Perímetro real: 94 cm (23.5 cm x 4 lados)")
        print(f"- Razão pixel/cm: {pixel_cm_ratio:.2f}")
        print(f"- 1 cm = {pixel_cm_ratio:.1f} pixels")
        print(f"- 1 pixel = {1/pixel_cm_ratio:.3f} cm")
    else:
        print("\nMarcador ArUco não encontrado.")
        print("Você precisará fornecer a medida de referência manualmente.")
        ref_cm = float(input("Digite o tamanho de referência em centímetros: "))
        pixel_cm_ratio = float(input("Digite quantos pixels equivalem a essa medida: "))
        pixel_cm_ratio = pixel_cm_ratio / ref_cm

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click_event)

    print("\nInstruções:")
    print("1. Clique em dois pontos para medir o diâmetro do tronco")
    print("2. Pressione 'r' para recomeçar")
    print("3. Pressione 'q' para sair")
    print("4. Pressione 's' para salvar")

    while True:
        cv2.imshow("Image", img_copy)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            points.clear()
            img_copy = img.copy()
            if corners and len(corners) > 0:
                cv2.polylines(img_copy, int_corners, True, (0, 255, 0), 2)
        
        elif key == ord('s'):
            if len(points) == 2:
                print("\n=== CÁLCULO DO DIÂMETRO ===")
                dist_pixels = np.sqrt((points[1][0] - points[0][0])**2 + 
                                    (points[1][1] - points[0][1])**2)

                dist_cm = dist_pixels / pixel_cm_ratio
                
                print("\nPontos selecionados:")
                print(f"- Ponto 1: ({points[0][0]}, {points[0][1]})")
                print(f"- Ponto 2: ({points[1][0]}, {points[1][1]})")
                print(f"- Distância em pixels: {dist_pixels:.1f}")
                
                print("\nMedidas do tronco:")
                print(f"- Diâmetro em pixels: {dist_pixels:.1f}")
                print(f"- Diâmetro em cm: {dist_cm:.1f}")
                print(f"- Raio em cm: {dist_cm/2:.1f}")
                circumference = dist_cm * np.pi
                print(f"- Circunferência (fita métrica): {circumference:.1f} cm")
                print(f"- Área da seção (π.r²): {np.pi * (dist_cm/2)**2:.1f} cm²")
                
                print("\nVerificação:")
                print("Se você medir com fita métrica ao redor do tronco,")
                print(f"deve obter aproximadamente {circumference:.1f} cm")
                
                mid_point = ((points[0][0] + points[1][0])//2, 
                           (points[0][1] + points[1][1])//2)
                cv2.putText(img_copy, f"Diametro: {dist_cm:.1f} cm", 
                          (mid_point[0]-100, mid_point[1]-20),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                output_filename = "medidas_" + IMAGE_FILE
                cv2.imwrite(output_filename, img_copy)
                print(f"\nArquivo de saída:")
                print(f"- Nome: {output_filename}")
                print("=== FIM DO PROCESSAMENTO ===")
        
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
