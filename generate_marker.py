import cv2

# Criar o dicionário ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

# Criar o marcador (ID 0)
marker_size = 600  # tamanho em pixels
marker_image = cv2.aruco.generateImageMarker(aruco_dict, 0, marker_size)

# Salvar o marcador
cv2.imwrite("aruco_marker.jpg", marker_image)
