import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

marker_size = 600  
marker_image = cv2.aruco.generateImageMarker(aruco_dict, 0, marker_size)

cv2.imwrite("aruco_marker.jpg", marker_image)
