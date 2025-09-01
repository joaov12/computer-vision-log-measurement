import cv2

class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

        cv2.imshow("Mask - Objetos detectados em branco", mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  
                objects_contours.append(cnt)

        mask_debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(mask_debug, contours, -1, (0,255,0), 2)
        cv2.imshow("Contornos detectados", mask_debug)

        print(f"Número de objetos detectados: {len(objects_contours)}")
        return objects_contours
