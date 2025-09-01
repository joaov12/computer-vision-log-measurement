import cv2

class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        # convert image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # create a mask with adaptive threshold
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

        # Mostrar a máscara para debug
        cv2.imshow("Mask - Objetos detectados em branco", mask)

        # find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Reduzindo a área mínima para detectar objetos menores
            if area > 1000:  # Era 2000, reduzi para 1000
                objects_contours.append(cnt)

        # Desenhar todos os contornos encontrados na máscara para debug
        mask_debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(mask_debug, contours, -1, (0,255,0), 2)
        cv2.imshow("Contornos detectados", mask_debug)

        print(f"Número de objetos detectados: {len(objects_contours)}")
        return objects_contours

    # def get_objects_rect(self):
    #     box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    #     box = np.int0(box)