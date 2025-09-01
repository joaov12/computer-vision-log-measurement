import cv2

def list_cameras():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Câmera {i} está disponível")
                cv2.imshow(f"Camera {i}", frame)
                cv2.waitKey(2000)  
                cv2.destroyWindow(f"Camera {i}")
            cap.release()
        else:
            print(f"Câmera {i} não está disponível")

print("Procurando câmeras disponíveis...")
list_cameras()
cv2.destroyAllWindows()
