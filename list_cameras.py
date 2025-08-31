import cv2

def list_cameras():
    # Testa as primeiras 5 portas de câmera
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Tenta ler um frame
            ret, frame = cap.read()
            if ret:
                print(f"Câmera {i} está disponível")
                # Mostra o frame para identificar qual câmera é
                cv2.imshow(f"Camera {i}", frame)
                cv2.waitKey(2000)  # Espera 2 segundos
                cv2.destroyWindow(f"Camera {i}")
            cap.release()
        else:
            print(f"Câmera {i} não está disponível")

print("Procurando câmeras disponíveis...")
list_cameras()
cv2.destroyAllWindows()
