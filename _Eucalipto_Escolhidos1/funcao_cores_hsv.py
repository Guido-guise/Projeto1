import cv2

def mostra_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"HSV em ({x},{y}): {param[y,x]}")

img = cv2.imread("Fig2_Curling1.bmp")
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("Clique na imagem", img)
cv2.setMouseCallback("Clique na imagem", mostra_hsv, img_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
