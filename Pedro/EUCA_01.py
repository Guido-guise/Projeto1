import cv2
import numpy as np
# import os

# path = r"C:\Users\pedro\Documents\INSPER\SEM_07\VISAO\Projeto1\_Eucalipto_Escolhidos1\Eucalipto1.jpg"
# print("Existe?", os.path.exists(path))

#Isolar fundo:
lower_azul = np.array([100, 120, 150])
upper_azul = np.array([120, 255, 255])

# altura y = 2005 - 2035 --> 2012 altura y


img0 = cv2.imread(r"C:\Users\pedro\Documents\INSPER\SEM_07\VISAO\Projeto1\_Eucalipto_Escolhidos1\Eucalipto1.jpg")
# if img0 is None:
#     print("Image not loaded. Check path.")
y_chao = 2950
# cortar a imagem para agilizar o processamento.
img0 = img0[:y_chao, :]

img_HSV = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)

(h,w,c) = img_HSV.shape

imgn = cv2.resize(img_HSV, (w//2, h//2))
# máscara de cada cor:
mask_fundo = cv2.inRange(imgn, lower_azul, upper_azul)
mask_planta = cv2.bitwise_not(mask_fundo)

# Aplicar máscara na imagem original (BGR)
imgn_bgr = cv2.resize(img0, (w//2, h//2))
resultado = cv2.bitwise_and(imgn_bgr, imgn_bgr, mask=mask_planta)


cv2.imshow("teste", mask_fundo)
cv2.imshow("Planta isolada", resultado)
#cv2.
cv2.waitKey(0)
cv2.destroyAllWindows()
