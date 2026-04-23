import cv2
import numpy as np
import matplotlib.pyplot as plt

# Caminho fixo de UMA imagem. O script foi feito para rodar uma imagem por vez.
img_1 = cv2.imread("Projeto1/_Eucalipto_Escolhidos2/Eucalipto5.jpg", cv2.IMREAD_COLOR)

# 2) Aplicar a logica do arquivo "corte":
# corta a parte inferior da imagem usando 69% da altura para tirar boa parte da base.
altura, largura, _ = img_1.shape
ini = int(altura * 0.69)
recorte_superior = img_1[0:ini, 0:largura]

# 3) Converter o recorte para HSV e detectar fundo azul.
# O objetivo e separar rapidamente planta (nao-azul) do fundo.
recorte_hsv = cv2.cvtColor(recorte_superior, cv2.COLOR_BGR2HSV)
lower_azul = np.array([100, 120, 150], dtype=np.uint8)
upper_azul = np.array([120, 255, 255], dtype=np.uint8)
mascara_fundo = cv2.inRange(recorte_hsv, lower_azul, upper_azul)
mascara_planta = cv2.bitwise_not(mascara_fundo)

# 4) Aplicar a mascara da planta e recortar somente a regiao util com bounding box.
# Isso centraliza a planta e remove areas vazias ao redor.
planta_sem_fundo = cv2.bitwise_and(recorte_superior, recorte_superior, mask=mascara_planta)
x, y, w, h = cv2.boundingRect(mascara_planta)
margem = 50
x0 = max(0, x - margem)
y0 = max(0, y - margem)
x1 = min(planta_sem_fundo.shape[1], x + w + margem)
y1 = min(planta_sem_fundo.shape[0], y + h + margem)
imagem_cortada_bgr = planta_sem_fundo[y0:y1, x0:x1]

# 5) Segmentar folhas na imagem ja cortada usando HSV simples.
# A mascara binaria e a base do calculo da area foliar por pixels.
imagem_cortada_hsv = cv2.cvtColor(imagem_cortada_bgr, cv2.COLOR_BGR2HSV)
mascara_folhas = cv2.inRange(imagem_cortada_hsv, (25, 40, 30), (95, 255, 255))

# 6) Pequena limpeza da mascara para reduzir ruido e remover mais caule.
# OPEN com kernel horizontal (3x11) enfraquece estruturas finas verticais.
# CLOSE leve (elipse 3x3) recompõe pequenos buracos nas folhas.
kernel_open = np.ones((3, 13), np.uint8)
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mascara_folhas = cv2.morphologyEx(mascara_folhas, cv2.MORPH_OPEN, kernel_open)
mascara_folhas = cv2.morphologyEx(mascara_folhas, cv2.MORPH_CLOSE, kernel_close)

# 7) Calcular area foliar por contagem de pixels brancos da mascara.
area_foliar_pixels = np.count_nonzero(mascara_folhas)

# 8) Exibir area foliar calculada.
print(f"Area foliar (pixels): {area_foliar_pixels}")

# 9) Mostrar apenas as duas imagens solicitadas:
# imagem ja cortada e mascara das folhas.
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(imagem_cortada_bgr, cv2.COLOR_BGR2RGB))
plt.title("Imagem Ja Cortada")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mascara_folhas, cmap="gray")
plt.title("Mascara das Folhas")
plt.axis("off")

plt.tight_layout()
plt.show()
