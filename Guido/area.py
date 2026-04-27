import cv2
import numpy as np
import matplotlib.pyplot as plt


CAMINHO_IMAGEM = "Projeto1/_Eucalipto_Escolhidos2/Eucalipto{}.jpg"
REFERENCIAS = [62484, 217423, 179931, 43952, 28524]

# Parametros calibrados previamente.
CORTE_ALTURA = 0.64
HSV_MIN = np.array([18, 38, 38], dtype=np.uint8)
HSV_MAX = np.array([96, 255, 255], dtype=np.uint8)
AREA_MINIMA = 15


def limpar_componentes_pequenos(mask, area_minima):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    mask_limpa = np.zeros_like(mask)

    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= area_minima:
            mask_limpa[labels == label] = 255

    return mask_limpa


erros = []

for i, referencia in enumerate(REFERENCIAS, start=1):
    caminho = CAMINHO_IMAGEM.format(i)
    img = cv2.imread(caminho, cv2.IMREAD_COLOR)

    if img is None:
        print(f"Eucalipto{i}: imagem nao encontrada em {caminho}")
        continue

    # Corte fixo remove vaso/suporte sem mexer nas folhas calibradas.
    altura = img.shape[0]
    img_recortada = img[: int(altura * CORTE_ALTURA), :]

    hsv = cv2.cvtColor(img_recortada, cv2.COLOR_BGR2HSV)
    mask_folhas = cv2.inRange(hsv, HSV_MIN, HSV_MAX)

    # Limpeza leve: remove apenas ruidos muito pequenos.
    mask_folhas = limpar_componentes_pequenos(mask_folhas, AREA_MINIMA)

    area = int(np.count_nonzero(mask_folhas))
    erro = abs((area - referencia) / referencia) * 100.0
    erros.append(erro)

    print(f"Eucalipto{i}:")
    print(f"Area: {area} px")
    print(f"Referencia: {referencia} px")
    print(f"Erro: {erro:.2f} %\n")

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_recortada, cv2.COLOR_BGR2RGB))
    plt.title("Imagem recortada")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_folhas, cmap="gray")
    plt.title("Mascara das folhas")
    plt.axis("off")

    plt.suptitle(f"Eucalipto{i}")
    plt.tight_layout()
    plt.show()

if erros:
    print(f"Erro medio final: {np.mean(erros):.2f} %")
