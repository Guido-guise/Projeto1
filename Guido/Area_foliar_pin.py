import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from collections import deque
import math

def bfs_mais_distante(seed, pixels_set):
    """
    BFS em grafo 8-conexo do skeleton.
    Retorna o pixel mais distante de 'seed', distancias e pais.
    """
    fila = deque([seed])
    dist = {seed: 0}
    pai = {seed: None}
    mais_distante = seed

    while fila:
        y, x = fila.popleft()
        d = dist[(y, x)]
        if d > dist[mais_distante]:
            mais_distante = (y, x)

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                viz = (y + dy, x + dx)
                if viz in pixels_set and viz not in dist:
                    dist[viz] = d + 1
                    pai[viz] = (y, x)
                    fila.append(viz)

    return mais_distante, dist, pai


def caminho_mais_longo_componente(pixels_comp):
    """
    Calcula um caminho longo aproximado no componente do skeleton
    usando BFS duas vezes (diametro aproximado).
    """
    pixels_set = set(pixels_comp)
    inicio = pixels_comp[0]
    a, _, _ = bfs_mais_distante(inicio, pixels_set)
    b, dist_bfs, pai = bfs_mais_distante(a, pixels_set)

    caminho = []
    atual = b
    while atual is not None:
        caminho.append(atual)
        atual = pai[atual]
    caminho.reverse()

    return caminho, dist_bfs[b]


for i in range(1, 7):
    img_1 = cv2.imread(f"Projeto1/_Pinheiro_Escolhidos2/Pinheiro{i}.jpg", cv2.IMREAD_COLOR)

    # 1) Recorte inicial no estilo do arquivo "corte".
    altura, largura, _ = img_1.shape
    ini = int(altura * 0.69)
    recorte_superior = img_1[0:ini, 0:largura]

    # 2) Segmentar fundo azul para isolar planta.
    recorte_hsv = cv2.cvtColor(recorte_superior, cv2.COLOR_BGR2HSV)
    lower_azul = np.array([100, 120, 150], dtype=np.uint8)
    upper_azul = np.array([120, 255, 255], dtype=np.uint8)
    mascara_fundo = cv2.inRange(recorte_hsv, lower_azul, upper_azul)
    mascara_planta = cv2.bitwise_not(mascara_fundo)

    # 3) Recortar regiao util da planta por bounding box.
    planta_sem_fundo = cv2.bitwise_and(recorte_superior, recorte_superior, mask=mascara_planta)
    x, y, w, h = cv2.boundingRect(mascara_planta)
    margem = 50
    x0 = max(0, x - margem)
    y0 = max(0, y - margem)
    x1 = min(planta_sem_fundo.shape[1], x + w + margem)
    y1 = min(planta_sem_fundo.shape[0], y + h + margem)
    imagem_cortada_bgr = planta_sem_fundo[y0:y1, x0:x1]

    # 4) Mascara binaria inicial das folhas (HSV simples).
    imagem_cortada_hsv = cv2.cvtColor(imagem_cortada_bgr, cv2.COLOR_BGR2HSV)
    mascara_antes_morf = cv2.inRange(imagem_cortada_hsv, (25, 40, 30), (95, 255, 255))

    # 5) Morfologia leve com kernels pequenos para preservar agulhas finas.
    kernel_close_leve = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_open_leve = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara_pos_morf = cv2.morphologyEx(mascara_antes_morf, cv2.MORPH_CLOSE, kernel_close_leve)
    mascara_pos_morf = cv2.morphologyEx(mascara_pos_morf, cv2.MORPH_OPEN, kernel_open_leve)

    # 6) Skeletonization (entrada booleana obrigatoria).
    mask_bool = mascara_pos_morf > 0
    skeleton_bool = skeletonize(mask_bool)
    skeleton_u8 = (skeleton_bool.astype(np.uint8)) * 255

    # 7) Detectar caule como caminho principal mais longo e aproximadamente vertical.
    # Para cada componente do skeleton:
    # - calcula longest path aproximado (BFS duas vezes)
    # - pontua por comprimento, verticalidade e centralidade
    altura_mask, largura_mask = skeleton_u8.shape
    centro_x_global = largura_mask / 2.0

    num_labels, labels = cv2.connectedComponents(skeleton_u8, connectivity=8)
    melhor_score = -1e9
    melhor_caminho = []

    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        if len(ys) < 30:
            continue

        pixels_comp = list(zip(ys.tolist(), xs.tolist()))
        caminho, comprimento = caminho_mais_longo_componente(pixels_comp)
        if len(caminho) < 2:
            continue

        y_ini, x_ini = caminho[0]
        y_fim, x_fim = caminho[-1]
        dy = abs(y_fim - y_ini)
        dx = abs(x_fim - x_ini)

        verticalidade = dy / float(max(1, dy + dx))
        x_medio = float(np.mean(xs))
        centralidade = 1.0 - min(1.0, abs(x_medio - centro_x_global) / max(1.0, centro_x_global))

        score = (2.0 * comprimento) + (220.0 * verticalidade) + (80.0 * centralidade)
        if score > melhor_score:
            melhor_score = score
            melhor_caminho = caminho

    # 8) Expandir levemente o eixo do caule para cobrir sua largura real.
    mask_caule_skel = np.zeros_like(mascara_pos_morf, dtype=np.uint8)
    for (y, x) in melhor_caminho:
        mask_caule_skel[y, x] = 255

    kernel_dilatacao_caule = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_caule = cv2.dilate(mask_caule_skel, kernel_dilatacao_caule, iterations=3)

    # 9) Mascara final sem caule.
    mascara_folhas = cv2.bitwise_and(mascara_pos_morf, cv2.bitwise_not(mask_caule))

    # 10) Area foliar por contagem de pixels brancos.
    area_foliar_pixels = int(np.count_nonzero(mascara_folhas))
    print(f"Pinheiro{i} - Area foliar (pixels): {area_foliar_pixels}")

    # 11) Visualizacao opcional para depuracao.
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(imagem_cortada_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Imagem Ja Cortada")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(mascara_antes_morf, cmap="gray")
    plt.title("Mascara Inicial")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(mascara_pos_morf, cmap="gray")
    plt.title("Mascara Pos-Morfologia")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(skeleton_u8, cmap="gray")
    plt.title("Skeleton")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(mask_caule, cmap="gray")
    plt.title("Caule Detectado")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(mascara_folhas, cmap="gray")
    plt.title("Mascara Final Sem Caule")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
