import cv2
import numpy as np
from skimage.morphology import skeletonize

from Funcs_guido import (
    melhorar_altura,
    extrair_planta_vaso,
    detecta_topo_vaso,
    so_a_planta,
    lower_azul,
    upper_azul,
)

# Parametros do pipeline
ALTURA_DO_CHAO = 2000
AREA_MIN_ABS = 200
AREA_MIN_REL = 0.15
USAR_REGULACAO_CLUSTER = True
DIST_MAX_CLUSTER = 260

# Parametros de visualizacao (logica do arquivo "corte")
FRACAO_CORTE_VIS = 0.69
MARGEM_CORTE_VIS = 50
LARGURA_VIS = 600

KERNEL_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
KERNEL_VIZINHOS = np.array(
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]],
    dtype=np.uint8,
)
KERNEL_ENDPOINTS_VIS = np.ones((5, 5), np.uint8)


def selecionar_componentes_folha(mask_candidata):
    """
    Seleciona componentes conectados relevantes por area.
    Opcionalmente aplica regulacao por cluster espacial.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_candidata, connectivity=8)
    mask_relevante = np.zeros_like(mask_candidata)

    if num_labels <= 1:
        return mask_relevante

    maior_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    area_minima = max(AREA_MIN_ABS, int(AREA_MIN_REL * maior_area))

    candidatos = []
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= area_minima:
            candidatos.append(label)

    if not candidatos:
        return mask_relevante

    finais = candidatos
    if USAR_REGULACAO_CLUSTER and len(candidatos) > 1:
        # Mantem apenas o cluster conectado ao maior componente (ancora).
        ancora = max(candidatos, key=lambda lb: int(stats[lb, cv2.CC_STAT_AREA]))
        cluster = {ancora}

        mudou = True
        while mudou:
            mudou = False
            for lb in candidatos:
                if lb in cluster:
                    continue
                cx_lb, cy_lb = centroids[lb]
                for lc in list(cluster):
                    cx_lc, cy_lc = centroids[lc]
                    if np.hypot(cx_lb - cx_lc, cy_lb - cy_lc) <= DIST_MAX_CLUSTER:
                        cluster.add(lb)
                        mudou = True
                        break

        finais = list(cluster)

    for label in finais:
        mask_relevante[labels == label] = 255

    return mask_relevante


def recorte_visualizacao_corte(img, resultado, mask_planta):
    altura, largura = img.shape[:2]
    ini = int(altura * FRACAO_CORTE_VIS)

    rec_img = img[0:ini, 0:largura]
    rec_resultado = resultado[0:ini, 0:largura]
    rec_mask_pipeline = mask_planta[0:ini, 0:largura]

    rec_hsv = cv2.cvtColor(rec_img, cv2.COLOR_BGR2HSV)
    mask_fundo_vis = cv2.inRange(rec_hsv, lower_azul, upper_azul)
    mask_planta_vis = cv2.bitwise_not(mask_fundo_vis)

    rec_resultado_sem_fundo = cv2.bitwise_and(rec_resultado, rec_resultado, mask=mask_planta_vis)

    x, y, w, h = cv2.boundingRect(mask_planta_vis)
    if w == 0 or h == 0:
        x0, y0 = 0, 0
        x1, y1 = rec_resultado_sem_fundo.shape[1], rec_resultado_sem_fundo.shape[0]
    else:
        x0 = max(0, x - MARGEM_CORTE_VIS)
        y0 = max(0, y - MARGEM_CORTE_VIS)
        x1 = min(rec_resultado_sem_fundo.shape[1], x + w + MARGEM_CORTE_VIS)
        y1 = min(rec_resultado_sem_fundo.shape[0], y + h + MARGEM_CORTE_VIS)

    recorte_mask = rec_mask_pipeline[y0:y1, x0:x1]
    recorte_resultado = rec_resultado_sem_fundo[y0:y1, x0:x1]

    # Fallback para evitar erro de resize em recortes vazios.
    if recorte_resultado.size == 0 or recorte_resultado.shape[1] == 0:
        recorte_resultado = rec_resultado_sem_fundo
        recorte_mask = rec_mask_pipeline

    alt = max(1, int(LARGURA_VIS * (recorte_resultado.shape[0] / recorte_resultado.shape[1])))
    mask_vis = cv2.resize(recorte_mask, (LARGURA_VIS, alt))
    resultado_vis = cv2.resize(recorte_resultado, (LARGURA_VIS, alt))

    return mask_vis, resultado_vis


for k in range(1, 4):
    img0 = cv2.imread(f"Projeto1\_Pinheiro_Escolhidos1\Pinheiro{k}.jpg", cv2.IMREAD_COLOR)
    if img0 is None:
        print(f"Imagem {k} nao encontrada")
        continue

    # 1) Pipeline original de segmentacao
    img = melhorar_altura(img0, altura_do_chao=ALTURA_DO_CHAO)
    estrutura, hsv = extrair_planta_vaso(img)
    _, topo_vaso, _ = detecta_topo_vaso(estrutura, hsv)
    if topo_vaso is None:
        print(f"Imagem {k}: topo do vaso nao detectado")
        continue

    mask_planta = so_a_planta(img, topo_vaso)

    # 2) Recupera componentes foliares relevantes (melhora em plantas pequenas)
    img_capada = img.copy()
    img_capada[topo_vaso:, :] = 0
    hsv_capada = cv2.cvtColor(img_capada, cv2.COLOR_BGR2HSV)
    mask_candidata = cv2.bitwise_not(cv2.inRange(hsv_capada, lower_azul, upper_azul))
    mask_candidata[hsv_capada[:, :, 2] < 10] = 0
    mask_candidata = cv2.morphologyEx(mask_candidata, cv2.MORPH_OPEN, KERNEL_OPEN)
    mask_candidata = cv2.morphologyEx(mask_candidata, cv2.MORPH_CLOSE, KERNEL_CLOSE)

    mask_relevante = selecionar_componentes_folha(mask_candidata)
    mask_planta = cv2.bitwise_or(mask_planta, mask_relevante)

    # 3) Skeleton + endpoints = estimativa de folhas
    skel = skeletonize(mask_planta > 0).astype(np.uint8)
    vizinhos = cv2.filter2D(skel, -1, KERNEL_VIZINHOS)
    endpoints = np.logical_and(skel == 1, vizinhos == 1)
    n_folhas_estimado = int(np.count_nonzero(endpoints))
    print(f"Numero estimado de folhas_{k}: {n_folhas_estimado}")

    # 4) Overlay de visualizacao
    resultado = img.copy()
    resultado[skel > 0] = (0, 255, 0)
    endpoints_vis = cv2.dilate(endpoints.astype(np.uint8), KERNEL_ENDPOINTS_VIS, iterations=1)
    resultado[endpoints_vis > 0] = (0, 0, 255)

    # 5) Aplica recorte estilo "corte" apenas para exibir
    mask_vis, resultado_vis = recorte_visualizacao_corte(img, resultado, mask_planta)

    cv2.namedWindow("Mascara planta", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mascara planta", mask_vis.shape[1], mask_vis.shape[0])
    cv2.imshow("Mascara planta", mask_vis)

    cv2.namedWindow("resultado", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("resultado", resultado_vis.shape[1], resultado_vis.shape[0])
    cv2.imshow("resultado", resultado_vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
