import cv2
import numpy as np
import matplotlib.pyplot as plt

for i in range(1, 4):
    img_1 = cv2.imread(f"Projeto1/_Pinheiro_Escolhidos1/Pinheiro{i}.jpg", cv2.IMREAD_COLOR)

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
    mascara_antes_morf = cv2.inRange(imagem_cortada_hsv, (25, 40, 30), (95, 255, 255))

    # 6) Morfologia leve com kernels pequenos (3x3), para preservar folhas finas.
    # CLOSE 3x3 conecta pequenas quebras das agulhas.
    # OPEN 3x3 remove ruido pontual sem "apagar" muita estrutura fina.
    kernel_close_leve = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_open_leve = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara_pos_morf = cv2.morphologyEx(mascara_antes_morf, cv2.MORPH_CLOSE, kernel_close_leve)
    mascara_pos_morf = cv2.morphologyEx(mascara_pos_morf, cv2.MORPH_OPEN, kernel_open_leve)

    # 7) Remocao de caule por criterio estrutural (nao apenas espessura):
    # - busca na banda central da planta (onde o caule fica)
    # - componente alongado verticalmente (alta razao de aspecto)
    # - componente com grande extensao vertical
    altura_mask, largura_mask = mascara_pos_morf.shape
    mask_banda_central = np.zeros_like(mascara_pos_morf)
    metade_banda = int(0.06 * largura_mask)  # banda central total ~12% da largura
    cx = largura_mask // 2
    mask_banda_central[:, max(0, cx - metade_banda):min(largura_mask, cx + metade_banda)] = 255

    mask_central = cv2.bitwise_and(mascara_pos_morf, mask_banda_central)

    # Opening vertical pequeno (3x5) para destacar estruturas lineares verticais.
    # Kernel pequeno evita agressividade excessiva nas folhas finas.
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    mask_caule_candidata = cv2.morphologyEx(mask_central, cv2.MORPH_OPEN, kernel_vertical)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_caule_candidata, connectivity=8)
    mask_caule = np.zeros_like(mask_caule_candidata)
    for label in range(1, num_labels):
        w_comp = int(stats[label, cv2.CC_STAT_WIDTH])
        h_comp = int(stats[label, cv2.CC_STAT_HEIGHT])
        area_comp = int(stats[label, cv2.CC_STAT_AREA])

        razao_aspecto = h_comp / float(max(1, w_comp))
        vertical_longo = h_comp >= int(0.35 * altura_mask)

        # Componente de caule: vertical, alongado e relativamente grande.
        if razao_aspecto >= 2.8 and vertical_longo and area_comp >= 120:
            mask_caule[labels == label] = 255

    # Mascara final de folhas = pos-morfologia menos o caule detectado.
    mascara_folhas = cv2.bitwise_and(mascara_pos_morf, cv2.bitwise_not(mask_caule))

    # 8) Calcular area foliar por contagem de pixels brancos da mascara.
    area_foliar_pixels = np.count_nonzero(mascara_folhas)

    # 9) Exibir area foliar calculada.
    print(f"Area foliar (pixels): {area_foliar_pixels}")

    # 10) Comparar antes/depois para calibracao visual dos kernels:
    # - antes da morfologia
    # - depois da morfologia
    # - resultado final sem caule
    plt.figure(figsize=(14, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(imagem_cortada_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Imagem Ja Cortada")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mascara_folhas, cmap="gray")
    plt.title("Mascara Final Sem Caule")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
