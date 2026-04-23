import cv2
import numpy as np
from skimage.morphology import skeletonize

from Funcs import (melhorar_altura, maior_blob, extrair_planta_vaso, detecta_topo_vaso, so_a_planta, achar_base_topo_caule, tracar_caule, desenhar_caule)

for k in range(1,6):
    # config inicial:
    path = fr"C:\Users\pedro\Documents\INSPER\SEM_07\VISAO\Projeto1\_Eucalipto_Escolhidos1\Eucalipto{k}.jpg"

    img0 = cv2.imread(path)
    # carrega + ajusta tamanho da imagem
    img = melhorar_altura(img0, altura_do_chao = 2000)
    # acha planata + vaso
    estrutura, hsv = extrair_planta_vaso(img)
    # detecta fim do vaso
    vaso, topo_vaso, cx = detecta_topo_vaso(estrutura, hsv)
    # Segmenta a planta
    mask_planta = so_a_planta(img, topo_vaso)
    # SKELETON
    skel = skeletonize(mask_planta > 0).astype(np.uint8) 
    # acha base e o topo
    base, topo = achar_base_topo_caule(skel, topo_vaso, cx, hsv)

    # desenha o caule
    caule, L_px = tracar_caule(skel, base, topo)

    print(f"Comprimento do caule_{k} em pixels {L_px:.1f} px")

    # visualização:
    resultado = desenhar_caule(img, skel, caule, topo_vaso)

    # redimensiona pra caber na tela:
    altura_tela = 800
    escala_vis = altura_tela / resultado.shape[0]
    resultado_vis = cv2.resize(resultado, (int(resultado.shape[1] * escala_vis), altura_tela))
    mask_vis = cv2.resize(mask_planta, (int(mask_planta.shape[1] * escala_vis), altura_tela))

    cv2.namedWindow("Mascara planta", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mascara planta", mask_vis.shape[1], mask_vis.shape[0])
    cv2.imshow("Mascara planta", mask_vis)

    cv2.namedWindow("resultado", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("resultado", resultado_vis.shape[1], resultado_vis.shape[0])
    cv2.imshow("resultado", resultado_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()