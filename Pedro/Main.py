import cv2
import numpy as np
from skimage.morphology import skeletonize
import pandas as pd

from Funcs import (melhorar_altura, maior_blob, extrair_planta_vaso, detecta_topo_vaso, so_a_planta, achar_base_topo_caule, tracar_caule, desenhar_caule, mede_diametro_coleto, desenha_coleto, calcula_altura_vertical, extrair_caule_mask)


#RAIO_CAULE = 18 
LARGURA_MAX_CAULE = 25 # tentar forçar um limite.
ALTURA_PADRAO = 2000
resultados = [] # vai virar csv
mape_alturas = []
mape_comprimentos = []
mape_diametros = []
# =========================================================

GABARITO = {
    1: dict(altura=772,  comp=697,  diam=12),
    2: dict(altura=1179, comp=961,  diam=19),
    3: dict(altura=1107, comp=1340, diam=21),
    4: dict(altura=794,  comp=630,  diam=14),
    5: dict(altura=269,  comp=75,   diam=16),
}

for k in range(1,6):
    # config inicial:
    path = fr"C:\Users\pedro\Documents\INSPER\SEM_07\VISAO\Projeto1\_Eucalipto_Escolhidos2\Eucalipto{k}.jpg"

    img0 = cv2.imread(path)
    # pega altura original:
    H_original = img0.shape[0]
    # carrega + ajusta tamanho da imagem
    img = melhorar_altura(img0, altura_do_chao = 2000)
    # Fator de correção da escala:
    F_escala = ALTURA_PADRAO / H_original
    # acha planata + vaso
    estrutura, hsv = extrair_planta_vaso(img)
    # detecta fim do vaso
    vaso, topo_vaso, cx = detecta_topo_vaso(estrutura, hsv)
    # Segmenta a planta
    # Agora, duas abordagens uma máscara suave e outra rude:
    mask_planta = so_a_planta(img, topo_vaso, suavizar=True)
    mask_planta_rude = so_a_planta(img, topo_vaso, suavizar=False)
    # SKELETON
    skel = skeletonize(mask_planta > 0).astype(np.uint8) 
    # DEBUG: salva skeleton puro

    # acha base e o topo0
    base, topo = achar_base_topo_caule(skel,mask_planta, topo_vaso, cx)
    # traça o caule
    caule, L_px = tracar_caule(skel, base, topo, topo_vaso=topo_vaso, mask_planta=mask_planta)
    # tira as folhas:
    mask_caule = extrair_caule_mask(mask_planta, caule, largura_max=LARGURA_MAX_CAULE)
    # altura básica:(agora forçando a encontrar folha real mais alta)
    altura_basica = calcula_altura_vertical(mask_planta_rude, topo_vaso)
    # Converter para a imagem reescalada originalmente:
    dy_alvo = 10 * F_escala
    ####################################
    # # Mede o diâmetro do coleto:
    # diametro_pixels, pontos_do_coleto = mede_diametro_coleto(mask_caule, caule, topo_vaso, dy_pedido=dy_alvo)
    ###################################33333
    kernel_dilate_diam = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_caule_diam = cv2.dilate(mask_planta_rude, kernel_dilate_diam)
    diametro_pixels, pontos_do_coleto = mede_diametro_coleto(mask_caule_diam, caule, topo_vaso, dy_pedido=dy_alvo)
    ###################
    # conversões:
    altura_orig  = int(round(altura_basica / F_escala))
    comp_orig    = round(L_px / F_escala)
    diametro_orig = int(round(diametro_pixels / F_escala))

    # comparação com gabarito
    ref = GABARITO[k]
    erro_A = np.abs(altura_orig    - ref['altura']) / ref['altura'] * 100
    erro_C = np.abs(comp_orig      - ref['comp'])   / ref['comp']   * 100
    erro_D = np.abs(diametro_orig  - ref['diam'])   / ref['diam']   * 100
    print(f"MAPE'S: {erro_A}, {erro_C}, {erro_D}")

    mape_alturas.append(erro_A)
    mape_comprimentos.append(erro_C)
    mape_diametros.append(erro_D)

    resultados.append({'Img': k,'Altura Vert.': altura_orig,'Compr Total': comp_orig,'Diâmetro': diametro_orig,'Área': '','Nro Folhas': '',})

    # # Prints no terminal dos resultados encontrados:
    # print(f"Altura básica da planta_{k}: {altura_basica} px")
    #print(f"Comprimento do caule_{k} {L_px:.1f} px")
    # print(f"Diâmetro do coleto_{k}: {diametro_pixels} px")
    

    # visualização:
    resultado = desenhar_caule(img, skel, caule, topo_vaso, base=base, topo=topo)
    # Força a "sobreposição das imagens":
    resultado = desenha_coleto(resultado, pontos_do_coleto)

    # redimensiona pra caber na tela:
    altura_tela = 800
    # altura_resultado = 1200
    escala_vis = altura_tela / resultado.shape[0]
    nova_L = int(resultado.shape[1] * escala_vis)
    # escala_resultado = altura_resultado / resultado.shape[0]
    # resultado_vis = cv2.resize(resultado, (int(resultado.shape[1] * escala_resultado), altura_tela))
    mask_vis = cv2.resize(mask_planta, (int(mask_planta.shape[1] * escala_vis), altura_tela))
    mask_caule_vis  = cv2.resize(mask_caule,  (nova_L, altura_tela))

    # namedWindows força o tamanho correto do imshow:
    cv2.namedWindow("Mascara planta", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mascara planta", mask_vis.shape[1], mask_vis.shape[0])
    cv2.imshow("Mascara planta", mask_vis)

    # Janela 2: máscara só do caule (depois do opening) — DEBUG principal
    cv2.namedWindow("2 - Mascara so caule", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("2 - Mascara so caule", nova_L, altura_tela)
    cv2.imshow("2 - Mascara so caule", mask_caule_vis)

    # cv2.namedWindow("resultado", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("resultado", resultado_vis.shape[1], resultado_vis.shape[0])
    cv2.imshow("resultado", resultado)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

MAPE_altura = np.mean(mape_alturas)
MAPE_comprimento = np.mean(mape_comprimentos)
MAPE_diametro = np.mean(mape_diametros)

# Pandas solicitado:
df = pd.DataFrame(resultados, columns=['Img', 'Altura Vert.', 'Compr Total', 'Diâmetro', 'Área', 'Nro Folhas'])
 
print("\n=== Tabela final ===")

print(df.to_string(index=False))

print("\n=== MAPE FINAL ===")
print(f"Altura: {MAPE_altura:.2f}%")
print(f"Comprimento: {MAPE_comprimento:.2f}%")
print(f"Diâmetro: {MAPE_diametro:.2f}%")