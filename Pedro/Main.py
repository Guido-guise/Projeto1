import cv2
import numpy as np
from skimage.morphology import skeletonize
import pandas as pd

from Funcs import (melhorar_altura, maior_blob, extrair_planta_vaso, detecta_topo_vaso, so_a_planta, achar_base_topo_caule, tracar_caule, desenhar_caule, mede_diametro_coleto, desenha_coleto, calcula_altura_vertical)

#RAIO_CAULE = 18 
ALTURA_PADRAO = 2000
resultados = [] # vai virar csv
erros_altura = []
erros_comp = []
erros_diam = []
# =========================================================

GABARITO = {
    1: dict(altura=772,  comp=697,  diam=12),
    2: dict(altura=1179, comp=961,  diam=19),
    3: dict(altura=1107, comp=1340, diam=21),
    4: dict(altura=794,  comp=630,  diam=14),
    5: dict(altura=269,  comp=75,   diam=16),
    6: dict(altura=394,  comp=263,  diam=13),
    7: dict(altura=1102, comp=941,  diam=16),
    8: dict(altura=997,  comp=948,  diam=13),
    9: dict(altura=1333, comp=1252, diam=16),
    10: dict(altura=873, comp=610,  diam=13),
    11: dict(altura=1039, comp=933, diam=14),
    12: dict(altura=1547, comp=1365, diam=17),
    13: dict(altura=273, comp=73,   diam=13),
}

for k in range(1,11):
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
    
    # altura básica:(agora forçando a encontrar folha real mais alta)
    altura_basica = calcula_altura_vertical(mask_planta_rude, topo_vaso)
    # Converter para a imagem reescalada originalmente:
    dy_alvo = 10 * F_escala
    ###########################
    diametro_pixels, pontos_do_coleto = mede_diametro_coleto(mask_planta_rude, caule, topo_vaso, dy_pedido=dy_alvo)
    ###################
    # conversões:
    altura_orig  = int(round(altura_basica / F_escala))
    comp_orig    = round(L_px / F_escala)
    diametro_orig = int(round(diametro_pixels / F_escala))

    # comparação com gabarito
    ref = GABARITO[k]
    erro_A = abs(altura_orig    - ref['altura']) / ref['altura'] * 100
    erro_C = abs(comp_orig      - ref['comp'])   / ref['comp']   * 100
    erro_D = abs(diametro_orig  - ref['diam'])   / ref['diam']   * 100

    erros_altura.append(erro_A)
    erros_comp.append(erro_C)
    erros_diam.append(erro_D)

    resultados.append({'Img': k,'Altura Vert.': altura_orig,'Compr Total': comp_orig,'Diâmetro': diametro_orig, 'Erro A %': round(erro_A, 2),'Erro C %': round(erro_C, 2),'Erro D %': round(erro_D, 2),'Área': '','Nro Folhas': '',})

    # # Prints no terminal dos resultados encontrados:
    # print(f"Altura básica da planta_{k}: {altura_basica} px")
    #print(f"Comprimento do caule_{k} {L_px:.1f} px")
    # print(f"Diâmetro do coleto_{k}: {diametro_pixels} px")
    

    # visualização:
    resultado = desenhar_caule(img, skel, caule, topo_vaso, base=base, topo=topo)
    # Força a "sobreposição das imagens":
    resultado = desenha_coleto(resultado, pontos_do_coleto)

    # # redimensiona pra caber na tela:
    # altura_tela = 800
    # # altura_resultado = 1200
    # escala_vis = altura_tela / resultado.shape[0]
    # nova_L = int(resultado.shape[1] * escala_vis)
    # # escala_resultado = altura_resultado / resultado.shape[0]
    # # resultado_vis = cv2.resize(resultado, (int(resultado.shape[1] * escala_resultado), altura_tela))
    # mask_vis = cv2.resize(mask_planta, (int(mask_planta.shape[1] * escala_vis), altura_tela))
    # #mask_caule_vis  = cv2.resize(mask_caule,  (nova_L, altura_tela))

    # # namedWindows força o tamanho correto do imshow:
    # cv2.namedWindow("Mascara planta", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Mascara planta", mask_vis.shape[1], mask_vis.shape[0])
    # cv2.imshow("Mascara planta", mask_vis)

    # # cv2.namedWindow("resultado", cv2.WINDOW_NORMAL)
    # # cv2.resizeWindow("resultado", resultado_vis.shape[1], resultado_vis.shape[0])
    # cv2.imshow("resultado", resultado)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Pandas solicitado:
df = pd.DataFrame(resultados, columns=['Img', 'Altura Vert.', 'Compr Total', 'Diâmetro', 'Área', 'Nro Folhas'])
 
print("\n=== Tabela final ===")

print(df.to_string(index=False))

# --- MAPE ---
mape_alt  = float(np.mean(erros_altura))
mape_comp = float(np.mean(erros_comp))
mape_diam = float(np.mean(erros_diam))

print("\n=== MAPE ===")
print(f"Altura básica : {mape_alt:6.2f}%   (limite < 2%)        {'OK' if mape_alt  < 2  else 'FALHA'}")
print(f"Compr total   : {mape_comp:6.2f}%   (básico < 2%, adv < 5%)  adv={'OK' if mape_comp < 5 else 'FALHA'}")
print(f"Diâmetro      : {mape_diam:6.2f}%   (limite < 20%)       {'OK' if mape_diam < 20 else 'FALHA'}")