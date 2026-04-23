# Bibliotecas importadas:
import cv2
import numpy as np
from skimage.graph import route_through_array

#Thresholds HSV fixos:
# (Fundo azul)
lower_azul = np.array([100, 120, 150])
upper_azul = np.array([120, 255, 255])

# Funções para utilizar no projeto:

def melhorar_altura(img, altura_do_chao):

    (h, w, c) = img.shape

    if h == altura_do_chao:
        return img

    escala =  altura_do_chao / h
    # https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
    
    img_redimensionada = cv2.resize(img, (int(w * escala), altura_do_chao), interpolation=cv2.INTER_AREA)

    return img_redimensionada

def maior_blob(mask):

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask
    # [0] É O FUNDO:
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    mcomp = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    return mcomp

def extrair_planta_vaso(img):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_fundo = cv2.inRange(img_hsv, lower_azul, upper_azul)
    mask_planta_vaso = cv2.bitwise_not(mask_fundo)

    # remove ruído pequeno:
    # só vizinhos horizontais e virtuais
    kernel = np.array(([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), dtype=np.uint8)
    #erode = cv2.erode(mask_planta_vaso, kernel)
    #dilate = cv2.dilate(eroded, kernel)
    mask_planta_vaso = cv2.morphologyEx(mask_planta_vaso,cv2.MORPH_OPEN,kernel)

    return maior_blob(mask_planta_vaso), img_hsv

def detecta_topo_vaso(vaso_tupete, img_hsv, saturacao_max = 40):
    
    saturacao = img_hsv[:, :, 1]
    vaso  = ((vaso_tupete > 0) & (saturacao < saturacao_max)).astype(np.uint8) * 255

    # MORPH_CLOSE preencher buracos internos:
    kernel_morph_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    vaso = cv2.morphologyEx(vaso, cv2.MORPH_CLOSE, kernel_morph_close)
    # MORPH_OPEN remove manhcas peuqenas:
    kernel_morph_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    vaso = cv2.morphologyEx(vaso, cv2.MORPH_OPEN, kernel_morph_open)

    vaso = maior_blob(vaso)
    if vaso.sum() == 0:
        return None, None, None
    
    # topo do vaso:
    # não preciso de x, então: _
    ys, _ = np.where(vaso > 0)

    y_topo = int(ys.min()) 

    # "centroide" do vaso:
    altura_faixa = 20

    # Recortar a faixa superior do vaso
    faixa_topo = vaso[y_topo:y_topo + altura_faixa]

    # Encontrar colunas onde existe pelo menos um pixel do vaso
    colunas_com_vaso = np.any(faixa_topo > 0, axis=0)

    # Pegar os índices dessas colunas
    indices_colunas = np.where(colunas_com_vaso)[0]

    # Calcular o centro horizontal
    x_esquerda = indices_colunas.min()
    x_direita = indices_colunas.max()
    x_centro = int((x_esquerda + x_direita) / 2)

    return vaso, y_topo, x_centro

def so_a_planta(img, topo_vaso):

    img_capada = img.copy()
    img_capada[topo_vaso:, :] = 0

    hsv = cv2.cvtColor(img_capada, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_not(cv2.inRange(hsv, lower_azul, upper_azul))
    # tira pixel preto
    mask[hsv[:, :, 2] < 10] = 0

    kernel_morph_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_morph_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_morph_open)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_morph_close)

    return maior_blob(mask_close)

def achar_base_topo_caule(skeleton, topo_vaso, cx, img_hsv):

    # Encontrar o ponto mais próximo do topo do vaso
    ys, xs = np.where(skeleton > 0)
    
    # Forçar a dar menos enfase no direcionamento em x
    distancia = np.sqrt(0.20 * (xs - cx) ** 2 + (ys - topo_vaso) ** 2)
    # base:
    indice_base = int(np.argmin(distancia))
    base = (int(ys[indice_base]), int(xs[indice_base]))

    # copa(topo):
    kernel = np.array(([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), dtype=np.uint8)
    numero_vizinhos = cv2.filter2D(skeleton, -1, kernel)
    fins = (skeleton > 0) & (numero_vizinhos == 1)
    ys_f, xs_f = np.where(fins)
    ##
    # indice_topo = int(np.argmin(ys))
    # topo = (int(ys[indice_topo]), int(xs[indice_topo]))

    # 2) mascara de pixels avermelhados
    H = img_hsv[:,:,0]
    S = img_hsv[:,:,1]
    mask_vermelho = (((H < 15) | (H > 165)) & (S > 40)).astype(np.uint8)

    # conta quantos pixels vermelhos tem em um quadrado 31x31 ao redor de cada pixel
    kernel_contagem = np.ones((31, 31), np.uint8)
    contagem_vermelho = cv2.filter2D(mask_vermelho, cv2.CV_32S, kernel_contagem)

    # 3) escolhe topo: endpoint na metade superior com vermelho ao redor,
    #    ou fallback para o endpoint mais alto
    meio_y = (ys.min() + ys.max()) / 2
    e_alto = ys_f < meio_y
    e_vermelho = contagem_vermelho[ys_f, xs_f] > 10
    candidatos = e_alto & e_vermelho

    if candidatos.any():
        ys_cand = np.where(candidatos, ys_f, np.inf)
        indice_topo = int(np.argmin(ys_cand))
        topo = (int(ys_f[indice_topo]), int(xs_f[indice_topo]))
        return base, topo

    # 4) FALLBACK: bifurcacao mais alta do skeleton (planta jovem sem broto)
    # bifurcacao = pixel com 3 ou mais vizinhos
    bifurcacoes = (skeleton > 0) & (numero_vizinhos >= 3)
    ys_b, xs_b = np.where(bifurcacoes)

    if len(ys_b) > 0:
        indice_topo = int(np.argmin(ys_b))
        topo = (int(ys_b[indice_topo]), int(xs_b[indice_topo]))
    else:
        # ultimo recurso: endpoint mais alto do skeleton
        indice_topo = int(np.argmin(ys_f))
        topo = (int(ys_f[indice_topo]), int(xs_f[indice_topo]))

    return base, topo

def tracar_caule(skeleton, base, topo):
    # Usa skimage.graph.route_through_array, que acha automaticamente o
    # caminho de menor custo entre dois pontos em uma matriz.

    # A matriz de custo tem valor 1 onde existe skeleton e 1000 onde não existe.
    # Assim o algoritmo vai preferir sempre andar pelo skeleton, porque sair
    # dele custa muito '"caro"

    custo = np.where(skeleton > 0, 1.0, 1000.0).astype(np.float32)

    # para funcionar o skeleton na diagonal o precisa do  fully connected+True 
    caminho, _ = route_through_array(custo, start = base, end = topo, fully_connected=True)

    # Comprimento do caule:
    comprimento = 0.0
    for i in range(1, len(caminho)):
        dy = caminho[i][0] - caminho[i-1][0]
        dx = caminho[i][1] - caminho[i-1][1]
        comprimento += np.hypot(dy, dx) # hipotenusa

    return caminho, comprimento

def desenhar_caule(img, skeleton, caule, topo_vaso):

    img_caule = img.copy()
    kernel = np.ones((3,3),np.uint8)
    skeleton_grosso = cv2.dilate(skeleton, kernel)
    img_caule[skeleton_grosso > 0] = (200, 200, 0) # skeleton ciano

    for (y, x) in caule:
        cv2.circle(img_caule, (x, y), radius=1, color=(0, 0, 255), thickness=-1) # caule vermelho

    cv2.line(img_caule, (0, topo_vaso), (img_caule.shape[1], topo_vaso), color=(255, 200, 100), thickness=2) # linha alaranjada
    
        
    return img_caule