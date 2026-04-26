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

def extrair_caule_mask(mask_planta, raio=5):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*raio+1, 2*raio+1))

    caule = cv2.morphologyEx(mask_planta, cv2.MORPH_OPEN, kernel)

    return maior_blob(caule)

def achar_base_topo_caule(skeleton, topo_vaso, cx):

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

    # topo: endpoint que minimiza "subir + ficar no centro"
    # peso w controla quanto a centralidade importa em relação à altura
    w = 0.5
    custo_topo = ys_f + w * np.abs(xs_f - cx)
    indice_topo = int(np.argmin(custo_topo))
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

def desenhar_caule(img, skeleton, caule, topo_vaso, base=None, topo=None):

    img_caule = img.copy()
    #  kernel = np.array(([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), dtype=np.uint8)
    kernel = np.ones((3,3),np.uint8)
    skeleton_grosso = cv2.dilate(skeleton, kernel)
    img_caule[skeleton_grosso > 0] = (200, 200, 0) # skeleton ciano

    for (y, x) in caule:
        cv2.circle(img_caule, (x, y), radius=1, color=(0, 0, 255), thickness=-1) # caule vermelho

    cv2.line(img_caule, (0, topo_vaso), (img_caule.shape[1], topo_vaso), color=(0, 165, 255), thickness=2) # linha alaranjada
    
    # debug:
    # marca base e topo do caule
    if base is not None:
        yb, xb = base
        cv2.circle(img_caule, (xb, yb), radius=12, color=(0, 255, 0), thickness=3)   # verde
    if topo is not None:
        yt, xt = topo
        cv2.circle(img_caule, (xt, yt), radius=12, color=(255, 0, 255), thickness=3) # magenta

        
    return img_caule
# ajustado o diametro do coleto para medir 10 linhas acima do vaso:
def mede_diametro_coleto(mask_planta, caminho, topo_vaso, dy_pedido = 10, tolerancia = 2):

    # Distance TRansform:
    dist = cv2.distanceTransform(mask_planta, cv2.DIST_L2, 5)

    # Filtra os pontos do caminho que estão indo pro skeleton
    # Diametros:
    D = []
    pontos_medidos = []

    for(y, x) in caminho:

        dist_y = topo_vaso - y # Se for positivo, estamos acima do topo do vaso

        if abs(dist_y - dy_pedido) <= tolerancia: 
            # Raio
            R = dist[int(y), int(x)]

            diametro = 2.0 * R
            D.append(diametro)

            pontos_medidos.append((int(y), int(x), float(diametro)))
    
    # Deu Ruim, não achou pontos nessa "janela"
    if not D:
        ponto_mais_proximo = None
        menor = float("inf")

        for (y, x) in caminho:

            d = abs((topo_vaso - y) - dy_pedido)

            if d < menor:
                menor = d
                ponto_mais_proximo = (y,x)
        if ponto_mais_proximo is not None:
            y, x = ponto_mais_proximo

            R = dist[int(y), int(x)]

            diametro = 2.0 * R

            D.append(diametro)

            pontos_medidos.append((int(y), int(x), float(diametro)))

    # mediana dos diametros (tentar evitar ruídos):

    mediana_D = np.median(D)

    diametro_pixels = int(round((mediana_D)))

    #print(f"diâmetro{diametro_pixels}")

    return diametro_pixels, pontos_medidos


def desenha_coleto(resultado, pontos_medidos):


    if not pontos_medidos:
        return resultado
    # usa o ponto do meio da região

    meio = pontos_medidos[len(pontos_medidos) // 2]

    y, x, d = meio

    raio = int(round(d / 2))

    cv2.line(img=resultado, pt1=(x - raio, y), pt2=(x + raio, y), color=(0, 255, 255), thickness=4)

    return resultado

# função que pega a altura básica da planta
def calcula_altura_vertical(skeleton, topo_vaso):

    ys, _ = np.where(skeleton > 0)

    if len(ys) == 0:

        return 0 

    altura_basica = int(topo_vaso - ys.min())

    return altura_basica


