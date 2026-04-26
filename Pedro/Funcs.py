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

def so_a_planta(img, topo_vaso, suavizar = True):

    img_capada = img.copy()
    img_capada[topo_vaso:, :] = 0

    hsv = cv2.cvtColor(img_capada, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_not(cv2.inRange(hsv, lower_azul, upper_azul))
    # tira pixel preto
    mask[hsv[:, :, 2] < 10] = 0

    if not suavizar:
        return maior_blob(mask)

    kernel_morph_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_morph_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_morph_open)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_morph_close)

    return maior_blob(mask_close)

# def extrair_caule_mask(mask_planta, raio=5):
    
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*raio+1, 2*raio+1))

#     caule = cv2.morphologyEx(mask_planta, cv2.MORPH_OPEN, kernel)

#     return maior_blob(caule)
###########################
def extrair_caule_mask(mask_planta, caminho, largura_max=25):
    """
    Constrói máscara do caule percorrendo o caminho. Para cada ponto,
    captura a largura horizontal contínua da máscara da planta naquela
    linha, limitada a `largura_max` pixels para não vazar em folhas.
    """
    h, w = mask_planta.shape
    caule = np.zeros((h, w), dtype=np.uint8)

    for (y, x) in caminho:
        y_int, x_int = int(y), int(x)
        if not (0 <= y_int < h) or not (0 <= x_int < w):
            continue
        if mask_planta[y_int, x_int] == 0:
            continue

        x_esq = x_int
        while (x_esq > 0 and mask_planta[y_int, x_esq - 1] > 0
               and (x_int - x_esq) < largura_max):
            x_esq -= 1
        x_dir = x_int
        while (x_dir < w - 1 and mask_planta[y_int, x_dir + 1] > 0
               and (x_dir - x_int) < largura_max):
            x_dir += 1

        caule[y_int, x_esq:x_dir+1] = 255

    # Closing vertical: conecta pixels onde o caminho saltou linhas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    caule = cv2.morphologyEx(caule, cv2.MORPH_CLOSE, kernel)

    return maior_blob(caule)
##########################################################
def achar_base_topo_caule(skeleton, mask_planta, topo_vaso, cx, raio_caule_max = 20, fracao_altura = 0.8):

    # Encontrar o ponto mais próximo do topo do vaso
    ys, xs = np.where(skeleton > 0)
    
    # Forçar a dar menos enfase no direcionamento em x
    distancia = np.sqrt(0.20 * (xs - cx) ** 2 + (ys - topo_vaso) ** 2)
    # base:
    indice_base = int(np.argmin(distancia))
    base = (int(ys[indice_base]), int(xs[indice_base]))

    # x de ref:
    x_eixo = base[1]

    # altura total da planata para comparação:
    y_topo_planta = int(ys.min())
    # h = altura:
    h_total = topo_vaso - y_topo_planta
    h_alvo = fracao_altura * h_total
    y_alvo = topo_vaso - h_alvo

    # copa(topo):
    kernel = np.array(([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), dtype=np.uint8)
    numero_vizinhos = cv2.filter2D(skeleton, -1, kernel)
    dist = cv2.distanceTransform(mask_planta, cv2.DIST_L2, 5)

    bifurc = (skeleton > 0) & (numero_vizinhos >= 3)
    ys_b, xs_b = np.where(bifurc)

    topo = None
    if len(ys_b) > 0:
        grossuras = dist[ys_b, xs_b]
        print(f"  Antes filtro: {len(ys_b)} bifurc, ys de {ys_b.min()} a {ys_b.max()}")
        mascara_fino = grossuras <= raio_caule_max
        if mascara_fino.sum() > 0:
            ys_b = ys_b[mascara_fino]
            xs_b = xs_b[mascara_fino]

        print(f"  Após filtro grossura<={raio_caule_max}: {len(ys_b)} bifurc, ys de {ys_b.min()} a {ys_b.max()}")
        custo_topo = np.abs(ys_b - y_alvo) + 0.5 * np.abs(xs_b - x_eixo)
        indice_topo = int(np.argmin(custo_topo))
        topo = (int(ys_b[indice_topo]), int(xs_b[indice_topo]))
        #debug:
        print("bifurcação")
        print(f"  bifurcação em y={topo[0]} (alvo={y_alvo:.0f}, h_total={h_total})")

    # reavalira esse plano B?>>>
    if topo is None:
        fins = (skeleton > 0) & (numero_vizinhos == 1)
        ys_f, xs_f = np.where(fins)
        grossuras = dist[ys_f, xs_f]
        mascara_fino = grossuras <= raio_caule_max
        if mascara_fino.sum() > 0:
            ys_f = ys_f[mascara_fino]
            xs_f = xs_f[mascara_fino]

        custo_topo = np.abs(ys_f - y_alvo) + 0.5 * np.abs(xs_f - x_eixo)
        indice_topo = int(np.argmin(custo_topo))
        topo = (int(ys_f[indice_topo]), int(xs_f[indice_topo]))
        # debug:
        print("fallback")
        print(f"  fallback em y={topo[0]} (alvo={y_alvo:.0f})")

    return base, topo


def tracar_caule(skeleton, base, topo, topo_vaso=None, mask_planta=None):
    # Usa skimage.graph.route_through_array, que acha automaticamente o
    # caminho de menor custo entre dois pontos em uma matriz.

    if mask_planta is not None:
        dist = cv2.distanceTransform(mask_planta, cv2.DIST_L2, 5)
        # Pixel do skeleton em região fina (caule): custo baixo
        # Pixel do skeleton em região grossa (folha): custo alto (~dist²)
        custo_skel = 1.0 + (dist.astype(np.float32) ** 2)
        custo = np.where(skeleton > 0, custo_skel, 1e6).astype(np.float32)
    else:
        # A matriz de custo tem valor 1 onde existe skeleton e 1000 onde não existe.
        # Assim o algoritmo vai preferir sempre andar pelo skeleton, porque sair
        # dele custa muito '"caro"
        custo = np.where(skeleton > 0, 1.0, 1000.0).astype(np.float32)

    # para funcionar o skeleton na diagonal o precisa do  fully connected+True 
    caminho, _ = route_through_array(custo, start = base, end = topo, fully_connected=True)
    caminho = list(caminho)

    # tentaiva de adiccionar mais pixels a partir do topo do vaso até a base do skeleton:
    if topo_vaso is not None:

        y_base, x_base = base

        if y_base < topo_vaso:

            extensao = [(y, x_base) for y in range(topo_vaso, y_base)]

            caminho = extensao + caminho

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
#########################################3
# ajustado o diametro do coleto para medir 10 linhas acima do vaso:
def mede_diametro_coleto(mask_planta, caminho, topo_vaso, dy_pedido=10, tolerancia=2):
    dist = cv2.distanceTransform(mask_planta, cv2.DIST_L2, 5)
    D = []
    pontos_medidos = []

    for (y, x) in caminho:
        dist_y = topo_vaso - y
        if abs(dist_y - dy_pedido) <= tolerancia:
            R = dist[int(y), int(x)]
            diametro = 2.0 * R
            D.append(diametro)
            pontos_medidos.append((int(y), int(x), float(diametro)))

    if not D:
        ponto_mais_proximo = None
        menor = float("inf")
        for (y, x) in caminho:
            d = abs((topo_vaso - y) - dy_pedido)
            if d < menor:
                menor = d
                ponto_mais_proximo = (y, x)
        if ponto_mais_proximo is not None:
            y, x = ponto_mais_proximo
            R = dist[int(y), int(x)]
            diametro = 2.0 * R
            D.append(diametro)
            pontos_medidos.append((int(y), int(x), float(diametro)))

    if not D:
        return 0, []
    return int(round(np.median(D))), pontos_medidos
#################################################################3
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
def calcula_altura_vertical(mask_ou_skeleton, topo_vaso):

    ys, _ = np.where(mask_ou_skeleton > 0)

    if len(ys) == 0:

        return 0 

    altura_basica = int(topo_vaso - ys.min())

    return altura_basica


