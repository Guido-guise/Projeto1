# Bibliotecas importadas:
import cv2
import numpy as np
from skimage.graph import route_through_array

#Thresholds HSV fixos:
# (Fundo azul)
lower_azul = np.array([100, 120, 150])
upper_azul = np.array([120, 255, 255])

## Funções para utilizar no projeto-Main.py ##

# Parte inicial de pré-processamento da imagem:

def melhorar_altura(img, altura_do_chao):
    """
    Redimensiona a imagem para uma altura fixa, preservando o aspect ratio.

    Padroniza a resolução vertical garante que parâmetros em pixels
    (ex.: raio do caule, tamanho de kernels morfológicos) sejam comparáveis
    entre imagens capturadas em diferentes distâncias.

    Utiliza INTER_AREA, que tende a preservar melhor a informação em
    downsampling ao considerar áreas de pixels (reduz aliasing em estruturas
    finas como folhas e caule).
    """
    (h, w, c) = img.shape

    if h == altura_do_chao:
        return img

    escala =  altura_do_chao / h
    # https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
    
    img_redimensionada = cv2.resize(img, (int(w * escala), altura_do_chao), interpolation=cv2.INTER_AREA)

    return img_redimensionada

## Parte de topologia interna utilizada por outras funções internamente:
def maior_blob(mask):
    """
    Mantém apenas a maior componente conexa (8-connectivity) da máscara.
 
    Usado em vários pontos do pipeline para descartar ruído desconectado:
    a planta+vaso, o vaso isolado, a planta segmentada — todos devem ser uma
    única componente. Qualquer pixel ligado por mero acaso é eliminado.
 
    Implementação: connectedComponentsWithStats retorna o rótulo 0 como fundo,
    por isso o argmax é deslocado em +1 e parte de stats[1:].
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask
    # [0] É O FUNDO:
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    mcomp = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    return mcomp
## Segmentações: planta + vaso, depois vaso isolado, depois só a planta:
def extrair_planta_vaso(img):
    """
    Separa do fundo azul tudo que é planta + vaso.
 
    Trabalha em HSV pois é invariante a brilho — o azul fotográfico tem H
    estável mesmo com sombra parcial, enquanto em RGB há variações.
 
    O kernel em cruz (4-vizinhos do centro) no MORPH_OPEN remove pixels
    isolados (poeira, partículas) sem afetar a topologia de estruturas finas
    como folhas alongadas — um kernel cheio (3x3) erode as pontas das folhas.
 
    Retorna a máscara binária e a imagem HSV (reaproveitada por outras
    funções para evitar reconverter).
    """
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
    """
    Isola o vaso e retorna o y do topo + x do centro horizontal do topo.
 
    Estratégia: o vaso é cinza/preto, ou seja, tem saturação muito baixa
    (≤ saturacao_max em HSV). A planta, por ser verde, tem saturação alta.
    Filtrar a máscara de planta+vaso por saturação baixa = vaso isolado.
 
    Sequência morfológica:
      - CLOSE com elipse 15 x 15: preenche reflexos brancos e furos internos
        do vaso (causados por iluminação direta no plástico).
      - OPEN com elipse 9 x 9: remove manchas pequenas que sobreviveram
        ao filtro de saturação (bordas de folhas escuras, sombras).
 
    O centro horizontal cx é a média dos extremos esquerdo/direito da
    primeira faixa de 20 px do topo. Serve como referência para identificar
    a base do caule (que deve estar próxima de cx).
    """
    
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
    """
    Extrai a máscara da planta (apenas acima do topo do vaso).
 
    Duas variantes selecionadas via `suavizar`:
 
      - suavizar=True: aplica OPEN(3 x 3) + CLOSE(5 x 5). OPEN remove pixels
        soltos e estreita conexões espúrias (folhas grudadas pelo limiar);
        CLOSE preenche micro-falhas internas. Resultado é uma máscara
        topologicamente limpa, ideal para esqueletização (skeletonize fica
        muito sensível a buracos e protuberâncias).
 
      - suavizar=False: máscara crua, sem morfologia. Preserva a folha mais
        alta com fidelidade, o que importa para a medida de altura vertical
        (qualquer erosão pode "podar" a folha do topo).
 
    Pixels muito escuros (V < 10) são forçados a 0: cobre o caso de pretos
    fora do range HSV do azul (bordas internas do vaso ainda visíveis).
    """
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
## CAMINHADA NO SKELETON
def subir_no_skeleton(ponto_inicial, skeleton, mask_planta, visitados=None, raio_max=11, passos_max=120, y_limite = None, y_topo_planta = None):
    """
    Caminhada gulosa subindo no skeleton a partir de um ponto.
 
    Critérios de parada (qualquer um interrompe):
      - Sem vizinhos do skeleton ainda não visitados acima do pixel atual.
      - Pixel candidato em região grossa (dist[ny,nx] > raio_max), o que
        sinaliza entrada em folha em vez de continuar pelo caule.
      - Limite vertical absoluto (`y_limite`) ou aproximação do topo da
        planta (`y_topo_planta + 3`).
      - Pixel atingido é uma bifurcação real (≥ 3 vizinhos no skeleton).
 
    Em pixels com múltiplos vizinhos viáveis, escolhe o que melhor alinha
    com a direção anterior (produto escalar máximo). Isso garante
    continuidade quando o skeleton tem pequenas ramificações laterais
    (folhas curtas que não interrompem o caule).
 
    O parâmetro `direcao_anterior` é mantido como vetor unitário:
      score = cos(ângulo) entre a direção candidata e a anterior.
    Score próximo de 1 = mesma direção, próximo de 0 = perpendicular,
    negativo = direção oposta.
    """
    h, w = skeleton.shape
    dist = cv2.distanceTransform(mask_planta, cv2.DIST_L2, 5)

    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    n_viz = cv2.filter2D(skeleton, -1, kernel)

    if visitados is None:
        visitados = set()

    y, x = int(ponto_inicial[0]), int(ponto_inicial[1])
    caminho = []

    visitados.add((y, x))

    direcao_anterior = None

    for _ in range(passos_max):
        candidatos = []

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue

                ny, nx = y + dy, x + dx

                if (0 <= ny < h and 0 <= nx < w
                    and skeleton[ny, nx] > 0
                    and (ny, nx) not in visitados
                    and ny < y):
                    candidatos.append((ny, nx))

        if not candidatos:
            print(f"    [subir] parou em y={y}: sem candidatos")
            break

        if len(candidatos) > 1 and direcao_anterior is not None:
            melhor = None
            melhor_score = -1

            for ny, nx in candidatos:
                dy = ny - y
                dx = nx - x

                norm = np.hypot(dy, dx)
                if norm == 0:
                    continue

                dy /= norm
                dx /= norm

                # produto escalar → alinhamento
                score = dy * direcao_anterior[0] + dx * direcao_anterior[1]

                if score > melhor_score:
                    melhor_score = score
                    melhor = (ny, nx)

            if melhor is None:
                break

            ny, nx = melhor
        else:
            ny, nx = candidatos[0]
        # limite global:
        if y_limite is not None and ny < y_limite:
            print(f"    [subir] parou em y={y}: y_limite={y_limite}")
            break
        
        # não passa topo
        if y_topo_planta is not None and ny < y_topo_planta + 3:
            print(f"    [subir] parou em y={y}: perto do topo da planta")
            break

        # entrou em folha
        if dist[ny, nx] > raio_max:
            print(f"    [subir] parou em y={y}: dist={dist[ny,nx]:.1f} > raio_max={raio_max}")
            break
        

        # atualiza direção
        dy = ny - y
        dx = nx - x
        norm = np.hypot(dy, dx)

        if norm > 0:
            direcao_anterior = (dy / norm, dx / norm)


        caminho.append((ny, nx))
        visitados.add((ny, nx))

        y, x = ny, nx

        # chegou em bifurcação → pode parar
        #if n_viz[ny, nx] >= 3:
            #print(f"    [subir] parou em y={y}: bifurcação")
                    # bifurcação: só para se nenhum ramo continuar fino e alinhado
        if n_viz[ny, nx] >= 3:
            # olha os vizinhos do pixel atual (excluindo o de onde veio)
            ramos_viaveis = []
            for ddy in (-1, 0, 1):
                for ddx in (-1, 0, 1):
                    if ddy == 0 and ddx == 0:
                        continue
                    ry, rx = ny + ddy, nx + ddx
                    if not (0 <= ry < h and 0 <= rx < w):
                        continue
                    if skeleton[ry, rx] == 0 or (ry, rx) in visitados:
                        continue
                    if ry >= ny:  # só pra cima
                        continue
                    if dist[ry, rx] > raio_max:
                        continue
                    ramos_viaveis.append((ry, rx))

            if not ramos_viaveis:
                break  # nenhum ramo viável, para mesmo

            # se há candidatos, escolhe o mais alinhado com a direção atual
            if direcao_anterior is not None and len(ramos_viaveis) > 1:
                melhor_ramo = None
                melhor_score = -2
                for ry, rx in ramos_viaveis:
                    rdy = ry - ny
                    rdx = rx - nx
                    rnorm = np.hypot(rdy, rdx)
                    if rnorm == 0:
                        continue
                    score = (rdy / rnorm) * direcao_anterior[0] + (rdx / rnorm) * direcao_anterior[1]
                    if score > melhor_score:
                        melhor_score = score
                        melhor_ramo = (ry, rx)
                # exige alinhamento mínimo (>0 = pelo menos não está indo pro lado oposto)
                if melhor_ramo is None or melhor_score < 0.3:
                    break
            #break

    return caminho, (y, x)

## IDENTIFICAÇÃO DA BASE E DO TOPO DO CAULE
def achar_base_topo_caule(skeleton, mask_planta, topo_vaso, cx, raio_caule_max = 14, fracao_altura = 0.80):
    """
    Localiza dois pontos críticos no skeleton: BASE (saída do vaso) e TOPO
    (final do caule, antes da copa).
 
    BASE — heurística de proximidade do topo do vaso:
        score(p) = sqrt( 0.20·(x_p - cx)² + (y_p - topo_vaso)² )
    O fator 0.20 em x reduz o peso lateral para que o algoritmo priorize
    pegar um pixel próximo do topo do vaso mesmo quando a planta está
    inclinada (caule curvado). Sem essa atenuação, a base ficaria presa
    no centro mesmo se a planta nasce ligeiramente fora dele.
 
    TOPO — busca entre as bifurcações finas:
      1. Filtra bifurcações (pixels com 3+ vizinhos) cuja grossura local
         (distance transform) seja ≤ raio_caule_max. Bifurcações em região
         grossa estão dentro de folhas, não no caule.
      2. Entre as restantes, escolhe a que minimiza o custo:
            |y - y_alvo| + 0.5·|x - x_eixo| + 2.0·dist
         onde:
           - y_alvo = topo_vaso - 0.8·h_total (altura típica do final do
             caule em mudas saudáveis);
           - x_eixo = x da base (caule tende a ser vertical);
           - dist = grossura local — favorece pixels mais finos (mais
             provavelmente caule do que junção de folhas).
      3. Refina o ponto com `subir_no_skeleton` para deslocar mais para
         cima caso o caule continue fino e único acima da bifurcação
         escolhida (situação comum quando a primeira bifurcação encontrada
         é uma folhinha lateral, não a copa real).
 
    Plano B (`if topo is None`): se não houver nenhuma bifurcação, usa
    pontas finas (vizinhos == 1) com o mesmo critério de custo.
    """

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

    y_limite = y_alvo - 0.1 * h_total

    # copa(topo):
    kernel = np.array(([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), dtype=np.uint8)
    numero_vizinhos = cv2.filter2D(skeleton, -1, kernel)
    dist = cv2.distanceTransform(mask_planta, cv2.DIST_L2, 5)

    bifurc = (skeleton > 0) & (numero_vizinhos >= 3)
    ys_b, xs_b = np.where(bifurc)

    topo = None
    if len(ys_b) > 0:
        grossuras = dist[ys_b, xs_b]
        #print(f"  Antes filtro: {len(ys_b)} bifurc, ys de {ys_b.min()} a {ys_b.max()}")
        mascara_fino = grossuras <= raio_caule_max
        if mascara_fino.sum() > 0:
            ys_b = ys_b[mascara_fino]
            xs_b = xs_b[mascara_fino]

        #print(f"  Após filtro grossura<={raio_caule_max}: {len(ys_b)} bifurc, ys de {ys_b.min()} a {ys_b.max()}")
        #custo = posição + penalidade de grossura
        custo_topo = (np.abs(ys_b - y_alvo) + 0.5 * np.abs(xs_b - x_eixo) + 2.0 * dist[ys_b, xs_b]) 
        indice_topo = int(np.argmin(custo_topo))
        topo = (int(ys_b[indice_topo]), int(xs_b[indice_topo]))

        _, topo = subir_no_skeleton(topo, skeleton, mask_planta, raio_max=raio_caule_max, y_limite=y_limite, y_topo_planta=y_topo_planta)
        #debug:
        #print("bifurcação")
        #print(f"  bifurcação em y={top[0]} (alvo={y_alvo:.0f}, h_total={h_total})")

    # reavalira esse plano B?>>>
    if topo is None:
        fins = (skeleton > 0) & (numero_vizinhos == 1)
        ys_f, xs_f = np.where(fins)

        grossuras = dist[ys_f, xs_f]
        mascara_fino = grossuras <= raio_caule_max

        if mascara_fino.sum() > 0:

            ys_f = ys_f[mascara_fino]
            xs_f = xs_f[mascara_fino]

        custo_topo = (np.abs(ys_f - y_alvo) + 0.5 * np.abs(xs_f - x_eixo) + 2.0 * dist[ys_f, xs_f])
        indice_topo = int(np.argmin(custo_topo))

        topo = (int(ys_f[indice_topo]), int(xs_f[indice_topo]))

        _, topo = subir_no_skeleton(topo, skeleton, mask_planta, raio_max=raio_caule_max, passos_max=120, y_limite=y_limite, y_topo_planta=y_topo_planta)
        # debug:
        #print("fallback")
        #print(f"  fallback em y={topo[0]} (alvo={y_alvo:.0f})")

    return base, topo
## Caminho do base ao topo:
def tracar_caule(skeleton, base, topo, topo_vaso=None, mask_planta=None):
    """
    Calcula o caminho ótimo da base ao topo no skeleton e retorna seu
    comprimento em pixels.
 
    Etapa 1 — Dijkstra (route_through_array):
        Define matriz de custo onde transitar pelo skeleton custa pouco
        e fora dele custa caro (1e6). Dentro do skeleton, o custo soma:
          - 1.0  : custo base por pixel
          - dist²: penaliza fortemente regiões grossas (folhas) — preferência
                   matemática pelo skeleton fino e contínuo do caule
          - 2.0·|x - x_base|: penalidade lateral; puxa o caminho para perto
                   do eixo vertical da base, evitando "atalhos" laterais
        `fully_connected=True` permite movimento diagonal — necessário pois
        skimage.morphology.skeletonize gera diagonais.
 
    Etapa 2 — Poda condicional:
        Trunca o caminho num ponto próximo a 0.85·h_total acima do vaso.
        O ponto exato é o que minimiza |y - y_alvo| + 0.02·y, onde o termo
        0.02·y desempata favorecendo pontos mais altos quando vários
        candidatos têm distância similar ao alvo. A poda é necessária
        porque o Dijkstra muitas vezes "exagera" e sobe pela copa,
        seguindo pelo skeleton de uma folha que desce e cruza o caminho.
 
    Etapa 3 — Extensão inferior:
        Se a base do skeleton está acima do topo do vaso, adiciona pixels
        sintéticos descendo verticalmente (mesmo x da base) até o topo
        do vaso. Garante que o coleto é incluído no comprimento total.
 
    Etapa 4 — Extensão superior pelo skeleton:
        A poda em 0.85 é conservadora; em algumas plantas o caule real
        continua fino e único acima desse ponto. `subir_no_skeleton`
        aproveita o que ainda existir de caule contínuo até esbarrar em
        folha ou bifurcação real.
 
    Comprimento: soma das hipotenusas dos passos do caminho final.
    """   
   
    # Usa skimage.graph.route_through_array, que acha automaticamente o
    # caminho de menor custo entre dois pontos em uma matriz.

    if mask_planta is not None:
        dist = cv2.distanceTransform(mask_planta, cv2.DIST_L2, 5)
        # Pixel do skeleton em região fina (caule): custo baixo
        # Pixel do skeleton em região grossa (folha): custo alto (~dist²)
        #custo_skel = 1.0 + (dist.astype(np.float32) ** 2)
        xs = np.arange(skeleton.shape[1])[None, :]
        penalidade_x = np.abs(xs - base[1])
        custo_skel = 1.0 + 3.0 *  (dist.astype(np.float32) ** 2) + 2.0 * penalidade_x

        custo = np.where(skeleton > 0, custo_skel, 1e6).astype(np.float32)
    else:
        # A matriz de custo tem valor 1 onde existe skeleton e 1000 onde não existe.
        # Assim o algoritmo vai preferir sempre andar pelo skeleton, porque sair
        # dele custa muito '"caro"
        custo = np.where(skeleton > 0, 1.0, 1000.0).astype(np.float32)

    # para funcionar o skeleton na diagonal o precisa do  fully connected+True 
    caminho, _ = route_through_array(custo, start = base, end = topo, fully_connected=True)
    caminho = list(caminho)
    # Poda condicional:
    if topo_vaso is not None and len(caminho) > 0:

        ys = [y for (y, _) in caminho]

        y_topo_planta = min(ys)
        h_total = topo_vaso - y_topo_planta
        fator  = 0.88 # 0.84 é top

        y_alvo = topo_vaso - fator * h_total

        idx_corte = np.argmin([abs(y - y_alvo) + 0.02 * y for y in ys])

        caminho = caminho[:idx_corte + 1]

    # tentaiva de adiccionar mais pixels a partir do topo do vaso até a base do skeleton:
    if topo_vaso is not None:

        y_base, x_base = base

        if y_base < topo_vaso:

            extensao = [(y, x_base) for y in range(topo_vaso, y_base)]

            caminho = extensao + caminho

    # NOVO: estende o caminho seguindo o skeleton para cima enquanto for
    # ramo único e fino (resolve Img 2 onde o filtro de grossura cortou cedo)
    if mask_planta is not None and caminho:
        ys, _ = np.where(skeleton > 0)

        y_topo_planta = int(ys.min())

        visitados = set((int(y), int(x)) for y, x in caminho)

        extensao, _ = subir_no_skeleton(caminho[-1], skeleton, mask_planta, visitados=visitados, raio_max=13, passos_max=60, y_limite=None, y_topo_planta=y_topo_planta)

        caminho.extend(extensao)

    # Comprimento do caule:
    comprimento = 0.0
    for i in range(1, len(caminho)):
        dy = caminho[i][0] - caminho[i-1][0]
        dx = caminho[i][1] - caminho[i-1][1]
        comprimento += np.hypot(dy, dx) # hipotenusa

    return caminho, comprimento

def desenhar_caule(img, skeleton, caule, topo_vaso, base=None, topo=None):
    """
    Sobrepõe na imagem original:
      - skeleton inteiro (ciano, dilatado para visibilidade);
      - caminho do caule (vermelho);
      - linha do topo do vaso (laranja);
      - base do caule (círculo verde);
      - topo do caule (círculo magenta).
 
    A dilatação do skeleton (kernel 3x3 cheio) é puramente cosmética —
    1 pixel de skeleton seria invisível em telas comuns.
    """
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
    """
    Mede o diâmetro do caule numa altura específica acima do topo do vaso.
 
    Usa a distance transform: para cada pixel da máscara, dist[y,x] = raio
    do maior círculo centrado em (y,x) que cabe inteiro dentro da máscara.
    Logo, 2·dist[y,x] = espessura local da estrutura naquele ponto.
 
    Estratégia:
      1. Coleta todos os pontos do caminho cuja distância vertical ao topo
         do vaso esteja em [dy_pedido - tol, dy_pedido + tol].
      2. Mede o diâmetro em cada um e retorna a mediana — robusta a outliers
         causados por imperfeições da máscara.
      3. Fallback: se nenhum ponto cair na faixa, usa o ponto mais próximo
         de dy_pedido (impede retorno vazio em mudas muito baixas).
 
    Retorna a mediana dos diâmetros e a lista de pontos medidos (usada
    pela visualização).
    """    
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
# desenha o local da medida do diâmetro do coleto:
def desenha_coleto(resultado, pontos_medidos):
    """
    Desenha um traço amarelo horizontal mostrando o diâmetro medido no
    coleto. Usa o ponto do meio da lista de medições — assim a linha
    aparece centrada na faixa onde o diâmetro foi efetivamente computado.
    """

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
    """
    Altura básica = distância vertical entre o topo do vaso e o pixel mais
    alto da máscara da planta.
 
    Recebe `mask_planta_rude` (sem morfologia) no Main.py: morfologia
    poderia "podar" pixels do topo de uma folha alta, subestimando a altura.
    A versão rude preserva toda a folha visível, ao custo de incluir
    eventual ruído — mas o `maior_blob` já garante que esse ruído não esteja
    desconectado da planta principal.
    """
    ys, _ = np.where(mask_ou_skeleton > 0)

    if len(ys) == 0:

        return 0 

    altura_basica = int(topo_vaso - ys.min())

    return altura_basica


