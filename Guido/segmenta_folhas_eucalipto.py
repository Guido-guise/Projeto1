import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from skimage.morphology import skeletonize


def ler_imagem(caminho_imagem):
    if not os.path.exists(caminho_imagem):
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho_imagem}")

    img_bgr = cv2.imread(caminho_imagem, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Falha ao ler imagem: {caminho_imagem}")
    return img_bgr


def estimar_corte_base_branca_estrutural(img_bgr):
    """
    Detecta a base branca (vaso/suporte) como objeto da cena, nao como fundo.
    A deteccao combina:
    - posicao (metade inferior),
    - aparencia (baixo S / alto V),
    - relacao espacial com a planta (regiao central e abaixo das folhas-semente).
    Nao depende de mask_folhas final.
    """
    h, w = img_bgr.shape[:2]
    if h < 40 or w < 40:
        return h, np.zeros((h, w), dtype=np.uint8), {"y_seed": h - 1, "y_white": h - 1}

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    b, g, r = cv2.split(img_bgr)
    exg = (2 * g.astype(np.int16)) - r.astype(np.int16) - b.astype(np.int16)

    # Semente foliar mais restrita para evitar contaminar o "fundo nao-azul" com a base branca.
    mask_green_seed = np.uint8(
        (h_ch >= 20) & (h_ch <= 110) & (s_ch >= 48) & (v_ch >= 26) & (exg > 4)
    ) * 255
    mask_red_seed = np.uint8(
        (((h_ch <= 24) | (h_ch >= 165)) & (s_ch >= 42) & (v_ch >= 24) & (r.astype(np.int16) >= g.astype(np.int16) + 6))
    ) * 255
    mask_seed = cv2.bitwise_or(mask_green_seed, mask_red_seed)
    mask_seed = cv2.morphologyEx(
        mask_seed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    ys_seed = np.where(mask_seed > 0)[0]
    if ys_seed.size > 120:
        y_seed = int(np.percentile(ys_seed, 97))
    else:
        y_seed = int(0.67 * h)

    # Candidatos da base branca: claros e pouco saturados na metade inferior.
    mask_white = np.uint8((s_ch <= 105) & (v_ch >= 90)) * 255
    lower_band = np.zeros((h, w), dtype=np.uint8)
    y_base_min = int(max(0.45 * h, y_seed - 0.17 * h))
    lower_band[y_base_min:, :] = 255
    mask_white = cv2.bitwise_and(mask_white, lower_band)
    mask_white = cv2.morphologyEx(
        mask_white,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 7)),
        iterations=1,
    )
    mask_white = cv2.morphologyEx(
        mask_white,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    # Seleciona o componente com perfil de vaso/suporte (objeto fisico inferior e claro).
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_white, connectivity=8)
    best_label = -1
    best_score = -1e9
    cx = w / 2.0
    img_area = float(h * w)
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(700, int(0.0015 * img_area)):
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        y_bottom = y + hh
        if y_bottom < int(0.68 * h):
            continue
        if y < int(0.35 * h):
            continue
        width_ratio = ww / float(max(1, w))
        if width_ratio < 0.06 or width_ratio > 0.42:
            continue
        comp = labels == label
        s_mean = float(np.mean(s_ch[comp]))
        v_mean = float(np.mean(v_ch[comp]))
        if s_mean > 118 or v_mean < 78:
            continue
        comp_cx = x + (ww / 2.0)
        centralidade = 1.0 - (abs(comp_cx - cx) / float(max(1.0, cx)))
        area_norm = min(1.0, area / float(max(1, int(0.03 * img_area))))
        bottomness = y_bottom / float(max(1, h))
        low_sat = np.clip((115.0 - s_mean) / 70.0, 0.0, 1.0)
        bright = np.clip((v_mean - 82.0) / 80.0, 0.0, 1.0)
        score = (
            2.6 * area_norm
            + 1.0 * bottomness
            + 1.1 * low_sat
            + 0.8 * bright
            + 0.6 * centralidade
            - 0.8 * abs(width_ratio - 0.18)
        )
        if score > best_score:
            best_score = score
            best_label = label

    mask_base = np.zeros((h, w), dtype=np.uint8)
    y_white = h - 1
    if best_label > 0:
        mask_base[labels == best_label] = 255
        # Expande para incluir reflexos claros conectados ao corpo da base.
        prox = cv2.dilate(mask_base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
        mask_base = cv2.bitwise_or(mask_base, cv2.bitwise_and(mask_white, prox))
        mask_base = cv2.morphologyEx(
            mask_base,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
        ys_base = np.where(mask_base > 0)[0]
        if ys_base.size > 0:
            y_white = int(np.min(ys_base))

    margem_seed = max(12, int(0.06 * h))
    y_min = min(h - 1, y_seed + margem_seed)
    if y_white < (h - 1):
        y_corte = int(np.clip(max(y_min, y_white), int(0.52 * h), h))
    else:
        row_ratio = np.mean(mask_white > 0, axis=1).astype(np.float32)
        row_ratio = np.convolve(row_ratio, np.ones(11, dtype=np.float32) / 11.0, mode="same")
        possiveis = np.where((np.arange(h) >= int(0.52 * h)) & (row_ratio > 0.08))[0]
        if possiveis.size > 0:
            y_corte = int(np.clip(max(y_min, int(possiveis[0])), int(0.52 * h), h))
            y_white = int(possiveis[0])
            mask_base = cv2.bitwise_and(mask_white, np.uint8(np.arange(h)[:, None] >= y_corte) * 255)
        else:
            y_corte = int(np.clip(max(y_min, int(0.72 * h)), int(0.58 * h), h))

    return y_corte, mask_base, {"y_seed": y_seed, "y_white": y_white}


def limpar_componentes_pequenos(mask, area_minima=250):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    mask_limpa = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= area_minima:
            mask_limpa[labels == label] = 255
    return mask_limpa


def obter_bbox_visual_estavel(img_hsv):
    """
    BBox visual independente da segmentacao:
    usa apenas a cena (nao-azul), mantendo planta completa + pote.
    O erro anterior era usar mask_folhas/mask_planta (instaveis) para recorte.
    """
    h, w = img_hsv.shape[:2]
    mask_azul = cv2.inRange(
        img_hsv,
        np.array([100, 120, 150], dtype=np.uint8),
        np.array([120, 255, 255], dtype=np.uint8),
    )
    mask_scene = cv2.bitwise_not(mask_azul)

    base_dim = float(min(h, w))
    k_close = max(5, int(0.012 * base_dim))
    k_open = max(3, int(0.006 * base_dim))
    if k_close % 2 == 0:
        k_close += 1
    if k_open % 2 == 0:
        k_open += 1
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    mask_scene = cv2.morphologyEx(mask_scene, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    mask_scene = cv2.morphologyEx(mask_scene, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Em vez de um unico componente (que pode cortar folhas desconectadas),
    # usa a uniao de componentes relevantes por area.
    img_area = float(h * w)
    area_min = max(120, int(0.00035 * img_area))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_scene, connectivity=8)
    mask_relevante = np.zeros_like(mask_scene)
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= area_min:
            mask_relevante[labels == label] = 255
    if np.count_nonzero(mask_relevante) == 0:
        mask_relevante = mask_scene.copy()

    pts = cv2.findNonZero(mask_relevante)
    if pts is None:
        return 0, 0, w, h, mask_scene

    x, y, ww, hh = cv2.boundingRect(pts)
    mx = max(int(0.06 * ww), int(0.02 * w))
    my_top = max(int(0.08 * hh), int(0.03 * h))
    my_bot = max(int(0.12 * hh), int(0.04 * h))

    # Se o objeto encostar em borda, amplia margem para reduzir risco de corte.
    if x <= int(0.01 * w):
        mx = max(mx, int(0.04 * w))
    if (x + ww) >= int(0.99 * w):
        mx = max(mx, int(0.04 * w))
    if y <= int(0.01 * h):
        my_top = max(my_top, int(0.04 * h))
    if (y + hh) >= int(0.99 * h):
        my_bot = max(my_bot, int(0.06 * h))

    x0 = max(0, x - mx)
    y0 = max(0, y - my_top)
    x1 = min(w, x + ww + mx)
    y1 = min(h, y + hh + my_bot)
    return x0, y0, x1, y1, mask_relevante


def cortar_base_branca_inicio(img_bgr, fracao_fallback=0.69):
    """
    PRIMEIRA etapa do pipeline: detecta base branca e corta a imagem antes de qualquer segmentacao.
    Essa etapa e independente de mask_planta/mask_folhas/mask_caule.
    """
    h, w = img_bgr.shape[:2]
    y_corte_base, mask_base_branca, corte_info = estimar_corte_base_branca_estrutural(img_bgr)

    ys_base = np.where(mask_base_branca > 0)[0]
    if ys_base.size > 0:
        y_top_base = int(np.min(ys_base))
        margem_seg = max(6, int(0.008 * h))
        y_det = y_top_base - margem_seg
    else:
        y_top_base = int(y_corte_base)
        y_det = int(y_corte_base)

    # Fallback inspirado no script "corte": recorte superior percentual para garantir robustez.
    y_fallback = int(fracao_fallback * h)
    y_corte_aplicado = int(np.clip(min(y_det, y_fallback), int(0.52 * h), int(0.80 * h)))
    img_cortada = img_bgr[0:y_corte_aplicado, 0:w].copy()

    # Mascara da base no dominio da imagem original (debug/inspecao).
    mask_corte = np.zeros((h, w), dtype=np.uint8)
    mask_corte[y_corte_aplicado:, :] = 255
    mask_base_original = cv2.bitwise_or(mask_base_branca, mask_corte)

    return img_cortada, {
        "y_corte_aplicado": int(y_corte_aplicado),
        "y_corte_base_estimado": int(y_corte_base),
        "y_top_base_estimado": int(y_top_base),
        "y_seed": int(corte_info["y_seed"]),
        "y_white": int(corte_info["y_white"]),
        "mask_base_branca_original": mask_base_original,
    }


def detectar_eixo_principal_skeleton(mask_binaria):
    skel = gerar_skeleton(mask_binaria)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
    axis = np.zeros_like(skel)
    if n_labels <= 1:
        return skel, axis

    h, w = skel.shape[:2]
    cx = w / 2.0
    best_label = -1
    best_score = -1e9
    for label in range(1, n_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        ww = stats[label, cv2.CC_STAT_WIDTH]
        hh = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        ccx = x + ww / 2.0
        vertical = hh / float(max(1, h))
        centralidade = 1.0 - (abs(ccx - cx) / float(max(1, cx)))
        score = 1.6 * vertical + 0.25 * area + 0.45 * centralidade
        if score > best_score:
            best_score = score
            best_label = label

    if best_label > 0:
        axis[labels == best_label] = 255
    return skel, axis


def criar_overlay_skeleton_axis(skeleton, axis):
    vis = np.zeros((skeleton.shape[0], skeleton.shape[1], 3), dtype=np.uint8)
    vis[skeleton > 0] = (170, 170, 170)
    vis[axis > 0] = (0, 0, 255)
    return vis


def construir_mascaras_cor_foliar(img_bgr, img_hsv):
    """
    Detecta folhas em dois perfis cromaticos:
    - verdes (estagio mais comum)
    - avermelhadas/alaranjadas (folhas jovens)
    """
    b, g, r = cv2.split(img_bgr)
    h_ch, s_ch, v_ch = cv2.split(img_hsv)
    exg = (2 * g.astype(np.int16)) - r.astype(np.int16) - b.astype(np.int16)

    # Folhas verdes (logica original, mantida).
    mask_green_hsv = cv2.inRange(
        img_hsv,
        np.array([18, 24, 28], dtype=np.uint8),
        np.array([98, 255, 255], dtype=np.uint8),
    )
    mask_green_dom = np.uint8(
        (g.astype(np.int16) >= (r.astype(np.int16) - 4))
        & (g.astype(np.int16) >= (b.astype(np.int16) - 6))
        & (s_ch >= 42)
        & (v_ch >= 26)
    ) * 255
    mask_green = cv2.bitwise_and(mask_green_hsv, cv2.bitwise_or(np.uint8(exg > 7) * 255, mask_green_dom))

    # Folhas avermelhadas: combina HSV + relacao entre canais + LAB(a positivo).
    # Isso evita depender de "verde dominante".
    mask_red_hsv_1 = cv2.inRange(img_hsv, np.array([0, 30, 20], dtype=np.uint8), np.array([22, 255, 255], dtype=np.uint8))
    mask_red_hsv_2 = cv2.inRange(img_hsv, np.array([165, 30, 20], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
    mask_red_hsv = cv2.bitwise_or(mask_red_hsv_1, mask_red_hsv_2)
    mask_rdom = np.uint8(
        (r.astype(np.int16) >= (g.astype(np.int16) + 6))
        & (r.astype(np.int16) >= (b.astype(np.int16) + 4))
        & (s_ch >= 24)
        & (v_ch >= 18)
    ) * 255
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a_ch = img_lab[:, :, 1]
    mask_red_lab = np.uint8((a_ch >= 138) & (v_ch >= 18)) * 255
    mask_red = cv2.bitwise_or(
        cv2.bitwise_and(mask_red_hsv, cv2.bitwise_or(mask_rdom, mask_red_lab)),
        cv2.bitwise_and(mask_rdom, mask_red_lab),
    )

    # Evita branco da base com baixa saturacao.
    mask_not_white = np.uint8((s_ch >= 18) | (v_ch <= 170)) * 255
    mask_red = cv2.bitwise_and(mask_red, mask_not_white)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, k3, iterations=1)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, k3, iterations=1)
    mask_cor = cv2.bitwise_or(mask_green, mask_red)
    return mask_green, mask_red, mask_cor


def detectar_caule(mask_folhas, eixo_principal=None, dist_map=None, mask_red_leaf=None):
    """Bloco 3: deteccao de caule com criterio geometrico compacto."""
    if np.count_nonzero(mask_folhas) == 0:
        return np.zeros_like(mask_folhas)

    dist = dist_map if dist_map is not None else cv2.distanceTransform(mask_folhas, cv2.DIST_L2, 3)
    esp = dist * 2.0
    vals = esp[mask_folhas > 0]
    if vals.size == 0:
        return np.zeros_like(mask_folhas)

    t_fino = float(np.clip(np.percentile(vals, 36), 2.8, 6.8))
    mask_fino = np.uint8((mask_folhas > 0) & (esp <= t_fino)) * 255
    mask_fino = cv2.morphologyEx(mask_fino, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    eixo_dil = np.zeros_like(mask_folhas)
    if eixo_principal is not None and np.count_nonzero(eixo_principal) > 0:
        eixo_dil = cv2.dilate(eixo_principal, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
        cand = cv2.bitwise_and(mask_fino, eixo_dil)
    else:
        cand = mask_fino

    area_total = max(1, int(np.count_nonzero(mask_folhas)))
    h_img = mask_folhas.shape[0]
    mask_caule = np.zeros_like(mask_folhas)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 14 or area > int(0.30 * area_total):
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        aspect = max(w, h) / float(max(1, min(w, h)))
        extent = area / float(max(1, w * h))
        vertical = h / float(max(1, h_img))
        comp = np.uint8(labels == label) * 255
        toca_eixo = True if np.count_nonzero(eixo_dil) == 0 else np.any(cv2.bitwise_and(comp, eixo_dil))
        eh_caule = toca_eixo and vertical > 0.05 and (aspect > 1.35 or extent < 0.72)
        if eh_caule and mask_red_leaf is not None and np.count_nonzero(mask_red_leaf) > 0:
            inter_red = int(np.count_nonzero(cv2.bitwise_and(comp, mask_red_leaf)))
            frac_red = inter_red / float(max(1, area))
            if frac_red > 0.34 and aspect < 4.2:
                eh_caule = False
        if eh_caule:
            mask_caule[comp > 0] = 255

    if np.count_nonzero(eixo_dil) > 0:
        conectores = np.uint8((mask_folhas > 0) & (esp <= (t_fino + 0.8))) * 255
        conectores = cv2.bitwise_and(conectores, cv2.dilate(eixo_dil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1))
        mask_caule = cv2.bitwise_or(mask_caule, conectores)

    # Evita remover folhas jovens avermelhadas muito saturadas.
    if mask_red_leaf is not None and np.count_nonzero(mask_red_leaf) > 0:
        mask_caule = cv2.bitwise_and(mask_caule, cv2.bitwise_not(cv2.dilate(mask_red_leaf, np.ones((3, 3), np.uint8))))

    mask_caule = cv2.morphologyEx(mask_caule, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    mask_caule = cv2.morphologyEx(mask_caule, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return mask_caule


def salvar_debug_etapas(nome_base, pasta_saida, etapas):
    debug_dir = os.path.join(pasta_saida, "debug", nome_base)
    os.makedirs(debug_dir, exist_ok=True)
    for nome, img in etapas.items():
        cv2.imwrite(os.path.join(debug_dir, f"{nome}.png"), img)


def gerar_skeleton(mask_binaria):
    mask_bool = mask_binaria > 0
    skel_bool = skeletonize(mask_bool)
    return (skel_bool.astype(np.uint8) * 255)


def segmentar_planta(img_bgr):
    """
    Bloco 1 apos corte inicial: segmentacao da planta no dominio ja sem base branca.
    """
    img_recortada = img_bgr.copy()
    img_hsv = cv2.cvtColor(img_recortada, cv2.COLOR_BGR2HSV)

    mask_azul = cv2.inRange(
        img_hsv,
        np.array([100, 120, 150], dtype=np.uint8),
        np.array([120, 255, 255], dtype=np.uint8),
    )
    mask_planta = cv2.bitwise_not(mask_azul)
    mask_planta = cv2.morphologyEx(
        mask_planta,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    mask_planta = limpar_componentes_pequenos(mask_planta, area_minima=120)
    return img_recortada, img_hsv, mask_azul, mask_planta


def detectar_folhas(img_bgr, img_hsv, mask_planta):
    """Bloco 2: folhas com alta sensibilidade e refino leve."""
    mask_green, mask_red, mask_cor = construir_mascaras_cor_foliar(img_bgr, img_hsv)
    mask_folhas = cv2.bitwise_and(mask_planta, mask_cor)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_folhas = cv2.morphologyEx(mask_folhas, cv2.MORPH_CLOSE, k5, iterations=1)
    mask_folhas = cv2.morphologyEx(mask_folhas, cv2.MORPH_OPEN, k3, iterations=1)
    mask_folhas = limpar_componentes_pequenos(mask_folhas, area_minima=55)

    # Restaurado: watershed condicional.
    # A simplificacao anterior removeu esse refinamento e piorou o contorno de folhas muito conectadas.
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_folhas, connectivity=8)
    area_total = max(1, int(np.count_nonzero(mask_folhas)))
    maior_comp = int(np.max(stats[1:, cv2.CC_STAT_AREA])) if n_labels > 1 else 0
    if n_labels <= 3 and maior_comp > int(0.60 * area_total):
        opening = cv2.morphologyEx(mask_folhas, cv2.MORPH_OPEN, k3, iterations=1)
        sure_bg = cv2.dilate(opening, k3, iterations=2)
        dist_ws = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
        thr = 0.36 * float(dist_ws.max()) if dist_ws.max() > 0 else 0
        _, sure_fg = cv2.threshold(dist_ws, thr, 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img_bgr.copy(), markers)
        ws = np.zeros_like(mask_folhas)
        ws[markers > 1] = 255
        ws = cv2.bitwise_and(ws, mask_folhas)
        if np.count_nonzero(ws) >= int(0.76 * area_total):
            mask_folhas = ws

    dist = cv2.distanceTransform(mask_folhas, cv2.DIST_L2, 3)
    return mask_folhas, dist, mask_green, mask_red, mask_cor


def combinar_resultados(mask_folhas, mask_caule):
    """Bloco 4: combinacao hibrida com salvaguarda de folhas grandes."""
    # Combinacao obrigatoria do pipeline hibrido.
    mask_final = cv2.bitwise_and(mask_folhas, cv2.bitwise_not(mask_caule))
    area_folhas = max(1, int(np.count_nonzero(mask_folhas)))
    area_final = int(np.count_nonzero(mask_final))
    # Salvaguarda simples: evita perder folhas grandes por excesso na mascara de caule.
    if area_final < int(0.62 * area_folhas):
        mask_caule_soft = cv2.erode(mask_caule, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        mask_final = cv2.bitwise_and(mask_folhas, cv2.bitwise_not(mask_caule_soft))
    if np.count_nonzero(mask_final) < int(0.55 * area_folhas):
        mask_final = mask_folhas.copy()

    # Restaurado: salvaguarda geometrica minima de folhas grandes.
    # Sem isso, a subtracao do caule ficava agressiva e removia folha valida.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_folhas, connectivity=8)
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(700, int(0.08 * area_folhas)):
            continue
        comp = np.uint8(labels == label) * 255
        area_comp = max(1, int(np.count_nonzero(comp)))
        overlap = int(np.count_nonzero(cv2.bitwise_and(comp, mask_final))) / float(area_comp)
        if overlap < 0.20:
            mask_final = cv2.bitwise_or(mask_final, comp)

    return limpar_componentes_pequenos(mask_final, area_minima=50)


def segmentar_folhas(img_bgr, fracao_corte=0.69):
    # 1) Detectar base branca e cortar imagem ANTES de qualquer mascara derivada.
    img_sem_base, info_corte = cortar_base_branca_inicio(img_bgr, fracao_fallback=fracao_corte)
    # 2) segmentar_planta no dominio ja cortado.
    img_recortada, img_hsv, mask_azul, mask_planta = segmentar_planta(img_sem_base)
    # 3) bbox visual estavel (independente das mascaras de segmentacao).
    x0, y0, x1, y1, mask_scene_visual = obter_bbox_visual_estavel(img_hsv)
    # 4) detectar_folhas
    mask_folhas, dist_folhas, mask_green, mask_red, mask_cor = detectar_folhas(
        img_recortada, img_hsv, mask_planta
    )
    # 5) detectar_caule (usa dist reaproveitada para evitar recomputacao)
    skel_planta, eixo_planta = detectar_eixo_principal_skeleton(mask_planta)
    mask_caule = detectar_caule(
        mask_folhas,
        eixo_principal=eixo_planta,
        dist_map=dist_folhas,
        mask_red_leaf=cv2.bitwise_and(mask_red, mask_folhas),
    )
    # 6) combinar_resultados
    mask_final_hibrida = combinar_resultados(mask_folhas, mask_caule)

    # Recorte visual separado do processamento.
    # Nunca usa mask_folhas/mask_planta para decidir janela final da imagem.

    img_planta_recortada_raw = img_recortada[y0:y1, x0:x1]
    mask_folhas_recortada = mask_final_hibrida[y0:y1, x0:x1]
    mask_folhas_inicial_crop = mask_folhas[y0:y1, x0:x1]
    # Salvaguarda de recorte: nunca retornar menos area foliar no crop que a mascara inicial menos caule.
    if np.count_nonzero(mask_folhas_recortada) < int(0.55 * max(1, np.count_nonzero(mask_folhas_inicial_crop))):
        mask_folhas_recortada = mask_folhas_inicial_crop.copy()

    # Requisito: planta recortada e apenas recorte da imagem original, sem mascaramento.
    img_planta_recortada = img_planta_recortada_raw.copy()

    area_foliar_px = int(np.count_nonzero(mask_folhas_recortada))
    img_segmentada = cv2.bitwise_and(img_planta_recortada_raw, img_planta_recortada_raw, mask=mask_folhas_recortada)
    skel = gerar_skeleton(mask_folhas_recortada)
    overlay_axis = criar_overlay_skeleton_axis(skel_planta, eixo_planta)
    removidos = cv2.subtract(mask_folhas, mask_final_hibrida)

    return {
        "img_original": img_bgr,
        "img_recortada": img_recortada,
        "img_planta_recortada": img_planta_recortada,
        "img_hsv": img_hsv,
        "mask_verde": mask_folhas,
        "mask_folhas_verdes_cor": cv2.bitwise_and(mask_green, mask_folhas),
        "mask_folhas_vermelhas_cor": cv2.bitwise_and(mask_red, mask_folhas),
        "mask_azul": mask_azul,
        "mask_inicial": mask_folhas,
        "mask_pos_limpeza": mask_folhas,
        "mask_pos_agressiva": mask_final_hibrida,
        "mask_final": mask_folhas_recortada,
        "mask_planta_sem_vaso": mask_planta[y0:y1, x0:x1],
        "mask_caule": mask_caule,
        "mask_watershed": mask_folhas,
        "mask_removidos": removidos,
        "mask_base_branca": info_corte["mask_base_branca_original"],
        "mask_scene_visual": mask_scene_visual,
        "img_segmentada": img_segmentada,
        "skeleton": skel,
        "skeleton_planta": skel_planta,
        "axis_overlay": overlay_axis,
        "area_foliar_px": area_foliar_px,
        "iteracoes_refino": 1,
        "metrics_watershed_before": None,
        "metrics_watershed_after": None,
        "metrics_final_before": None,
        "metrics_final_after": None,
        "corte_base_info": {
            "y_corte_base": info_corte["y_corte_base_estimado"],
            "y_corte_aplicado": info_corte["y_corte_aplicado"],
            "y_top_base_estimado": info_corte["y_top_base_estimado"],
            "y_seed": info_corte["y_seed"],
            "y_white": info_corte["y_white"],
        },
        "janela_info": {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)},
        "hist_refino": [{"step": "hybrid", "ok": True}],
    }


def plotar_etapas(img_bgr, resultados, titulo="Segmentacao Foliar"):
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs = axs.ravel()

    axs[0].imshow(cv2.cvtColor(resultados["img_recortada"], cv2.COLOR_BGR2RGB))
    axs[0].set_title("Imagem Recortada (0-69%)")
    axs[0].axis("off")

    axs[1].imshow(resultados["mask_inicial"], cmap="gray")
    axs[1].set_title("Mascara Inicial (Verde)")
    axs[1].axis("off")

    axs[2].imshow(resultados["mask_final"], cmap="gray")
    axs[2].set_title("Mascara Final (Folhas sem Caule)")
    axs[2].axis("off")

    axs[3].imshow(cv2.cvtColor(resultados["img_segmentada"], cv2.COLOR_BGR2RGB))
    axs[3].set_title("Resultado Segmentado")
    axs[3].axis("off")

    axs[4].imshow(resultados["skeleton"], cmap="gray")
    axs[4].set_title("Skeleton (Opcional)")
    axs[4].axis("off")

    axs[5].imshow(resultados["mask_pos_agressiva"], cmap="gray")
    axs[5].set_title("Pos Filtro Agressivo")
    axs[5].axis("off")

    fig.suptitle(f"{titulo} | Area foliar: {resultados['area_foliar_px']} px", fontsize=12)
    plt.tight_layout()
    plt.show()


def processar_imagem(caminho_imagem, salvar_saida=True, pasta_saida="Projeto1/Guido/saida_segmentacao"):
    img_bgr = ler_imagem(caminho_imagem)
    resultados = segmentar_folhas(img_bgr)

    if salvar_saida:
        os.makedirs(pasta_saida, exist_ok=True)
        nome_base = os.path.splitext(os.path.basename(caminho_imagem))[0]

        cv2.imwrite(os.path.join(pasta_saida, f"{nome_base}_mask_folhas.png"), resultados["mask_final"])
        cv2.imwrite(os.path.join(pasta_saida, f"{nome_base}_segmentada.png"), resultados["img_segmentada"])
        cv2.imwrite(os.path.join(pasta_saida, f"{nome_base}_planta_recortada.png"), resultados["img_planta_recortada"])
        cv2.imwrite(os.path.join(pasta_saida, f"{nome_base}_skeleton.png"), resultados["skeleton"])
        salvar_debug_etapas(
            nome_base,
            pasta_saida,
            {
                "01_recortada": resultados["img_recortada"],
                "02_mask_inicial": resultados["mask_inicial"],
                "03_mask_watershed": resultados["mask_watershed"],
                "04_mask_pos_agressiva": resultados["mask_pos_agressiva"],
                "05_mask_final": resultados["mask_final"],
                "06_componentes_removidos": resultados["mask_removidos"],
                "07_skeleton_axis_overlay": resultados["axis_overlay"],
                "08_segmentada": resultados["img_segmentada"],
            },
        )

    return resultados


def processar_lote(lista_imagens):
    linhas = []

    for caminho in lista_imagens:
        resultados = processar_imagem(caminho, salvar_saida=True)
        linhas.append(
            {
                "arquivo": os.path.basename(caminho),
                "area_foliar_px": resultados["area_foliar_px"],
            }
        )

    tabela = pd.DataFrame(linhas).sort_values("arquivo").reset_index(drop=True)
    return tabela


def plotar_lote(lista_imagens, salvar_figura=True, caminho_figura="Projeto1/Guido/saida_segmentacao/painel_lote.png"):
    n = len(lista_imagens)
    fig, axs = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axs = np.array([axs])

    for i, caminho in enumerate(lista_imagens):
        img = ler_imagem(caminho)
        res = segmentar_folhas(img)

        axs[i, 0].imshow(cv2.cvtColor(res["img_planta_recortada"], cv2.COLOR_BGR2RGB))
        axs[i, 0].set_title(f"{os.path.basename(caminho)} | Planta Recortada")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(res["mask_final"], cmap="gray")
        axs[i, 1].set_title(f"Mascara Final | Area: {res['area_foliar_px']} px")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(cv2.cvtColor(res["img_segmentada"], cv2.COLOR_BGR2RGB))
        axs[i, 2].set_title("Segmentada (Folhas)")
        axs[i, 2].axis("off")

    plt.tight_layout()

    if salvar_figura:
        os.makedirs(os.path.dirname(caminho_figura), exist_ok=True)
        fig.savefig(caminho_figura, dpi=150, bbox_inches="tight")

    plt.show()


def plotar_individual_por_imagem(
    lista_imagens,
    salvar_figuras=True,
    pasta_saida="Projeto1/Guido/saida_segmentacao/plots_individuais",
):
    if salvar_figuras:
        os.makedirs(pasta_saida, exist_ok=True)

    for caminho in lista_imagens:
        img = ler_imagem(caminho)
        res = segmentar_folhas(img)
        nome_base = os.path.splitext(os.path.basename(caminho))[0]

        fig, axs = plt.subplots(1, 4, figsize=(16, 4))

        axs[0].imshow(cv2.cvtColor(res["img_original"], cv2.COLOR_BGR2RGB))
        axs[0].set_title(f"{nome_base} | Original")
        axs[0].axis("off")

        axs[1].imshow(cv2.cvtColor(res["img_planta_recortada"], cv2.COLOR_BGR2RGB))
        axs[1].set_title("Planta Recortada")
        axs[1].axis("off")

        axs[2].imshow(res["mask_final"], cmap="gray")
        axs[2].set_title(f"Mascara Final | Area: {res['area_foliar_px']} px")
        axs[2].axis("off")

        axs[3].imshow(cv2.cvtColor(res["img_segmentada"], cv2.COLOR_BGR2RGB))
        axs[3].set_title("Segmentada")
        axs[3].axis("off")

        plt.tight_layout()

        if salvar_figuras:
            caminho_plot = os.path.join(pasta_saida, f"{nome_base}_plot.png")
            fig.savefig(caminho_plot, dpi=150, bbox_inches="tight")

        plt.show()


def preparar_tile_cv2(img_bgr, tamanho=(420, 320), titulo=""):
    alvo_w, alvo_h = tamanho
    h, w = img_bgr.shape[:2]
    escala = min(alvo_w / max(1, w), alvo_h / max(1, h))
    novo_w = max(1, int(w * escala))
    novo_h = max(1, int(h * escala))
    img_r = cv2.resize(img_bgr, (novo_w, novo_h))

    canvas = np.zeros((alvo_h + 36, alvo_w, 3), dtype=np.uint8)
    y0 = 36 + (alvo_h - novo_h) // 2
    x0 = (alvo_w - novo_w) // 2
    canvas[y0 : y0 + novo_h, x0 : x0 + novo_w] = img_r

    cv2.putText(
        canvas,
        titulo,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def exibir_lote_cv2(
    lista_imagens,
    salvar_figura=True,
    caminho_figura="Projeto1/Guido/saida_segmentacao/painel_lote_cv2.png",
):
    linhas = []

    for caminho in lista_imagens:
        img = ler_imagem(caminho)
        res = segmentar_folhas(img)
        nome = os.path.basename(caminho)

        t1 = preparar_tile_cv2(res["img_planta_recortada"], titulo=f"{nome} | Planta")
        t2 = preparar_tile_cv2(res["img_segmentada"], titulo=f"Folhas | Area {res['area_foliar_px']} px")

        linha = np.hstack([t1, t2])
        linhas.append(linha)

    painel = np.vstack(linhas) if linhas else np.zeros((360, 840, 3), dtype=np.uint8)

    if salvar_figura:
        os.makedirs(os.path.dirname(caminho_figura), exist_ok=True)
        cv2.imwrite(caminho_figura, painel)

    cv2.imshow("Lote - Segmentacao de Folhas", painel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def exibir_individual_cv2(
    lista_imagens,
    salvar_figuras=True,
    pasta_saida="Projeto1/Guido/saida_segmentacao/plots_individuais_cv2",
):
    if salvar_figuras:
        os.makedirs(pasta_saida, exist_ok=True)

    for caminho in lista_imagens:
        img = ler_imagem(caminho)
        res = segmentar_folhas(img)
        nome_base = os.path.splitext(os.path.basename(caminho))[0]

        planta = preparar_tile_cv2(res["img_planta_recortada"], titulo="Planta Recortada")
        seg = preparar_tile_cv2(res["img_segmentada"], titulo=f"Folhas | Area {res['area_foliar_px']} px")
        painel = np.hstack([planta, seg])

        if salvar_figuras:
            cv2.imwrite(os.path.join(pasta_saida, f"{nome_base}_plot_cv2.png"), painel)

        cv2.imshow(f"Imagem - {nome_base}", painel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    imagens = [
        "Projeto1/_Eucalipto_Escolhidos1/Eucalipto1.jpg",
        "Projeto1/_Eucalipto_Escolhidos1/Eucalipto2.jpg",
        "Projeto1/_Eucalipto_Escolhidos1/Eucalipto3.jpg",
        "Projeto1/_Eucalipto_Escolhidos1/Eucalipto4.jpg",
        "Projeto1/_Eucalipto_Escolhidos1/Eucalipto5.jpg",
    ]

    # Processa em lote e mostra tabela de areas.
    df_areas = processar_lote(imagens)
    print("\nArea foliar total por imagem (pixels):")
    print(df_areas.to_string(index=False))

    # Sem painel geral: somente visualizacoes individuais.
    exibir_individual_cv2(imagens, salvar_figuras=True)
