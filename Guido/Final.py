import os, cv2
import numpy as np
from skimage.morphology import skeletonize
CAMINHO_IMAGEM = "Projeto1/_Eucalipto_Escolhidos2/Eucalipto{}.jpg"
CORTE_ALTURA = 0.64
HSV_MIN = np.array([18, 38, 38], dtype=np.uint8)
HSV_MAX = np.array([96, 255, 255], dtype=np.uint8)
K8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
AREA_MINIMA = 15
SAIDA = "Projeto1/Guido/saida_contagem"
#Area
def limpar_componentes_pequenos(mask, area_minima):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    mask_limpa = np.zeros_like(mask)

    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= area_minima:
            mask_limpa[labels == label] = 255

    return mask_limpa


def segmentar(img):
    img = img[: int(img.shape[0] * CORTE_ALTURA), :]
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_MIN, HSV_MAX)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return img, mask
def eixo_caule(mask, n_faixas=26):
    h, w = mask.shape
    ys, xs = [], []
    cortes = np.linspace(0, h, n_faixas + 1, dtype=int)
    skel = skeletonize(mask > 0).astype(np.uint8)
    x_ant = w / 2.0
    for y0, y1 in zip(cortes[-2::-1], cortes[:0:-1]):
        _, faixa_x = np.where(skel[y0:y1] > 0)
        if len(faixa_x) > 4:
            perto = faixa_x[np.abs(faixa_x - x_ant) < 120]
            base = perto if len(perto) > 2 else faixa_x
            x_ant = float(np.median(base))
            ys.append((y0 + y1) / 2)
            xs.append(x_ant)
    if len(xs) < 2:
        return np.array([0.0, h - 1.0]), np.array([w / 2.0, w / 2.0])
    return np.array(ys[::-1]), np.array(xs[::-1])
def x_caule(y, eixo_y, eixo_x): return np.interp(y, eixo_y, eixo_x)
def angulo_diff(a, b):
    d = abs(a - b) % 180.0; return min(d, 180.0 - d)
def podar_skeleton(skel, n=2):
    skel = skel.copy()
    for _ in range(n):
        grau = cv2.filter2D(skel, -1, K8)
        skel[(skel > 0) & (grau == 1)] = 0
    return skel
def componente_info(pontos, eixo_y, eixo_x):
    pts = pontos.astype(np.float32)
    cx, cy = pts.mean(axis=0)
    cov = np.cov((pts - [cx, cy]).T)
    vals, vecs = np.linalg.eigh(cov)
    vx, vy = vecs[:, int(np.argmax(vals))]
    ang = (np.degrees(np.arctan2(vy, vx)) + 180.0) % 180.0
    dx, dy = np.ptp(pts[:, 0]) + 1, np.ptp(pts[:, 1]) + 1
    dist_eixo = np.mean(np.abs(pts[:, 0] - x_caule(pts[:, 1], eixo_y, eixo_x)))
    return {"pts": pts, "cx": cx, "cy": cy, "ang": ang, "n": len(pts),
            "dx": dx, "dy": dy, "dist": dist_eixo}
def componentes_skeleton(mask, eixo_y, eixo_x):
    h, w = mask.shape
    skel = podar_skeleton(skeletonize(mask > 0).astype(np.uint8), 1)
    grau = cv2.filter2D(skel, -1, K8)
    segmentos = np.uint8((skel > 0) & (grau <= 2))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(segmentos, connectivity=8)
    comps = []
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] < max(5, int(0.003 * h)):
            continue
        ys, xs = np.where(labels == lab)
        c = componente_info(np.column_stack([xs, ys]), eixo_y, eixo_x)
        vertical = angulo_diff(c["ang"], 90.0) < 25
        central = c["dist"] < 0.014 * w
        caule = central and c["dx"] < 0.011 * w and c["dy"] > 1.25 * c["dx"] and (vertical or c["dy"] > 0.018 * h)
        if not caule:
            c["H"], c["W"] = h, w
            comps.append(c)
    return skel, comps
def mesmo_grupo(a, b):
    h = max(a["H"], b["H"])
    dist = np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
    ang = angulo_diff(a["ang"], b["ang"])
    vx, vy = np.cos(np.radians(a["ang"])), np.sin(np.radians(a["ang"]))
    alinh = abs((b["cx"] - a["cx"]) * vy - (b["cy"] - a["cy"]) * vx)
    perto = dist < min(0.06 * h + 0.26 * max(a["n"], b["n"]), 0.100 * h)
    return ang < 42 and perto and alinh < 0.022 * h and abs(a["cy"] - b["cy"]) < 0.12 * h
def caixa_grupo(g):
    pts = np.vstack([c["pts"] for c in g["itens"]]).astype(np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    return pts, x, y, w, h, x + w / 2, y + h / 2
def agrupar_componentes(comps):
    grupos = []
    for c in sorted(comps, key=lambda x: (x["cy"], x["cx"])):
        melhor, score = None, 1e9
        for g in grupos:
            for ref in g["itens"]:
                if mesmo_grupo(c, ref):
                    s = np.hypot(c["cx"] - ref["cx"], c["cy"] - ref["cy"])
                    if s < score:
                        melhor, score = g, s
        grupos.append({"itens": [c]}) if melhor is None else melhor["itens"].append(c)
    saida = []
    for g in grupos:
        _, x, y, w, h, cx, cy = caixa_grupo(g)
        dup = False
        for s in saida:
            _, x2, y2, w2, h2, cx2, cy2 = caixa_grupo(s)
            perto = np.hypot(cx - cx2, cy - cy2) < 0.032 * max(g["itens"][0]["H"], s["itens"][0]["H"])
            if perto and abs(w - w2) < 0.5 * max(w, w2) and abs(h - h2) < 0.5 * max(h, h2):
                s["itens"] += g["itens"]; dup = True; break
        if not dup:
            saida.append(g)
    return saida
def filtrar_caule_grupos(grupos, largura):
    caixas = []
    for g in grupos:
        _, x, y, w, h, cx, _ = caixa_grupo(g)
        caixas.append((g, x, y, w, h, cx))
    candidatos = [cx for _, x, _, w, h, cx in caixas if w < 0.035 * largura and h > 1.25 * w and 0.35 * largura < cx < 0.65 * largura]
    if not candidatos:
        return grupos
    stem_x = float(np.median(candidatos))
    folhas = []
    for g, x, y, w, h, cx in caixas:
        dens = h / max(1, w)
        estreito = w < 0.032 * largura and dens > 1.15 and abs(cx - stem_x) < 0.025 * largura
        fragmento = w < 0.038 * largura and h < 0.04 * largura and abs(cx - stem_x) < 0.03 * largura
        if not (estreito or fragmento):
            folhas.append(g)
    return folhas
def desenhar(img, grupos):
    out = img.copy()
    for i, g in enumerate(grupos, start=1):
        pts, x, y, w, h, _, _ = caixa_grupo(g)
        cor = (40 + (53 * i) % 215, 220 - (31 * i) % 160, 70 + (47 * i) % 185)
        cv2.rectangle(out, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(out, str(i), (x + 3, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
    cv2.putText(out, f"Folhas: {len(grupos)}", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    return out


