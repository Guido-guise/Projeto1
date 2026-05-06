import os

import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

from Final import K8, angulo_diff, caixa_grupo, componente_info, desenhar, x_caule


CAMINHO_IMAGEM = "D:/Visao/Projeto1/Conjunto_VALIDACAO/Eucalipto{}.jpg"
GABARITO = "Projeto1/Conjunto_VALIDACAO/Resultados_VALIDACAO.xlsx"
SAIDA_BASE = "D:/Visao/Projeto1/Guido/otimizacao_folhas"
IMAGENS_CACHE = None
REFERENCIAS_CACHE = None


PARAMS_BASE = {
    "corte_altura": 0.64,
    "h_min": 18,
    "h_max": 96,
    "s_min": 38,
    "v_min": 38,
    "close_kernel": 7,
    "open_kernel": 3,
    "open_iter": 1,
    "n_faixas": 26,
    "eixo_janela": 120,
    "min_comp_frac_h": 0.003,
    "caule_vertical_max": 25.0,
    "caule_dist_frac_w": 0.014,
    "caule_dx_frac_w": 0.015,
    "caule_dy_dx": 1.25,
    "caule_dy_frac_h": 0.018,
    "grupo_ang_max": 42.0,
    "grupo_dist_base_h": 0.060,
    "grupo_dist_n": 0.260,
    "grupo_dist_cap_h": 0.085,
    "grupo_alinh_frac_h": 0.022,
    "grupo_dy_frac_h": 0.120,
    "dup_dist_frac_h": 0.035,
    "stem_cand_w_frac": 0.035,
    "stem_cand_dens": 1.25,
    "stem_cand_x_min": 0.35,
    "stem_cand_x_max": 0.65,
    "stem_narrow_w_frac": 0.032,
    "stem_narrow_dens": 1.15,
    "stem_narrow_center_frac": 0.025,
    "stem_frag_w_frac": 0.018,
    "stem_frag_h_frac": 0.040,
    "stem_frag_center_frac": 0.030,
}


def carregar_referencias():
    global REFERENCIAS_CACHE
    if REFERENCIAS_CACHE is not None:
        return REFERENCIAS_CACHE
    df = pd.read_excel(GABARITO, header=1)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.dropna(axis=1, how="all")
    REFERENCIAS_CACHE = df["Nro Folhas"].astype(float).to_numpy()
    return REFERENCIAS_CACHE


def carregar_imagens():
    global IMAGENS_CACHE
    if IMAGENS_CACHE is not None:
        return IMAGENS_CACHE
    imagens = []
    for k in range(1, 11):
        img = cv2.imread(CAMINHO_IMAGEM.format(k), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(CAMINHO_IMAGEM.format(k))
        imagens.append(img)
    IMAGENS_CACHE = imagens
    return IMAGENS_CACHE


def segmentar_param(img, p):
    img = img[: int(img.shape[0] * p["corte_altura"]), :]
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    lower = np.array([p["h_min"], p["s_min"], p["v_min"]], dtype=np.uint8)
    upper = np.array([p["h_max"], 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    k_close = np.ones((p["close_kernel"], p["close_kernel"]), np.uint8)
    k_open = np.ones((p["open_kernel"], p["open_kernel"]), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=p["open_iter"])
    return img, mask


def eixo_caule_param(mask, p):
    h, w = mask.shape
    ys, xs = [], []
    cortes = np.linspace(0, h, p["n_faixas"] + 1, dtype=int)
    skel = skeletonize(mask > 0).astype(np.uint8)
    x_ant = w / 2.0
    for y0, y1 in zip(cortes[-2::-1], cortes[:0:-1]):
        _, faixa_x = np.where(skel[y0:y1] > 0)
        if len(faixa_x) > 4:
            perto = faixa_x[np.abs(faixa_x - x_ant) < p["eixo_janela"]]
            base = perto if len(perto) > 2 else faixa_x
            x_ant = float(np.median(base))
            ys.append((y0 + y1) / 2)
            xs.append(x_ant)
    if len(xs) < 2:
        return np.array([0.0, h - 1.0]), np.array([w / 2.0, w / 2.0])
    return np.array(ys[::-1]), np.array(xs[::-1])


def podar_skeleton_param(skel, n=1):
    skel = skel.copy()
    for _ in range(n):
        grau = cv2.filter2D(skel, -1, K8)
        skel[(skel > 0) & (grau == 1)] = 0
    return skel


def componentes_skeleton_param(mask, eixo_y, eixo_x, p):
    h, w = mask.shape
    skel = podar_skeleton_param(skeletonize(mask > 0).astype(np.uint8), 1)
    grau = cv2.filter2D(skel, -1, K8)
    segmentos = np.uint8((skel > 0) & (grau <= 2))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(segmentos, connectivity=8)
    comps = []
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] < max(5, int(p["min_comp_frac_h"] * h)):
            continue
        ys, xs = np.where(labels == lab)
        c = componente_info(np.column_stack([xs, ys]), eixo_y, eixo_x)
        vertical = angulo_diff(c["ang"], 90.0) < p["caule_vertical_max"]
        central = c["dist"] < p["caule_dist_frac_w"] * w
        caule = (
            central
            and c["dx"] < p["caule_dx_frac_w"] * w
            and c["dy"] > p["caule_dy_dx"] * c["dx"]
            and (vertical or c["dy"] > p["caule_dy_frac_h"] * h)
        )
        if not caule:
            c["H"], c["W"] = h, w
            comps.append(c)
    return skel, comps


def mesmo_grupo_param(a, b, p):
    h = max(a["H"], b["H"])
    dist = np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
    ang = angulo_diff(a["ang"], b["ang"])
    vx, vy = np.cos(np.radians(a["ang"])), np.sin(np.radians(a["ang"]))
    alinh = abs((b["cx"] - a["cx"]) * vy - (b["cy"] - a["cy"]) * vx)
    perto = dist < min(
        p["grupo_dist_base_h"] * h + p["grupo_dist_n"] * max(a["n"], b["n"]),
        p["grupo_dist_cap_h"] * h,
    )
    return (
        ang < p["grupo_ang_max"]
        and perto
        and alinh < p["grupo_alinh_frac_h"] * h
        and abs(a["cy"] - b["cy"]) < p["grupo_dy_frac_h"] * h
    )


def agrupar_componentes_param(comps, p):
    grupos = []
    for c in sorted(comps, key=lambda x: (x["cy"], x["cx"])):
        melhor, score = None, 1e9
        for g in grupos:
            for ref in g["itens"]:
                if mesmo_grupo_param(c, ref, p):
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
            perto = np.hypot(cx - cx2, cy - cy2) < p["dup_dist_frac_h"] * max(
                g["itens"][0]["H"], s["itens"][0]["H"]
            )
            if perto and abs(w - w2) < 0.5 * max(w, w2) and abs(h - h2) < 0.5 * max(h, h2):
                s["itens"] += g["itens"]
                dup = True
                break
        if not dup:
            saida.append(g)
    return saida


def filtrar_caule_grupos_param(grupos, largura, p):
    caixas = []
    for g in grupos:
        _, x, y, w, h, cx, _ = caixa_grupo(g)
        caixas.append((g, x, y, w, h, cx))
    candidatos = [
        cx
        for _, x, _, w, h, cx in caixas
        if w < p["stem_cand_w_frac"] * largura
        and h > p["stem_cand_dens"] * w
        and p["stem_cand_x_min"] * largura < cx < p["stem_cand_x_max"] * largura
    ]
    if not candidatos:
        return grupos
    stem_x = float(np.median(candidatos))
    folhas = []
    for g, x, y, w, h, cx in caixas:
        dens = h / max(1, w)
        estreito = (
            w < p["stem_narrow_w_frac"] * largura
            and dens > p["stem_narrow_dens"]
            and abs(cx - stem_x) < p["stem_narrow_center_frac"] * largura
        )
        fragmento = (
            w < p["stem_frag_w_frac"] * largura
            and h < p["stem_frag_h_frac"] * largura
            and abs(cx - stem_x) < p["stem_frag_center_frac"] * largura
        )
        if not (estreito or fragmento):
            folhas.append(g)
    return folhas


def contar_folhas(img, p):
    img10, mask = segmentar_param(img, p)
    eixo_y, eixo_x = eixo_caule_param(mask, p)
    _, comps = componentes_skeleton_param(mask, eixo_y, eixo_x, p)
    folhas = filtrar_caule_grupos_param(agrupar_componentes_param(comps, p), img10.shape[1], p)
    return img10, folhas


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def salvar_painel(caminhos, destino):
    imgs = []
    for caminho in caminhos:
        img = cv2.imread(caminho)
        if img is None:
            continue
        escala = 360 / img.shape[0]
        imgs.append(cv2.resize(img, (int(img.shape[1] * escala), 360)))
    if not imgs:
        return
    rows = []
    for i in range(0, len(imgs), 5):
        row = imgs[i : i + 5]
        rows.append(np.hstack(row))
    largura = max(r.shape[1] for r in rows)
    rows = [
        np.hstack([r, np.full((r.shape[0], largura - r.shape[1], 3), 255, np.uint8)])
        if r.shape[1] < largura
        else r
        for r in rows
    ]
    cv2.imwrite(destino, np.vstack(rows))


def avaliar_parametros(p, nome, salvar_imagens=False):
    y_true = carregar_referencias()
    y_pred, caminhos = [], []
    pasta = os.path.join(SAIDA_BASE, nome)
    if salvar_imagens:
        os.makedirs(pasta, exist_ok=True)
    for k, img in enumerate(carregar_imagens(), start=1):
        img10, folhas = contar_folhas(img, p)
        y_pred.append(len(folhas))
        if salvar_imagens:
            caminho = os.path.join(pasta, f"eucalipto{k}_contagem.png")
            cv2.imwrite(caminho, desenhar(img10, folhas))
            caminhos.append(caminho)
    erro = mape(y_true, y_pred)
    if salvar_imagens:
        pd.DataFrame({"Img": range(1, 11), "Nro Folhas ref": y_true, "Nro Folhas": y_pred}).to_csv(
            os.path.join(pasta, "previsoes.csv"), index=False
        )
        salvar_painel(caminhos, os.path.join(pasta, "painel_contagem.png"))
    return erro, y_pred, pasta


def grade_inicial():
    ajustes = [
        {},
        {"stem_frag_w_frac": 0.010, "stem_frag_h_frac": 0.030},
        {"grupo_ang_max": 34.0, "grupo_dist_cap_h": 0.070},
        {"h_min": 16, "s_min": 34, "min_comp_frac_h": 0.0022},
        {
            "h_min": 16,
            "s_min": 34,
            "min_comp_frac_h": 0.0022,
            "stem_frag_w_frac": 0.010,
            "stem_frag_h_frac": 0.030,
            "grupo_ang_max": 34.0,
            "grupo_dist_cap_h": 0.070,
        },
        {"stem_frag_w_frac": 0.022, "stem_frag_h_frac": 0.050},
        {"grupo_ang_max": 50.0, "grupo_dist_cap_h": 0.100},
        {"h_min": 20, "s_min": 42, "min_comp_frac_h": 0.0040},
        {
            "stem_frag_w_frac": 0.014,
            "stem_frag_h_frac": 0.030,
            "grupo_ang_max": 34.0,
            "grupo_dist_cap_h": 0.070,
        },
        {
            "h_min": 16,
            "s_min": 34,
            "min_comp_frac_h": 0.0022,
            "stem_frag_w_frac": 0.010,
            "stem_frag_h_frac": 0.030,
        },
        {"caule_dist_frac_w": 0.008},
        {"caule_dx_frac_w": 0.010},
        {"caule_dist_frac_w": 0.008, "caule_dx_frac_w": 0.010},
        {
            "stem_narrow_w_frac": 0.020,
            "stem_narrow_center_frac": 0.015,
            "stem_frag_w_frac": 0.010,
            "stem_frag_center_frac": 0.015,
        },
        {
            "caule_dist_frac_w": 0.008,
            "caule_dx_frac_w": 0.010,
            "stem_narrow_w_frac": 0.020,
            "stem_narrow_center_frac": 0.015,
            "stem_frag_w_frac": 0.010,
            "stem_frag_center_frac": 0.015,
        },
        {
            "h_min": 20,
            "s_min": 42,
            "min_comp_frac_h": 0.0040,
            "caule_dist_frac_w": 0.008,
            "caule_dx_frac_w": 0.010,
            "stem_narrow_w_frac": 0.020,
            "stem_narrow_center_frac": 0.015,
            "stem_frag_w_frac": 0.010,
            "stem_frag_center_frac": 0.015,
        },
        {"dup_dist_frac_h": 0.060},
        {"grupo_alinh_frac_h": 0.035, "dup_dist_frac_h": 0.060},
    ]
    for ajuste in ajustes:
        p = dict(PARAMS_BASE)
        p.update(ajuste)
        yield p


def nome_parametros(p):
    return (
        f"h{p['h_min']}_s{p['s_min']}_op{p['open_kernel']}_min{p['min_comp_frac_h']:.4f}_"
        f"cdx{p['caule_dx_frac_w']:.3f}_dup{p['dup_dist_frac_h']:.3f}_"
        f"fw{p['stem_frag_w_frac']:.3f}_fh{p['stem_frag_h_frac']:.3f}_"
        f"ga{p['grupo_ang_max']:.0f}_gc{p['grupo_dist_cap_h']:.3f}"
    )


def otimizar_parametros(limite_salvar=8):
    os.makedirs(SAIDA_BASE, exist_ok=True)
    resultados = []
    for idx, p in enumerate(grade_inicial(), start=1):
        nome = nome_parametros(p)
        erro, pred, _ = avaliar_parametros(p, nome, salvar_imagens=False)
        resultados.append({**p, "MAPE Nro Folhas": erro, "predicoes": pred, "imagens": ""})
        if idx % 200 == 0:
            print(f"{idx} combinacoes testadas; melhor MAPE={min(r['MAPE Nro Folhas'] for r in resultados):.3f}%")

    resultados.sort(key=lambda r: r["MAPE Nro Folhas"])
    for i, r in enumerate(resultados[:limite_salvar], start=1):
        p = {k: r[k] for k in PARAMS_BASE}
        nome = f"rank{i:02d}_" + nome_parametros(p)
        erro, pred, pasta = avaliar_parametros(p, nome, salvar_imagens=True)
        r["MAPE Nro Folhas"] = erro
        r["predicoes"] = pred
        r["imagens"] = pasta

    df = pd.DataFrame(resultados)
    df.to_csv(os.path.join(SAIDA_BASE, "resultados_otimizacao.csv"), index=False)
    print(df.head(20)[["MAPE Nro Folhas", "predicoes", "imagens"]].to_string(index=False))
    print("\nMelhor conjunto:")
    melhor = resultados[0]
    for k in PARAMS_BASE:
        print(f"{k}: {melhor[k]}")


if __name__ == "__main__":
    otimizar_parametros()
