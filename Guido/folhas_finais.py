import argparse
import csv
import os
from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class ParametrosFolhas:
    corte_altura: float = 0.64
    hsv_min: tuple[int, int, int] = (18, 38, 38)
    hsv_max: tuple[int, int, int] = (96, 255, 255)
    blur_kernel: int = 5
    close_kernel: int = 7
    open_kernel: int = 3
    n_faixas: int = 26
    janela_eixo_px: float = 120.0
    min_comp_frac_h: float = 0.003
    caule_dist_frac_w: float = 0.014
    caule_dx_frac_w: float = 0.015
    caule_dy_dx: float = 1.25
    caule_vertical_max: float = 25.0
    caule_dy_frac_h: float = 0.018
    grupo_ang_max: float = 42.0
    grupo_dist_base_h: float = 0.060
    grupo_dist_n: float = 0.260
    grupo_dist_cap_h: float = 0.085
    grupo_alinh_frac_h: float = 0.022
    grupo_dy_frac_h: float = 0.120
    dup_dist_frac_h: float = 0.035
    stem_cand_w_frac: float = 0.035
    stem_cand_dens: float = 1.25
    stem_cand_x_min: float = 0.35
    stem_cand_x_max: float = 0.65
    stem_narrow_w_frac: float = 0.032
    stem_narrow_dens: float = 1.15
    stem_narrow_center_frac: float = 0.025
    stem_frag_w_frac: float = 0.018
    stem_frag_h_frac: float = 0.040
    stem_frag_center_frac: float = 0.030


PARAMS_ORIGINAIS = ParametrosFolhas()
PARAMS_OTIMIZADOS = ParametrosFolhas(
    caule_dx_frac_w=0.011,
    stem_narrow_w_frac=0.032,
    stem_frag_w_frac=0.038,
    dup_dist_frac_h=0.032,
    grupo_dist_cap_h=0.100,
)
PARAMS_GRID_MELHOR = PARAMS_OTIMIZADOS
PARAMS_GRID_CICLO1 = ParametrosFolhas(
    caule_dx_frac_w=0.010,
    stem_narrow_w_frac=0.032,
    stem_frag_w_frac=0.040,
    dup_dist_frac_h=0.035,
    grupo_dist_cap_h=0.100,
)
K8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

RANGES_GRID_INICIAL = {
    "caule_dx_frac_w": [0.008, 0.010],
    "stem_narrow_w_frac": [0.020, 0.025, 0.032],
    "stem_frag_w_frac": [0.032, 0.040],
    "dup_dist_frac_h": [0.035, 0.050],
    "grupo_dist_cap_h": [0.085, 0.100],
}

RANGES_GRID_REFINADO = {
    "caule_dx_frac_w": [0.009, 0.010, 0.011],
    "stem_narrow_w_frac": [0.030, 0.032, 0.034],
    "stem_frag_w_frac": [0.038, 0.040, 0.042],
    "dup_dist_frac_h": [0.032, 0.035, 0.050],
    "grupo_dist_cap_h": [0.095, 0.100, 0.105],
}


def segmentar(imagem_bgr, params=PARAMS_ORIGINAIS):
    h_corte = int(imagem_bgr.shape[0] * params.corte_altura)
    imagem = imagem_bgr[:h_corte, :]
    k_blur = params.blur_kernel
    hsv = cv2.cvtColor(cv2.GaussianBlur(imagem, (k_blur, k_blur), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(params.hsv_min, dtype=np.uint8),
        np.array(params.hsv_max, dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((params.close_kernel, params.close_kernel), np.uint8),
        iterations=1,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((params.open_kernel, params.open_kernel), np.uint8),
        iterations=1,
    )
    return imagem, mask


def angulo_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def x_caule(y, eixo_y, eixo_x):
    return np.interp(y, eixo_y, eixo_x)


def skeleton_da_mascara(mask):
    return skeletonize(mask > 0).astype(np.uint8)


def eixo_caule(skel, params=PARAMS_ORIGINAIS):
    h, w = skel.shape
    ys, xs = [], []
    cortes = np.linspace(0, h, params.n_faixas + 1, dtype=int)
    x_ant = w / 2.0
    for y0, y1 in zip(cortes[-2::-1], cortes[:0:-1]):
        _, faixa_x = np.where(skel[y0:y1] > 0)
        if len(faixa_x) > 4:
            perto = faixa_x[np.abs(faixa_x - x_ant) < params.janela_eixo_px]
            base = perto if len(perto) > 2 else faixa_x
            x_ant = float(np.median(base))
            ys.append((y0 + y1) / 2)
            xs.append(x_ant)
    if len(xs) < 2:
        return np.array([0.0, h - 1.0]), np.array([w / 2.0, w / 2.0])
    return np.array(ys[::-1]), np.array(xs[::-1])


def podar_skeleton(skel, n=1):
    podado = skel.copy()
    for _ in range(n):
        grau = cv2.filter2D(podado, -1, K8)
        podado[(podado > 0) & (grau == 1)] = 0
    return podado


def componente_info(pontos, eixo_y, eixo_x):
    pts = pontos.astype(np.float32)
    cx, cy = pts.mean(axis=0)
    cov = np.cov((pts - [cx, cy]).T)
    vals, vecs = np.linalg.eigh(cov)
    vx, vy = vecs[:, int(np.argmax(vals))]
    ang = (np.degrees(np.arctan2(vy, vx)) + 180.0) % 180.0
    dx, dy = np.ptp(pts[:, 0]) + 1, np.ptp(pts[:, 1]) + 1
    dist_eixo = np.mean(np.abs(pts[:, 0] - x_caule(pts[:, 1], eixo_y, eixo_x)))
    return {
        "pts": pts,
        "cx": cx,
        "cy": cy,
        "ang": ang,
        "n": len(pts),
        "dx": dx,
        "dy": dy,
        "dist": dist_eixo,
    }


def componentes_skeleton(skel, eixo_y, eixo_x, params=PARAMS_ORIGINAIS):
    h, w = skel.shape
    skel_podado = podar_skeleton(skel, 1)
    grau = cv2.filter2D(skel_podado, -1, K8)
    segmentos = np.uint8((skel_podado > 0) & (grau <= 2))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(segmentos, connectivity=8)
    comps = []

    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] < max(5, int(params.min_comp_frac_h * h)):
            continue
        ys, xs = np.where(labels == lab)
        c = componente_info(np.column_stack([xs, ys]), eixo_y, eixo_x)
        vertical = angulo_diff(c["ang"], 90.0) < params.caule_vertical_max
        central = c["dist"] < params.caule_dist_frac_w * w
        caule = (
            central
            and c["dx"] < params.caule_dx_frac_w * w
            and c["dy"] > params.caule_dy_dx * c["dx"]
            and (vertical or c["dy"] > params.caule_dy_frac_h * h)
        )
        if not caule:
            c["H"], c["W"] = h, w
            comps.append(c)
    return comps


def componentes_base_skeleton(skel, eixo_y, eixo_x):
    h, w = skel.shape
    skel_podado = podar_skeleton(skel, 1)
    grau = cv2.filter2D(skel_podado, -1, K8)
    segmentos = np.uint8((skel_podado > 0) & (grau <= 2))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(segmentos, connectivity=8)
    comps = []
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] < 5:
            continue
        ys, xs = np.where(labels == lab)
        c = componente_info(np.column_stack([xs, ys]), eixo_y, eixo_x)
        c["area_comp"] = int(stats[lab, cv2.CC_STAT_AREA])
        c["H"], c["W"] = h, w
        comps.append(c)
    return comps


def filtrar_componentes_base(comps_base, params=PARAMS_ORIGINAIS):
    comps = []
    for c in comps_base:
        h, w = c["H"], c["W"]
        if c["area_comp"] < max(5, int(params.min_comp_frac_h * h)):
            continue
        vertical = angulo_diff(c["ang"], 90.0) < params.caule_vertical_max
        central = c["dist"] < params.caule_dist_frac_w * w
        caule = (
            central
            and c["dx"] < params.caule_dx_frac_w * w
            and c["dy"] > params.caule_dy_dx * c["dx"]
            and (vertical or c["dy"] > params.caule_dy_frac_h * h)
        )
        if not caule:
            comps.append(c)
    return comps


def mesmo_grupo(a, b, params=PARAMS_ORIGINAIS):
    h = max(a["H"], b["H"])
    dist = np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
    ang = angulo_diff(a["ang"], b["ang"])
    vx, vy = np.cos(np.radians(a["ang"])), np.sin(np.radians(a["ang"]))
    alinh = abs((b["cx"] - a["cx"]) * vy - (b["cy"] - a["cy"]) * vx)
    perto = dist < min(
        params.grupo_dist_base_h * h + params.grupo_dist_n * max(a["n"], b["n"]),
        params.grupo_dist_cap_h * h,
    )
    return (
        ang < params.grupo_ang_max
        and perto
        and alinh < params.grupo_alinh_frac_h * h
        and abs(a["cy"] - b["cy"]) < params.grupo_dy_frac_h * h
    )


def caixa_grupo(grupo):
    pts = np.vstack([c["pts"] for c in grupo["itens"]]).astype(np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    return pts, x, y, w, h, x + w / 2, y + h / 2


def agrupar_componentes(comps, params=PARAMS_ORIGINAIS):
    grupos = []
    for c in sorted(comps, key=lambda x: (x["cy"], x["cx"])):
        melhor, score = None, 1e9
        for g in grupos:
            for ref in g["itens"]:
                if mesmo_grupo(c, ref, params):
                    s = np.hypot(c["cx"] - ref["cx"], c["cy"] - ref["cy"])
                    if s < score:
                        melhor, score = g, s
        if melhor is None:
            grupos.append({"itens": [c]})
        else:
            melhor["itens"].append(c)

    saida = []
    for g in grupos:
        _, x, y, w, h, cx, cy = caixa_grupo(g)
        duplicado = False
        for s in saida:
            _, x2, y2, w2, h2, cx2, cy2 = caixa_grupo(s)
            perto = np.hypot(cx - cx2, cy - cy2) < params.dup_dist_frac_h * max(
                g["itens"][0]["H"], s["itens"][0]["H"]
            )
            if perto and abs(w - w2) < 0.5 * max(w, w2) and abs(h - h2) < 0.5 * max(h, h2):
                s["itens"] += g["itens"]
                duplicado = True
                break
        if not duplicado:
            saida.append(g)
    return saida


def filtrar_caule_grupos(grupos, largura, params=PARAMS_ORIGINAIS):
    caixas = []
    for g in grupos:
        _, x, y, w, h, cx, _ = caixa_grupo(g)
        caixas.append((g, x, y, w, h, cx))

    candidatos = [
        cx
        for _, _, _, w, h, cx in caixas
        if w < params.stem_cand_w_frac * largura
        and h > params.stem_cand_dens * w
        and params.stem_cand_x_min * largura < cx < params.stem_cand_x_max * largura
    ]
    if not candidatos:
        return grupos

    stem_x = float(np.median(candidatos))
    folhas = []
    for g, _, _, w, h, cx in caixas:
        dens = h / max(1, w)
        estreito = (
            w < params.stem_narrow_w_frac * largura
            and dens > params.stem_narrow_dens
            and abs(cx - stem_x) < params.stem_narrow_center_frac * largura
        )
        fragmento = (
            w < params.stem_frag_w_frac * largura
            and h < params.stem_frag_h_frac * largura
            and abs(cx - stem_x) < params.stem_frag_center_frac * largura
        )
        if not (estreito or fragmento):
            folhas.append(g)
    return folhas


def desenhar(imagem, grupos):
    out = imagem.copy()
    for i, g in enumerate(grupos, start=1):
        _, x, y, w, h, _, _ = caixa_grupo(g)
        cor = (40 + (53 * i) % 215, 220 - (31 * i) % 160, 70 + (47 * i) % 185)
        cv2.rectangle(out, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(out, str(i), (x + 3, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
    cv2.putText(out, f"Folhas: {len(grupos)}", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    return out


def contar_folhas(imagem_bgr, params=PARAMS_ORIGINAIS, retornar_intermediarios=False):
    imagem, mask = segmentar(imagem_bgr, params)
    skel = skeleton_da_mascara(mask)
    eixo_y, eixo_x = eixo_caule(skel, params)
    comps = componentes_skeleton(skel, eixo_y, eixo_x, params)
    grupos = agrupar_componentes(comps, params)
    folhas = filtrar_caule_grupos(grupos, imagem.shape[1], params)
    resultado = {
        "numero_folhas": len(folhas),
        "folhas": folhas,
        "imagem_recortada": imagem,
        "anotacao": desenhar(imagem, folhas),
    }
    if retornar_intermediarios:
        resultado["mask"] = mask
        resultado["skeleton"] = skel
        resultado["eixo_y"] = eixo_y
        resultado["eixo_x"] = eixo_x
        resultado["componentes"] = comps
    return resultado


def preparar_para_grid(imagem_bgr, params=PARAMS_ORIGINAIS):
    imagem, mask = segmentar(imagem_bgr, params)
    skel = skeleton_da_mascara(mask)
    eixo_y, eixo_x = eixo_caule(skel, params)
    comps_base = componentes_base_skeleton(skel, eixo_y, eixo_x)
    return {
        "imagem_recortada": imagem,
        "mask": mask,
        "skeleton": skel,
        "eixo_y": eixo_y,
        "eixo_x": eixo_x,
        "componentes_base": comps_base,
    }


def contar_folhas_preprocessado(base, params=PARAMS_ORIGINAIS):
    comps = filtrar_componentes_base(base["componentes_base"], params)
    grupos = agrupar_componentes(comps, params)
    folhas = filtrar_caule_grupos(grupos, base["imagem_recortada"].shape[1], params)
    return {
        "numero_folhas": len(folhas),
        "folhas": folhas,
        "imagem_recortada": base["imagem_recortada"],
        "anotacao": desenhar(base["imagem_recortada"], folhas),
    }


def mape(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_real - y_pred) / y_real)) * 100.0)


def painel_anotacoes(anotacoes, altura_tile=320, colunas=5):
    tiles = []
    for img in anotacoes:
        escala = altura_tile / img.shape[0]
        tiles.append(cv2.resize(img, (int(img.shape[1] * escala), altura_tile)))
    linhas = []
    for i in range(0, len(tiles), colunas):
        linha = tiles[i : i + colunas]
        largura = max(tile.shape[1] for tile in linha)
        linha = [
            np.hstack([tile, np.full((tile.shape[0], largura - tile.shape[1], 3), 255, np.uint8)])
            if tile.shape[1] < largura
            else tile
            for tile in linha
        ]
        linhas.append(np.hstack(linha))
    largura_total = max(linha.shape[1] for linha in linhas)
    linhas = [
        np.hstack([linha, np.full((linha.shape[0], largura_total - linha.shape[1], 3), 255, np.uint8)])
        if linha.shape[1] < largura_total
        else linha
        for linha in linhas
    ]
    return np.vstack(linhas)


def aplicar_ajustes(params_base, ajustes):
    dados = params_base.__dict__.copy()
    dados.update(ajustes)
    return ParametrosFolhas(**dados)


def combinacoes_grid(ranges):
    chaves = list(ranges.keys())
    for valores in product(*(ranges[chave] for chave in chaves)):
        yield dict(zip(chaves, valores))


def assinatura_segmentacao(params):
    return (
        params.corte_altura,
        params.hsv_min,
        params.hsv_max,
        params.blur_kernel,
        params.close_kernel,
        params.open_kernel,
        params.n_faixas,
        params.janela_eixo_px,
    )


def grid_search(
    imagens,
    referencias,
    ranges,
    saida,
    params_base=PARAMS_ORIGINAIS,
    prefixo="grid",
    top_n=5,
    salvar_todos=True,
):
    os.makedirs(saida, exist_ok=True)
    cache = {}
    resultados = []
    for idx, ajustes in enumerate(combinacoes_grid(ranges), start=1):
        params = aplicar_ajustes(params_base, ajustes)
        sig = assinatura_segmentacao(params)
        if sig not in cache:
            cache[sig] = [preparar_para_grid(img, params) for img in imagens]

        predicoes = []
        anotacoes = []
        for base in cache[sig]:
            res = contar_folhas_preprocessado(base, params)
            predicoes.append(res["numero_folhas"])
            anotacoes.append(res["anotacao"])

        erro = mape(referencias, predicoes)
        caminho_img = os.path.join(saida, f"{prefixo}_{idx:04d}.png")
        if salvar_todos:
            cv2.imwrite(caminho_img, painel_anotacoes(anotacoes))

        linha = {
            "idx": idx,
            "MAPE": erro,
            "predicoes": predicoes,
            "imagem": caminho_img,
            **ajustes,
        }
        resultados.append(linha)

    resultados.sort(key=lambda item: item["MAPE"])
    csv_path = os.path.join(saida, f"{prefixo}_resultados.csv")
    chaves = sorted({chave for row in resultados for chave in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as arq:
        writer = csv.DictWriter(arq, fieldnames=chaves)
        writer.writeheader()
        writer.writerows(resultados)

    for rank, row in enumerate(resultados[:top_n], start=1):
        src = row["imagem"]
        dst = os.path.join(saida, f"{prefixo}_top{rank:02d}_idx{row['idx']:04d}_mape{row['MAPE']:.3f}.png")
        img = cv2.imread(src)
        if img is not None:
            cv2.imwrite(dst, img)
        row["imagem_top"] = dst
    return resultados, csv_path


def carregar_validacao(pasta_imagens, caminho_gabarito, total=10):
    import pandas as pd

    imagens = []
    for k in range(1, total + 1):
        caminho = os.path.join(pasta_imagens, f"Eucalipto{k}.jpg")
        img = cv2.imread(caminho, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Imagem nao encontrada: {caminho}")
        imagens.append(img)

    df = pd.read_excel(caminho_gabarito, header=1)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.dropna(axis=1, how="all")
    referencias = df["Nro Folhas"].astype(float).to_numpy()[:total]
    return imagens, referencias


def executar_otimizacao_validacao(
    pasta_imagens="D:/Visao/Projeto1/Conjunto_VALIDACAO",
    caminho_gabarito="Projeto1/Conjunto_VALIDACAO/Resultados_VALIDACAO.xlsx",
    saida="D:/Visao/Projeto1/Guido/grid_folhas_finais",
    top_n=5,
):
    imagens, referencias = carregar_validacao(pasta_imagens, caminho_gabarito)
    saida_inicial = os.path.join(saida, "ciclo1_inicial")
    resultados_inicial, csv_inicial = grid_search(
        imagens,
        referencias,
        RANGES_GRID_INICIAL,
        saida_inicial,
        params_base=PARAMS_ORIGINAIS,
        prefixo="grid_ciclo1",
        top_n=top_n,
        salvar_todos=True,
    )

    melhor_inicial = resultados_inicial[0]
    params_refino = aplicar_ajustes(
        PARAMS_ORIGINAIS,
        {chave: melhor_inicial[chave] for chave in RANGES_GRID_INICIAL.keys()},
    )
    saida_refino = os.path.join(saida, "ciclo2_refinado")
    resultados_refino, csv_refino = grid_search(
        imagens,
        referencias,
        RANGES_GRID_REFINADO,
        saida_refino,
        params_base=params_refino,
        prefixo="grid_ciclo2",
        top_n=top_n,
        salvar_todos=True,
    )
    return {
        "ciclo1": resultados_inicial,
        "csv_ciclo1": csv_inicial,
        "ciclo2": resultados_refino,
        "csv_ciclo2": csv_refino,
    }


def contar_arquivo(caminho_imagem, params=PARAMS_ORIGINAIS, salvar_anotacao=None):
    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_COLOR)
    if imagem is None:
        raise FileNotFoundError(f"Imagem nao encontrada: {caminho_imagem}")
    resultado = contar_folhas(imagem, params=params, retornar_intermediarios=salvar_anotacao is not None)
    if salvar_anotacao:
        cv2.imwrite(salvar_anotacao, resultado["anotacao"])
    return resultado


def main():
    parser = argparse.ArgumentParser(description="Conta folhas em uma imagem de eucalipto.")
    parser.add_argument("imagem", nargs="?", help="Caminho da imagem de entrada.")
    parser.add_argument("--saida", help="Caminho opcional para salvar a imagem anotada.")
    parser.add_argument("--otimizado", action="store_true", help="Usa o conjunto parametrico otimizado.")
    parser.add_argument("--grid", action="store_true", help="Executa grid search em duas etapas na validacao.")
    parser.add_argument("--pasta-imagens", default="D:/Visao/Projeto1/Conjunto_VALIDACAO")
    parser.add_argument("--gabarito", default="Projeto1/Conjunto_VALIDACAO/Resultados_VALIDACAO.xlsx")
    parser.add_argument("--saida-grid", default="D:/Visao/Projeto1/Guido/grid_folhas_finais")
    args = parser.parse_args()

    if args.grid:
        resultado = executar_otimizacao_validacao(args.pasta_imagens, args.gabarito, args.saida_grid)
        print("Ciclo 1:")
        for row in resultado["ciclo1"][:5]:
            print(row)
        print("Ciclo 2:")
        for row in resultado["ciclo2"][:5]:
            print(row)
        return

    if not args.imagem:
        parser.error("informe uma imagem ou use --grid")

    params = PARAMS_OTIMIZADOS if args.otimizado else PARAMS_ORIGINAIS
    resultado = contar_arquivo(args.imagem, params=params, salvar_anotacao=args.saida)
    print(resultado["numero_folhas"])


if __name__ == "__main__":
    main()
