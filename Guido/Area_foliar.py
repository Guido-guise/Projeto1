import cv2
import numpy as np
from itertools import product


CAMINHO_IMAGEM = "Projeto1/_Eucalipto_Escolhidos2/Eucalipto{}.jpg"
VALORES_REFERENCIA = np.array([62484, 217423, 179931, 43952, 28524], dtype=float)

# Grade pequena: suficiente para este cenario controlado, sem virar ML.
GRADE = {
    "corte": [0.63, 0.64, 0.65, 0.67, 0.69],
    "h_min": [18, 20],
    "h_max": [95, 96, 97, 98],
    "s_min": [36, 38, 40],
    "v_min": [38, 40],
    "area_min": [0, 5, 10, 15, 25],
    "metodo": ["hsv", "hsv_azul", "lab"],
}

ESCALA_CALIBRACAO = 0.25
TOP_CANDIDATOS = 20
MORFOLOGIA = False  # fechamento 3x3 opcional; a calibracao mostrou aumento de erro.


def carregar_imagens():
    imagens = []
    for i in range(1, len(VALORES_REFERENCIA) + 1):
        caminho = CAMINHO_IMAGEM.format(i)
        img = cv2.imread(caminho, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Imagem nao encontrada: {caminho}")
        imagens.append(img)
    return imagens


def erro_relativo_medio(areas, referencias):
    areas = np.array(areas, dtype=float)
    return float(np.mean(np.abs((areas - referencias) / referencias) * 100.0))


def limpar_componentes(mask, area_minima):
    if area_minima <= 0:
        return mask

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    limpa = np.zeros_like(mask)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= area_minima:
            limpa[labels == label] = 255
    return limpa


def mascara_fundo_lab_adaptativa(img_bgr):
    """Fundo por LAB usando media das bordas e threshold por percentil/desvio."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    by, bx = max(1, int(h * 0.06)), max(1, int(w * 0.06))

    borda = np.zeros((h, w), dtype=bool)
    borda[:by, :] = True
    borda[-by:, :] = True
    borda[:, :bx] = True
    borda[:, -bx:] = True

    cor_fundo = lab[borda].mean(axis=0)
    dist = np.linalg.norm(lab - cor_fundo, axis=2)
    dist_borda = np.linalg.norm(lab[borda] - cor_fundo, axis=1)
    limiar = np.clip(np.percentile(dist_borda, 90) + 2.5 * np.std(dist_borda), 12, 40)

    return np.where(dist <= limiar, 255, 0).astype(np.uint8), float(limiar)


def segmentar(img_bgr, params, escala_area=1.0):
    """
    Fluxo direto:
    corte fixo -> HSV folhas -> opcionalmente remover fundo -> limpeza leve.
    """
    h = img_bgr.shape[0]
    recorte = img_bgr[: int(h * params["corte"]), :]
    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)

    lower = np.array([params["h_min"], params["s_min"], params["v_min"]], dtype=np.uint8)
    upper = np.array([params["h_max"], 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    if params["metodo"] == "hsv_azul":
        fundo = cv2.inRange(hsv, np.array([95, 80, 80], dtype=np.uint8), np.array([125, 255, 255], dtype=np.uint8))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(fundo))
    elif params["metodo"] == "lab":
        fundo, _ = mascara_fundo_lab_adaptativa(recorte)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(fundo))

    if MORFOLOGIA:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    area_min = max(0, int(round(params["area_min"] * escala_area)))
    mask = limpar_componentes(mask, area_min)
    return mask, int(np.count_nonzero(mask))


def avaliar(imagens, referencias, params, escala_area=1.0):
    areas = [segmentar(img, params, escala_area=escala_area)[1] for img in imagens]
    return erro_relativo_medio(areas, referencias), areas


def candidatos_hsv_reduzidos(imagens, referencias):
    """Primeira fase em baixa resolucao: acha bons cortes e intervalos HSV."""
    imagens_menores = [
        cv2.resize(img, None, fx=ESCALA_CALIBRACAO, fy=ESCALA_CALIBRACAO, interpolation=cv2.INTER_AREA)
        for img in imagens
    ]
    refs_menores = referencias * (ESCALA_CALIBRACAO ** 2)

    resultados = []
    for corte, h_min, h_max, s_min, v_min in product(
        GRADE["corte"], GRADE["h_min"], GRADE["h_max"], GRADE["s_min"], GRADE["v_min"]
    ):
        params = {
            "corte": corte,
            "h_min": h_min,
            "h_max": h_max,
            "s_min": s_min,
            "v_min": v_min,
            "area_min": 0,
            "metodo": "hsv",
        }
        erro, _ = avaliar(imagens_menores, refs_menores, params, ESCALA_CALIBRACAO ** 2)
        resultados.append((erro, params))

    resultados.sort(key=lambda item: item[0])
    return [params for _, params in resultados[:TOP_CANDIDATOS]]


def calibrar_parametros(imagens, referencias):
    """
    Segunda fase em resolucao original:
    refina a area minima dos melhores HSV.
    LAB e HSV-azul ficam para comparacao, pois nao melhoraram o erro neste conjunto.
    """
    melhores_hsv = candidatos_hsv_reduzidos(imagens, referencias)
    resultados = []

    for base, area_min in product(melhores_hsv, GRADE["area_min"]):
        params = dict(base)
        params["metodo"] = "hsv"
        params["area_min"] = area_min
        erro, areas = avaliar(imagens, referencias, params)
        resultados.append((erro, params, areas))

    resultados.sort(key=lambda item: item[0])
    return resultados[0], resultados


def avaliar_impacto(imagens, referencias, melhor):
    """Varia um parametro por vez para medir seu impacto no erro final."""
    print("\nImpacto dos parametros (variando um por vez):")
    for nome, valores in GRADE.items():
        linhas = []
        for valor in valores:
            params = dict(melhor)
            params[nome] = valor
            erro, _ = avaliar(imagens, referencias, params)
            linhas.append((erro, valor))
        linhas.sort(key=lambda item: item[0])
        melhor_erro, melhor_valor = linhas[0]
        pior_erro, pior_valor = linhas[-1]
        print(
            f"- {nome}: melhor={melhor_valor} ({melhor_erro:.2f}%), "
            f"pior={pior_valor} ({pior_erro:.2f}%), impacto={pior_erro - melhor_erro:.2f} p.p."
        )


def imprimir_resultados(imagens, referencias, params):
    erros = []
    print("\nResultado final:")
    for i, (img, ref) in enumerate(zip(imagens, referencias), start=1):
        _, area = segmentar(img, params)
        erro = abs((area - ref) / ref) * 100.0
        erros.append(erro)
        print(f"Eucalipto{i}: area={area} px | referencia={int(ref)} px | erro={erro:.2f}%")
    print(f"Erro medio final: {np.mean(erros):.2f}%")


def main():
    imagens = carregar_imagens()
    (erro, melhores_params, _), _ = calibrar_parametros(imagens, VALORES_REFERENCIA)

    print("Parametros calibrados automaticamente:")
    for chave, valor in melhores_params.items():
        print(f"- {chave}: {valor}")
    print(f"Erro medio na calibracao: {erro:.2f}%")

    # Comparacao direta dos metodos de fundo usando os mesmos limiares HSV.
    print("\nComparacao de metodo de fundo:")
    for metodo in GRADE["metodo"]:
        params = dict(melhores_params)
        params["metodo"] = metodo
        erro_metodo, _ = avaliar(imagens, VALORES_REFERENCIA, params)
        print(f"- {metodo}: erro medio={erro_metodo:.2f}%")

    imprimir_resultados(imagens, VALORES_REFERENCIA, melhores_params)
    avaliar_impacto(imagens, VALORES_REFERENCIA, melhores_params)


if __name__ == "__main__":
    main()
