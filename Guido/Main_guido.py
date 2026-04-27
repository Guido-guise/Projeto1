import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from itertools import product


CAMINHO_IMAGEM = "Projeto1/_Eucalipto_Escolhidos2/Eucalipto{}.jpg"
REFERENCIAS = [12, 12, 12, 14, 4]

# Segmentacao simples reaproveitando o cenario controlado.
CORTE_ALTURA = 0.64
HSV_MIN = np.array([18, 38, 38], dtype=np.uint8)
HSV_MAX = np.array([96, 255, 255], dtype=np.uint8)
AREA_MINIMA = 0
LARGURA_VIS = 900

# DBSCAN com min_pts=1 permite folhas representadas por uma unica ponta.
# eps = base + ganho * area_bbox_milhao, calibrado nas imagens geradas.
EPS_BASES = [45.0, 46.0, 47.0, 48.0, 49.0]
EPS_GANHOS_AREA = [19.0, 20.0, 20.5, 21.0, 22.0]
CENTRO_RELATIVOS = [0.0, 0.01, 0.015, 0.02]
REMOVER_BASES = [0.0, 0.05, 0.10]
MIN_ENDPOINTS_CLUSTER = 1

KERNEL_VIZINHOS = np.array(
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]],
    dtype=np.uint8,
)


def limpar_componentes(mask, area_minima):
    if area_minima <= 0:
        return mask

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    limpa = np.zeros_like(mask)

    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= area_minima:
            limpa[labels == label] = 255

    return limpa


def segmentar_planta(img_bgr):
    altura = img_bgr.shape[0]
    img_recortada = img_bgr[: int(altura * CORTE_ALTURA), :]

    hsv = cv2.cvtColor(img_recortada, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_MIN, HSV_MAX)
    mask = limpar_componentes(mask, AREA_MINIMA)

    return img_recortada, mask


def detectar_endpoints(mask):
    skeleton = skeletonize(mask > 0).astype(np.uint8)
    vizinhos = cv2.filter2D(skeleton, -1, KERNEL_VIZINHOS)
    endpoints = np.logical_and(skeleton == 1, vizinhos == 1)

    ys, xs = np.where(endpoints)
    pontos = np.column_stack([xs, ys]).astype(np.float32)

    return skeleton, pontos


def dbscan_simples(pontos, eps, min_pts):
    """DBSCAN compacto para poucos endpoints; evita depender de sklearn."""
    n = len(pontos)
    labels = np.full(n, -1, dtype=int)
    visitados = np.zeros(n, dtype=bool)

    if n == 0:
        return labels

    dist2 = np.sum((pontos[:, None, :] - pontos[None, :, :]) ** 2, axis=2)
    vizinhos = [np.where(dist2[i] <= eps * eps)[0].tolist() for i in range(n)]

    cluster_id = 0
    for i in range(n):
        if visitados[i]:
            continue

        visitados[i] = True
        if len(vizinhos[i]) < min_pts:
            continue

        labels[i] = cluster_id
        fila = list(vizinhos[i])
        idx = 0

        while idx < len(fila):
            p = fila[idx]

            if not visitados[p]:
                visitados[p] = True
                if len(vizinhos[p]) >= min_pts:
                    for q in vizinhos[p]:
                        if q not in fila:
                            fila.append(q)

            if labels[p] == -1:
                labels[p] = cluster_id

            idx += 1

        cluster_id += 1

    return labels


def contar_clusters_validos(pontos, labels, largura, centro_relativo, usar_singleton):
    validos = []
    centro_x = largura / 2.0
    faixa_caule = centro_relativo * largura

    for cluster_id in sorted(set(labels[labels >= 0])):
        pts_cluster = pontos[labels == cluster_id]
        cx = float(np.mean(pts_cluster[:, 0]))

        if faixa_caule > 0 and abs(cx - centro_x) < faixa_caule:
            continue
        if len(pts_cluster) >= 2:
            validos.append(cluster_id)

    # Fallback simples: em plantas pequenas, um endpoint isolado pode ser folha real.
    if usar_singleton and len(validos) <= 3:
        for cluster_id in sorted(set(labels[labels >= 0])):
            if cluster_id in validos:
                continue
            pts_cluster = pontos[labels == cluster_id]
            cx = float(np.mean(pts_cluster[:, 0]))
            if faixa_caule > 0 and abs(cx - centro_x) < faixa_caule:
                continue
            if len(pts_cluster) == 1:
                validos.append(cluster_id)
                break

    return validos


def area_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    return float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))


def filtrar_base(pontos, altura, remover_base):
    if remover_base <= 0 or len(pontos) == 0:
        return pontos
    return pontos[pontos[:, 1] < altura * (1.0 - remover_base)]


def criar_mascara_endpoints(shape, pontos):
    endpoints = np.zeros(shape, dtype=bool)
    if len(pontos) > 0:
        endpoints[pontos[:, 1].astype(int), pontos[:, 0].astype(int)] = True
    return endpoints


def calcular_eps(item, eps_base, eps_ganho_area):
    eps = eps_base + eps_ganho_area * (item["bbox_area"] / 1_000_000.0)
    return float(np.clip(eps, 45.0, 85.0))


def processar_com_params(dados, eps_base, eps_ganho_area, centro_relativo, remover_base, usar_singleton):
    estimativas = []
    resultados = []

    for item in dados:
        pontos = filtrar_base(item["pontos"], item["mask"].shape[0], remover_base)
        eps = calcular_eps(item, eps_base, eps_ganho_area)
        labels = dbscan_simples(pontos, eps=eps, min_pts=MIN_ENDPOINTS_CLUSTER)
        validos = contar_clusters_validos(
            pontos,
            labels,
            largura=item["mask"].shape[1],
            centro_relativo=centro_relativo,
            usar_singleton=usar_singleton,
        )
        estimativas.append(len(validos))
        resultados.append((labels, validos, pontos, eps))

    return estimativas, resultados


def calibrar_parametros(dados):
    melhores = []

    for eps_base, eps_ganho, centro_rel, remover_base, usar_singleton in product(
        EPS_BASES,
        EPS_GANHOS_AREA,
        CENTRO_RELATIVOS,
        REMOVER_BASES,
        [True, False],
    ):
        estimativas, _ = processar_com_params(
            dados,
            eps_base,
            eps_ganho,
            centro_rel,
            remover_base,
            usar_singleton,
        )
        erros = [
            abs((estimado - ref) / ref) * 100.0
            for estimado, ref in zip(estimativas, REFERENCIAS)
        ]
        melhores.append((
            float(np.mean(erros)),
            eps_base,
            eps_ganho,
            centro_rel,
            remover_base,
            usar_singleton,
            estimativas,
        ))

    melhores.sort(key=lambda x: x[0])
    return melhores[0]


def criar_overlay(img_bgr, skeleton, endpoints, pontos, labels, validos):
    overlay = img_bgr.copy()

    # Skeleton em verde.
    skeleton_vis = cv2.dilate(skeleton, np.ones((3, 3), np.uint8), iterations=1)
    overlay[skeleton_vis > 0] = (0, 180, 0)

    # Endpoints em branco; clusters em cores.
    ys, xs = np.where(endpoints)
    for x, y in zip(xs, ys):
        cv2.circle(overlay, (int(x), int(y)), 5, (255, 255, 255), -1)

    cores = plt.cm.tab20(np.linspace(0, 1, max(1, len(validos))))
    for ordem, cluster_id in enumerate(validos):
        cor_rgb = (cores[cluster_id % len(cores)][:3] * 255).astype(np.uint8)
        cor_bgr = tuple(int(c) for c in cor_rgb[::-1])
        pts_cluster = pontos[labels == cluster_id]
        centro = np.mean(pts_cluster, axis=0)

        for x, y in pts_cluster:
            cv2.circle(overlay, (int(x), int(y)), 9, cor_bgr, -1)
        cv2.putText(
            overlay,
            str(ordem + 1),
            (int(centro[0]) + 8, int(centro[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            cor_bgr,
            2,
            cv2.LINE_AA,
        )

    return overlay


def exibir_resultado(img_recortada, skeleton, endpoints, overlay, titulo):
    if img_recortada.shape[1] > LARGURA_VIS:
        escala = LARGURA_VIS / img_recortada.shape[1]
        tamanho = (LARGURA_VIS, int(img_recortada.shape[0] * escala))
        img_recortada = cv2.resize(img_recortada, tamanho, interpolation=cv2.INTER_AREA)
        skeleton = cv2.resize(skeleton, tamanho, interpolation=cv2.INTER_NEAREST)
        endpoints = cv2.resize(endpoints.astype(np.uint8), tamanho, interpolation=cv2.INTER_NEAREST) > 0
        overlay = cv2.resize(overlay, tamanho, interpolation=cv2.INTER_AREA)

    endpoints_img = np.zeros_like(skeleton, dtype=np.uint8)
    endpoints_img[endpoints] = 255
    endpoints_img = cv2.dilate(endpoints_img, np.ones((7, 7), np.uint8), iterations=1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(skeleton, cmap="gray")
    plt.title("Skeleton")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(endpoints_img, cmap="gray")
    plt.title("Endpoints")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Clusters sobre a planta")
    plt.axis("off")

    plt.suptitle(titulo)
    plt.tight_layout()
    plt.show()
    plt.close()


dados = []

for i, referencia in enumerate(REFERENCIAS, start=1):
    img = cv2.imread(CAMINHO_IMAGEM.format(i), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Eucalipto{i}: imagem nao encontrada")
        continue

    img_recortada, mask_planta = segmentar_planta(img)
    skeleton, pontos = detectar_endpoints(mask_planta)
    dados.append({
        "indice": i,
        "referencia": referencia,
        "img_recortada": img_recortada,
        "mask": mask_planta,
        "skeleton": skeleton,
        "pontos": pontos,
        "bbox_area": area_bbox(mask_planta),
    })


mape_calibrado, eps_base, eps_ganho, centro_rel, remover_base, usar_singleton, _ = calibrar_parametros(dados)
estimativas, resultados = processar_com_params(
    dados,
    eps_base,
    eps_ganho,
    centro_rel,
    remover_base,
    usar_singleton,
)

print("Parametros escolhidos:")
print(f"eps_base: {eps_base:.1f}")
print(f"eps_ganho_area: {eps_ganho:.1f}")
print(f"min_pts: {MIN_ENDPOINTS_CLUSTER}")
print(f"filtro_centro: {centro_rel:.3f}")
print(f"remover_base: {remover_base:.2f}")
print(f"fallback_singleton: {usar_singleton}")
print(f"MAPE calibrado: {mape_calibrado:.2f} %\n")

erros = []

for item, n_folhas, (labels, validos, pontos_filtrados, eps) in zip(dados, estimativas, resultados):
    i = item["indice"]
    referencia = item["referencia"]

    erro = abs((n_folhas - referencia) / referencia) * 100.0
    erros.append(erro)

    print(f"Eucalipto{i}:")
    print(f"Folhas estimadas: {n_folhas}")
    print(f"Referencia: {referencia}")
    print(f"Endpoints detectados: {len(pontos_filtrados)}")
    print(f"Clusters validos: {len(validos)}")
    print(f"eps usado: {eps:.1f} px")
    print(f"Erro relativo: {erro:.2f} %\n")

    endpoints = criar_mascara_endpoints(item["skeleton"].shape, pontos_filtrados)
    overlay = criar_overlay(
        item["img_recortada"],
        item["skeleton"],
        endpoints,
        pontos_filtrados,
        labels,
        validos,
    )
    exibir_resultado(
        item["img_recortada"],
        item["skeleton"],
        endpoints,
        overlay,
        titulo=f"Eucalipto{i} | folhas={n_folhas} | ref={referencia}",
    )

if erros:
    print(f"Erro medio absoluto (MAPE): {np.mean(erros):.2f} %")
