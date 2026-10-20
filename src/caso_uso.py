"""
Caso de uso: roteiro turistico em Pirenopolis (GO), a pe ou de carro.

PERGUNTA
--------
Quanto de experiencia imersiva, medida pela Meta-Imersao, se perde ao
desconcentrar a visitacao dos pontos turisticos consolidados para os menos
consolidados?

O dataset real de atencao (UOAL, 30 usuarios x 96 objetos, sem geografia) nao
responde essa pergunta. O gerador validado fornece quantos visitantes forem
necessarios; este modulo os coloca sobre a malha viaria real da cidade.

SEPARACAO ENTRE PREFERENCIA E ATENCAO
-------------------------------------
Sao duas grandezas distintas, usadas em momentos distintos:

    preferencia  declarada ANTES da visita, sobre a categoria do ponto.
                 Decide QUAIS pontos entram na rota.
    atencao      medida NO LOCAL (exogena, Du et al.). Decide como a
                 capacidade de renderizacao e repartida entre os pontos
                 visitados.

ESCOLHA DA ROTA
---------------
    max_{S, pi}  SUM_{n in S} pref[u,n] * pop[n]^(-lambda)
    s.a.         dist(pi) / v + t_visita * (|S| - 1) <= T
                 pi parte da Igreja Matriz

    pop[n] = c[n] / max(c)       c = numero de avaliacoes publicas do ponto
    lambda em [0, 1]             0: o visitante segue a multidao
                                 1: a popularidade deixa de pesar; resta so
                                    o gosto individual por categoria
    T                            tempo disponivel do visitante
    t_visita                     permanencia por ponto (constante declarada)

A penalidade multiplicativa pop^(-lambda) foi preferida a (1 - lambda*pop):
esta ultima e monotona crescente em pop sempre que pop < 1/(2*lambda), o que
vale para todos os pontos exceto a partida; ela atenua o contraste mas nao
reordena os pontos, e a curva sai plana. Com lambda = 1, pref * pop^(-1) e
exatamente o gosto individual (H1), sem o componente de popularidade.

O orcamento e de TEMPO, nao de distancia: num centro historico compacto o
deslocamento a pe e pequeno frente a permanencia em cada ponto, e e a
permanencia que obriga o visitante a escolher.

Resolvido exatamente: programacao dinamica sobre subconjuntos da o menor
percurso de cada subconjunto (Held-Karp em caminho aberto); o otimo e o
subconjunto viavel de maior peso.

ALOCACAO E MEDIDA
-----------------
Dada a rota, a capacidade (CAP_POR_OBJETO por ponto visitado) e repartida por
water-filling sobre a atencao (metricas.aloca) e comparada com a divisao
uniforme. A experiencia e medida pela Meta-Imersao e pela QoE-CAC.

Decomposicao usada nos resultados:

    MI = T(R) * T(1-E) * A,      A = SUM_n K_n ln(P_n / P_th)

A (componente de atencao) depende so da rota e da atencao. T(R) depende do
campo de rede, que e simulado (H3). A rota nao depende da rede, entao A e
calculado uma vez; a MI e a QoE-CAC sao promediadas sobre REPLICAS do campo
de rede (posicoes de estacoes base sorteadas), com intervalo de confianca
entre replicas. Sem isso, o resultado de uma unica disposicao de estacoes
confunde o efeito da desconcentracao com o acaso da cobertura.

EQUIDADE
--------
    equidade = 1 - Gini(visitas por ponto, somadas sobre os visitantes)

A Igreja Matriz e excluida: todos partem dela.

CURVA DE COMPROMISSO
--------------------
Cada lambda produz um par (equidade, Meta-Imersao media). A curva quantifica o
custo, em experiencia, de desconcentrar a visitacao.

HIPOTESES DE MODELAGEM (a declarar no texto)
--------------------------------------------
H1  Preferencia: pref[u,n] = pop[n] * exp(gama * z[u, categoria(n)]
    + delta * eta[u,n]), z, eta ~ N(0,1). Em valor esperado a preferencia
    agregada e proporcional a popularidade observada; o gosto individual varia
    por categoria e, com menor peso, por ponto.
H2  Correspondencia ponto-objeto por posto: os pontos, ordenados por
    popularidade, recebem objetos do UOAL ordenados por atencao media,
    escolhidos em postos igualmente espacados para cobrir toda a faixa.
    Ambos medem popularidade global; o 1o componente do UOAL (91,9% da
    variancia) e esse ranking.
H3  Rede: campo espacial simulado (perda de percurso + sombreamento), por
    ausencia de dado publico de cobertura por municipio.

USO
---
    python3 src/caso_uso.py --modo pe
    python3 src/caso_uso.py --modo pe --tempo 120 --permanencia 20
    python3 src/caso_uso.py --sem-osm          # malha sintetica, sem internet
"""

import argparse
import csv
import os
import sys
import textwrap
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from gerador import (MTP_COMPONENTES, S_FATORES, ajusta_atencao, carrega_referencia,
                     gera_atencao)
from metricas import (CAP_POR_OBJETO, LIM_DOWNLINK, LIM_LATENCIA, P_TH, aloca,
                      aloca_uniforme, somatorio_atencao, tau, tau_bep)

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "figure.constrained_layout.use": True, "pdf.fonttype": 42,
    "legend.fontsize": 8,
})
AZUL, LARANJA, VERDE, CINZA = "#3b6ea5", "#c2703d", "#4c8055", "#9a9a9a"

MODOS = {
    # nome: (tipo de malha OSM, raio em m, velocidade km/h, tempo padrao em min)
    "pe": ("walk", 1500.0, 5.0, 90.0),
    "carro": ("drive", 20000.0, 30.0, 240.0),
}
T_VISITA = 15.0  # min de permanencia por ponto
LIMITE_EXATO = 15
GAMA_PREFERENCIA = 0.5   # heterogeneidade de gosto por categoria
DELTA_PREFERENCIA = 0.25 # variacao individual por ponto
LAMBDAS = np.round(np.linspace(0.0, 1.0, 11), 2)
BEP_FIXA = 1e-5
# Latencia motion-to-photon: parcelas locais no ponto medio das faixas do
# gerador; a parcela de rede varia com a posicao dentro da faixa do gerador.
LAT_LOCAL = sum(np.mean(MTP_COMPONENTES[c]) for c in ("sensor", "processamento",
                                                       "renderizacao"))
N_ESTACOES, EXPOENTE_PERDA, DIST_REF, SOMBRA_DB = 4, 3.0, 50.0, 6.0

ROTULOS_ATENCAO = {1: "1 nenhum interesse", 2: "2 pouco", 3: "3 moderado",
                   4: "4 alto", 5: "5 plena experiência"}


# --------------------------------------------------------------------------- #
# Pontos turisticos
# --------------------------------------------------------------------------- #
def carrega_pois(caminho):
    """Le a lista de pontos. Linhas iniciadas por '#' sao comentarios.

    O primeiro ponto e a partida (Igreja Matriz). Devolve tambem os comentarios,
    para que a origem e a data da coleta acompanhem os resultados.
    """
    comentarios, linhas = [], []
    with open(caminho, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            (comentarios if l.startswith("#") else linhas).append(l)
    leitor = csv.DictReader(linhas, delimiter=";")
    pois = [{"nome": r["nome"].strip(), "categoria": r["categoria"].strip(),
             "avaliacoes": float(r["avaliacoes"]), "lat": float(r["lat"]),
             "lon": float(r["lon"]),
             "nota": float(r["nota"]) if (r.get("nota") or "").strip() else float("nan")}
            for r in leitor]
    if any(p["avaliacoes"] < 0 for p in pois):
        raise ValueError("numero de avaliacoes negativo")
    if len(pois) < 3:
        raise ValueError("lista de pontos com menos de 3 entradas")
    if len(pois) > LIMITE_EXATO:
        raise ValueError(f"{len(pois)} pontos: a solucao exata vai ate {LIMITE_EXATO}")
    return pois, comentarios


SUAVIZACAO = 1.0  # contagem acrescida a todos os pontos (suavizacao aditiva)


def popularidade(pois, alfa=SUAVIZACAO):
    """pop[n] = (c[n] + alfa) / (max(c) + alfa). Fonte unica: numero de avaliacoes.

    A suavizacao aditiva trata pontos com perfil sem avaliacoes (c = 0) como os
    menos populares, sem popularidade nula. Com pop = 0 a penalidade pop^-lambda
    seria indefinida. O efeito sobre os demais pontos e desprezivel (alfa = 1
    frente a contagens de centenas). Note que pref * pop^-lambda = pop^(1-lambda)
    * exp(gama z): o peso permanece limitado para todo lambda em [0, 1].
    """
    c = np.array([p["avaliacoes"] for p in pois], float)
    return (c + alfa) / (c.max() + alfa)


def projeta_local(pois):
    """Projecao equirretangular em metros, com origem no primeiro ponto.
    Erro desprezivel na escala de poucos km."""
    R = 6371000.0
    lat0, lon0 = np.radians(pois[0]["lat"]), np.radians(pois[0]["lon"])
    lat = np.radians([p["lat"] for p in pois])
    lon = np.radians([p["lon"] for p in pois])
    return R * (lon - lon0) * np.cos(lat0), R * (lat - lat0)


# --------------------------------------------------------------------------- #
# Mapa
# --------------------------------------------------------------------------- #
def malha_sintetica(xs, ys, passo=60.0, margem=300.0, semente=0):
    """Grade irregular cobrindo os pontos, para rodar sem OpenStreetMap."""
    import networkx as nx

    rng = np.random.default_rng(semente)
    gx = np.arange(xs.min() - margem, xs.max() + margem + passo, passo)
    gy = np.arange(ys.min() - margem, ys.max() + margem + passo, passo)
    H = nx.Graph()
    for i, x in enumerate(gx):
        for j, y in enumerate(gy):
            H.add_node((i, j), x=x + rng.normal(0, 8), y=y + rng.normal(0, 8))
    for a, b in nx.grid_2d_graph(len(gx), len(gy)).edges:
        d = float(np.hypot(H.nodes[a]["x"] - H.nodes[b]["x"],
                           H.nodes[a]["y"] - H.nodes[b]["y"]))
        H.add_edge(a, b, length=d)
    return H


def carrega_mapa(modo, pois, dados="dados", usar_osm=True):
    """Malha viaria em metros e posicoes dos pontos no mesmo sistema.

    Com OSM: baixa uma vez por modo, centrada na partida, e guarda em disco.
    Sem OSM: projecao local e grade sintetica.
    """
    tipo, raio, _, _ = MODOS[modo]
    if usar_osm:
        try:
            import osmnx as ox
            from pyproj import Transformer

            caminho = os.path.join(dados, f"pirenopolis_{modo}.graphml")
            if os.path.exists(caminho):
                G = ox.load_graphml(caminho)
            else:
                G = ox.graph_from_point((pois[0]["lat"], pois[0]["lon"]),
                                        dist=raio, network_type=tipo)
                ox.save_graphml(G, caminho)
            G = ox.project_graph(G)
            t = Transformer.from_crs("EPSG:4326", G.graph["crs"], always_xy=True)
            xs, ys = t.transform([p["lon"] for p in pois], [p["lat"] for p in pois])
            return G, np.asarray(xs), np.asarray(ys), f"osm {tipo} (raio {raio / 1000:.1f} km)"
        except Exception as e:
            warnings.warn(f"OpenStreetMap indisponivel ({type(e).__name__}); "
                          "usando malha sintetica", RuntimeWarning, stacklevel=2)
    xs, ys = projeta_local(pois)
    return malha_sintetica(xs, ys), xs, ys, "sintetica (teste)"


def matriz_distancias(G, xs, ys):
    """Distancia pelo caminho minimo na malha entre os nos mais proximos, somada
    aos trechos de acesso (ponto -> no mais proximo, em linha reta) nas duas
    pontas. Devolve tambem um diagnostico: pares sem caminho na malha (para os
    quais se usa a distancia euclidiana), maior assimetria antes da
    simetrizacao e maior trecho de acesso."""
    import networkx as nx

    nos = list(G.nodes)
    nx_ = np.array([float(G.nodes[n]["x"]) for n in nos])
    ny_ = np.array([float(G.nodes[n]["y"]) for n in nos])
    idx = [int(np.argmin((nx_ - x) ** 2 + (ny_ - y) ** 2)) for x, y in zip(xs, ys)]
    prox = [nos[i] for i in idx]
    acesso = np.array([np.hypot(nx_[i] - x, ny_[i] - y) for i, x, y in zip(idx, xs, ys)])
    n = len(xs)
    sem_caminho = 0
    euclid = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
    D = np.zeros((n, n))
    for i in range(n):
        try:
            comp = nx.single_source_dijkstra_path_length(G, prox[i], weight="length")
        except Exception:
            comp = {}
        for j in range(n):
            if i == j:
                continue
            if prox[j] in comp:
                D[i, j] = comp[prox[j]] + acesso[i] + acesso[j]
            else:
                D[i, j] = euclid[i, j]
                sem_caminho += 1
    assimetria = float(np.abs(D - D.T).max())
    D = np.minimum(D, D.T)
    np.fill_diagonal(D, 0.0)
    return D, {"sem_caminho": sem_caminho, "assimetria": assimetria,
               "acesso_max": float(acesso.max())}


# --------------------------------------------------------------------------- #
# Rede espacial (H3)
# --------------------------------------------------------------------------- #
def campo_rede(xs, ys, semente=0):
    rng = np.random.default_rng(semente)
    ex = rng.uniform(xs.min(), xs.max(), N_ESTACOES)
    ey = rng.uniform(ys.min(), ys.max(), N_ESTACOES)
    d = np.maximum(np.hypot(xs[:, None] - ex[None, :], ys[:, None] - ey[None, :]).min(1),
                   DIST_REF)
    perda = 10 * EXPOENTE_PERDA * np.log10(d / DIST_REF) + rng.normal(0, SOMBRA_DB, len(d))
    q = np.clip(1.0 - (perda - perda.min()) / (np.ptp(perda) + 1e-9), 0.0, 1.0)
    lo, hi = LIM_DOWNLINK
    rlo, rhi = MTP_COMPONENTES["rede"]
    return lo + q * (hi - lo), rlo + (1.0 - q) * (rhi - rlo), (ex, ey)


# --------------------------------------------------------------------------- #
# Preferencia (H1) e atencao (H2)
# --------------------------------------------------------------------------- #
def gera_preferencia(pop, categorias, n_visitantes, rng, gama=GAMA_PREFERENCIA,
                     delta=DELTA_PREFERENCIA):
    """pref[u,n] = pop[n] * exp(gama * z[u, cat(n)] + delta * eta[u,n]),
    z, eta ~ N(0,1) independentes.

    E[pref[u,n]] = pop[n] * exp((gama^2 + delta^2) / 2): a media sobre
    visitantes reproduz a popularidade observada a menos de uma constante.
    O termo por ponto (delta) representa o gosto individual por um ponto
    especifico dentro da categoria; sem ele, pontos da mesma categoria teriam
    peso identico quando lambda = 1, e a escolha entre eles seria decidida por
    desempate, nao por preferencia.
    """
    cats = sorted(set(categorias))
    idx = np.array([cats.index(c) for c in categorias])
    z = rng.standard_normal((n_visitantes, len(cats)))
    eta = rng.standard_normal((n_visitantes, len(categorias)))
    return pop[None, :] * np.exp(gama * z[:, idx] + delta * eta)


def objetos_por_posto(gd, pop):
    """H2: emparelha pontos e objetos do UOAL pelo posto de popularidade.

    Seleciona len(pop) objetos em postos igualmente espacados da atencao media
    (do mais ao menos atendido) e atribui o i-esimo mais atendido ao i-esimo
    ponto mais popular.
    """
    ordem_obj = np.argsort(-gd.mean(axis=0))
    postos = np.rint(np.linspace(0, len(ordem_obj) - 1, len(pop))).astype(int)
    escolhidos = ordem_obj[postos]
    obj = np.empty(len(pop), int)
    obj[np.argsort(-pop, kind="stable")] = escolhidos
    return obj


# --------------------------------------------------------------------------- #
# Rota exata
# --------------------------------------------------------------------------- #
def menor_percurso_por_subconjunto(D, inicio=0):
    """Held-Karp em caminho aberto: menor distancia que visita cada subconjunto
    partindo de `inicio`. Devolve mascaras, distancias e o predecessor para
    reconstruir a ordem."""
    n = len(D)
    INF = np.inf
    custo = np.full((1 << n, n), INF)
    pred = np.full((1 << n, n), -1, int)
    custo[1 << inicio, inicio] = 0.0
    for masc in range(1 << n):
        if not masc & (1 << inicio):
            continue
        for ult in range(n):
            c = custo[masc, ult]
            if c == INF:
                continue
            for prox in range(n):
                if masc & (1 << prox):
                    continue
                m2 = masc | (1 << prox)
                c2 = c + D[ult, prox]
                if c2 < custo[m2, prox]:
                    custo[m2, prox], pred[m2, prox] = c2, ult
    melhor_ult = custo.argmin(axis=1)
    melhor = custo[np.arange(1 << n), melhor_ult]
    validas = np.isfinite(melhor)
    return np.nonzero(validas)[0], melhor[validas], melhor_ult[validas], pred


def reconstroi(masc, ult, pred):
    rota = []
    while ult != -1:
        rota.append(int(ult))
        m_ant, ult = masc, pred[masc, ult]
        masc = m_ant & ~(1 << rota[-1])
    return rota[::-1]


class Roteador:
    """Pre-calcula os percursos minimos uma vez; cada visitante so escolhe o
    subconjunto viavel de maior peso."""

    def __init__(self, D, vel_kmh, t_visita=T_VISITA, inicio=0):
        self.n = len(D)
        self.mascaras, self.dist, self.ult, self.pred = menor_percurso_por_subconjunto(D, inicio)
        bits = (self.mascaras[:, None] >> np.arange(self.n)) & 1
        self.bits = bits.astype(float)
        self.tempo = self.dist / (vel_kmh / 3.6) / 60.0 + t_visita * (self.bits.sum(1) - 1)

    def rota(self, pesos, tempo_disp):
        viaveis = self.tempo <= tempo_disp
        valor = np.where(viaveis, self.bits @ pesos, -np.inf)
        # desempate: menor distancia entre subconjuntos de mesmo valor
        i = np.lexsort((self.dist, -valor))[0]
        return (reconstroi(int(self.mascaras[i]), int(self.ult[i]), self.pred),
                float(self.dist[i]), float(self.tempo[i]))


# --------------------------------------------------------------------------- #
# Avaliacao
# --------------------------------------------------------------------------- #
def componentes(rota, K_u):
    """Componente de atencao A da rota, com alocacao por atencao e uniforme."""
    idx = np.asarray(rota, int)
    orc = len(idx) * CAP_POR_OBJETO
    P_at, P_un = aloca(K_u[idx], orc, P_TH), aloca_uniforme(len(idx), orc, P_TH)
    return somatorio_atencao(K_u[idx], P_at, P_TH), somatorio_atencao(K_u[idx], P_un, P_TH)


def gini(x):
    x = np.asarray(x, float)
    if x.sum() == 0:
        return 0.0
    return float(np.abs(x[:, None] - x[None, :]).sum() / (2 * len(x) ** 2 * x.mean()))


def _ic(v):
    """Media e meia-largura do IC de 95% entre replicas."""
    v = np.asarray(v, float)
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def varre_lambda(roteador, pref, pop, K, campos, tempo_disp, lambdas=LAMBDAS):
    """Para cada lambda: rotas, equidade, componente de atencao e metricas
    promediadas sobre as replicas do campo de rede.

    campos: lista de (downlink, lat_rede) por ponto, uma por replica.
    """
    n_vis, n = pref.shape
    tb = tau_bep(BEP_FIXA)
    linhas, rotas_ex = [], {}
    for lam in lambdas:
        visitas = np.zeros(n)
        rotas, A_at, A_un, npt, dist, tmp = [], [], [], [], [], []
        for u in range(n_vis):
            rota, d, t = roteador.rota(pref[u] * pop ** (-lam), tempo_disp)
            visitas[rota] += 1
            a, b = componentes(rota, K[u])
            rotas.append(np.asarray(rota, int)); A_at.append(a); A_un.append(b)
            npt.append(len(rota)); dist.append(d); tmp.append(t)
        rotas_ex[float(lam)] = rotas[0]
        A_at, A_un = np.array(A_at), np.array(A_un)

        por_rep = {k: [] for k in ("mi", "mi_uniforme", "cac", "cac_uniforme")}
        tdl_rep, tlat_rep = [], []
        for dl, lr in campos:
            t_dl = np.array([tau(dl[r].mean(), LIM_DOWNLINK) for r in rotas])
            t_lat = np.array([1.0 - tau(LAT_LOCAL + lr[r].mean(), LIM_LATENCIA)
                              for r in rotas])
            tdl_rep.append(t_dl)
            tlat_rep.append(t_lat)
            por_rep["mi"].append(float((t_dl * tb * A_at).mean()))
            por_rep["mi_uniforme"].append(float((t_dl * tb * A_un).mean()))
            por_rep["cac"].append(float((t_lat * t_dl * tb * A_at).mean()))
            por_rep["cac_uniforme"].append(float((t_lat * t_dl * tb * A_un).mean()))

        linha = {"lambda": float(lam), "equidade": 1.0 - gini(visitas[1:]),
                 "pontos": float(np.mean(npt)), "dist": float(np.mean(dist)),
                 "tempo": float(np.mean(tmp)), "visitas": visitas,
                 "A": float(A_at.mean()), "A_uniforme": float(A_un.mean()),
                 "A_vis": A_at, "A_un_vis": A_un,
                 "tdl_rep": np.array(tdl_rep), "tlat_rep": np.array(tlat_rep)}
        for k, v in por_rep.items():
            linha[k], linha[k + "_ic"] = _ic(v)
            linha[k + "_rep"] = np.asarray(v)
        linhas.append(linha)
    return linhas, rotas_ex


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def _fundo_malha(ax, G):
    for a, b in G.edges():
        ax.plot([G.nodes[a]["x"], G.nodes[b]["x"]], [G.nodes[a]["y"], G.nodes[b]["y"]],
                "-", color="#dcdcdc", lw=0.4, zorder=1)


CMAP_ATENCAO = ListedColormap(plt.get_cmap("YlOrRd")(np.linspace(0.15, 0.95, 5)))
NORM_ATENCAO = BoundaryNorm(np.arange(0.5, 6.5, 1.0), 5)


def _desenha_rota(ax, G, xs, ys, rota, K_u, estacoes, titulo, margem=120.0):
    _fundo_malha(ax, G)
    ax.scatter(*estacoes, marker="^", s=45, c=VERDE, zorder=3, edgecolors="w", lw=0.6)
    niveis = np.clip(np.rint(K_u).astype(int), 1, 5)
    fora = np.setdiff1d(np.arange(len(xs)), rota)
    ax.scatter(xs[fora], ys[fora], s=28, c=CINZA, alpha=0.6, zorder=3)
    r = np.asarray(rota, int)
    for a, b in zip(r[:-1], r[1:]):  # cada trecho na cor do ponto de chegada
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], "-", lw=2.6, zorder=4,
                color=CMAP_ATENCAO(NORM_ATENCAO(niveis[b])))
    sc = ax.scatter(xs[r], ys[r], s=55, c=niveis[r], cmap=CMAP_ATENCAO, norm=NORM_ATENCAO,
                    zorder=5, edgecolors="k", linewidths=0.6)
    ax.scatter(xs[r[0]], ys[r[0]], marker="s", s=80, c="k", zorder=6)
    # enquadra os pontos turisticos, nao a malha inteira: a malha baixada cobre
    # o raio do modo, muito maior que a regiao dos pontos
    ax.set_xlim(xs.min() - margem, xs.max() + margem)
    ax.set_ylim(ys.min() - margem, ys.max() + margem)
    ax.set_title(titulo, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.grid(False)
    return sc


def fig_rotas(G, xs, ys, rotas_ex, K_u, estacoes, modo, saida):
    """Rota do mesmo visitante sem penalidade e com penalidade maxima."""
    lams = (min(rotas_ex), max(rotas_ex))
    fig, ax = plt.subplots(1, 2, figsize=(7.8, 4.0))
    for a, lam in zip(ax, lams):
        sc = _desenha_rota(a, G, xs, ys, list(rotas_ex[lam]), K_u, estacoes,
                           f"λ = {lam:.1f} — {len(rotas_ex[lam])} pontos")
    cb = fig.colorbar(sc, ax=ax, ticks=[1, 2, 3, 4, 5], shrink=0.75, pad=0.02)
    cb.ax.set_yticklabels([ROTULOS_ATENCAO[i] for i in range(1, 6)])
    cb.set_label("atenção do visitante no ponto")
    h = [Line2D([], [], marker="^", ls="", color=VERDE, ms=7),
         Line2D([], [], marker="s", ls="", color="k", ms=7),
         Line2D([], [], marker="o", ls="", color=CINZA, ms=6)]
    ax[0].legend(h, ["estação base (réplica 1)", "partida (Igreja Matriz)", "fora da rota"],
                 frameon=True, framealpha=0.9, loc="best")
    fig.savefig(os.path.join(saida, f"caso_uso_rotas_{modo}.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_curva(linhas, modo, saida):
    """(a) Curva de compromisso: equidade x Meta-Imersao media, IC de 95% entre
    replicas do campo de rede. (b) Variacao relativa a lambda = 0, separando o
    componente de atencao da metrica completa."""
    eq = np.array([l["equidade"] for l in linhas])
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for k, cor, rot in (("mi", AZUL, "alocação por atenção"),
                        ("mi_uniforme", LARANJA, "alocação uniforme")):
        m = np.array([l[k] for l in linhas]); e = np.array([l[k + "_ic"] for l in linhas])
        ax[0].plot(eq, m, "o-", color=cor, ms=4, label=rot)
        ax[0].fill_between(eq, m - e, m + e, color=cor, alpha=0.18)
    for l, dx, ha in ((linhas[0], 4, "left"), (linhas[len(linhas) // 2], 4, "left"),
                      (linhas[-1], -4, "right")):
        ax[0].annotate(f"λ={l['lambda']:.1f}", (l["equidade"], l["mi"]),
                       textcoords="offset points", xytext=(dx, 6), ha=ha, fontsize=7)
    ax[0].set_xlabel("equidade da visitação (1 − Gini)")
    ax[0].set_ylabel("Meta-Imersão média por visitante")
    ax[0].legend(frameon=False)
    ax[0].set_title("(a)", fontsize=9, loc="left")

    lam = np.array([l["lambda"] for l in linhas])
    a0 = linhas[0]["A"]
    ax[1].plot(lam, [100 * (l["A"] / a0 - 1) for l in linhas], "o-", color=VERDE, ms=4,
               label="componente de atenção A")
    r0 = linhas[0]["mi_rep"]  # variacao pareada por replica do campo de rede
    pares = [_ic(100 * (l["mi_rep"] / r0 - 1)) for l in linhas]
    m, e = np.array([p[0] for p in pares]), np.array([p[1] for p in pares])
    ax[1].plot(lam, m, "s--", color=AZUL, ms=3.5, label="Meta-Imersão (IC 95% pareado)")
    ax[1].fill_between(lam, m - e, m + e, color=AZUL, alpha=0.15)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("λ (peso contra a popularidade)")
    ax[1].set_ylabel("variação em relação a λ = 0 (%)")
    ax[1].legend(frameon=False)
    ax[1].set_title("(b)", fontsize=9, loc="left")
    fig.savefig(os.path.join(saida, f"caso_uso_curva_{modo}.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_visitas(linhas, nomes, modo, saida):
    """Visitas por ponto sem penalidade e com penalidade maxima."""
    a, b = linhas[0], linhas[-1]
    ordem = np.argsort(-a["visitas"][1:]) + 1
    x = np.arange(len(ordem))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x - 0.2, a["visitas"][ordem], 0.4, color=AZUL, label=f"λ = {a['lambda']:.1f}")
    ax.bar(x + 0.2, b["visitas"][ordem], 0.4, color=LARANJA, label=f"λ = {b['lambda']:.1f}")
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.shorten(nomes[i], 34, placeholder="…") for i in ordem],
                       rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("visitantes")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(saida, f"caso_uso_visitas_{modo}.pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def ic_desconcentrar(a, b, reps=2000, semente=7):
    """IC de 95% por bootstrap sobre visitantes E replicas da rede (pareado por
    visitante entre lambda = 0 e 1) para Delta A, ganho em lambda = 0 e 1,
    Delta MI e Delta QoE-CAC, em %. Delta MI e Delta QoE-CAC usam o mesmo
    estimador da Figura 3(b): variacao dentro de cada replica da rede, media
    entre replicas."""
    rng = np.random.default_rng(semente)
    n_vis, n_rep = len(a["A_vis"]), a["tdl_rep"].shape[0]
    out = {k: [] for k in ("dA", "g0", "g1", "dmi", "dcac")}
    for _ in range(reps):
        iu = rng.integers(0, n_vis, n_vis)
        ir = rng.integers(0, n_rep, n_rep)
        Aa, Ab = a["A_vis"][iu], b["A_vis"][iu]
        out["dA"].append(100 * (Ab.mean() / Aa.mean() - 1))
        out["g0"].append(100 * (Aa.mean() / a["A_un_vis"][iu].mean() - 1))
        out["g1"].append(100 * (Ab.mean() / b["A_un_vis"][iu].mean() - 1))
        mia = (a["tdl_rep"][ir][:, iu] * Aa).mean(axis=1)
        mib = (b["tdl_rep"][ir][:, iu] * Ab).mean(axis=1)
        out["dmi"].append(100 * (mib / mia - 1).mean())
        caa = (a["tlat_rep"][ir][:, iu] * a["tdl_rep"][ir][:, iu] * Aa).mean(axis=1)
        cab = (b["tlat_rep"][ir][:, iu] * b["tdl_rep"][ir][:, iu] * Ab).mean(axis=1)
        out["dcac"].append(100 * (cab / caa - 1).mean())
    return {k: tuple(np.percentile(v, [2.5, 97.5])) for k, v in out.items()}


def sensibilidade_mapeamento(roteador, pref, pop, K, obj, tempo_disp, n_obj,
                             nota=None, reps=50, semente=99):
    """H2 em teste: o mapeamento por ranking contra mapeamentos alternativos.

    As rotas dependem so da preferencia e da popularidade, nao da atencao; por
    isso sao calculadas uma vez (lambda = 0 e 1) e reaproveitadas. Para cada
    mapeamento, recalculam-se o componente de atencao e o ganho da alocacao
    por atencao sobre a uniforme.
    """
    rotas = {lam: [np.asarray(roteador.rota(pref[u] * pop ** (-lam), tempo_disp)[0], int)
                   for u in range(pref.shape[0])] for lam in (0.0, 1.0)}

    def mede(o):
        Kp = K[:, o].astype(float)
        res = {}
        for lam, rs in rotas.items():
            ab = np.array([componentes(r, Kp[u]) for u, r in enumerate(rs)])
            res[lam] = (ab[:, 0].mean(), ab[:, 1].mean())
        dA = 100 * (res[1.0][0] / res[0.0][0] - 1)
        g0 = 100 * (res[0.0][0] / res[0.0][1] - 1)
        g1 = 100 * (res[1.0][0] / res[1.0][1] - 1)
        return dA, g0, g1

    rng = np.random.default_rng(semente)
    saida = {"ranking (H2)": [mede(obj)]}
    if nota is not None:
        # mapeamento empirico alternativo: os mesmos objetos, ordenados pela nota
        # media do Google Maps (sem nota = ultimo; empate decidido pela popularidade)
        escolhidos = obj[np.argsort(-pop, kind="stable")]
        nota_f = np.where(np.isnan(nota), -np.inf, nota)
        obj_nota = np.empty_like(obj)
        obj_nota[np.lexsort((-pop, -nota_f))] = escolhidos
        saida["nota do Google Maps"] = [mede(obj_nota)]
    saida["mesmos objetos permutados"] = [mede(rng.permutation(obj)) for _ in range(reps)]
    saida["objetos sorteados entre os 96"] = [mede(rng.choice(n_obj, len(obj), replace=False))
                                              for _ in range(reps)]
    return saida


class Relatorio:
    """Escreve na tela e num arquivo ao mesmo tempo (dispensa o tee)."""

    def __init__(self, caminho):
        self.tela, self.arquivo = sys.stdout, open(caminho, "w", encoding="utf-8")

    def write(self, texto):
        self.tela.write(texto)
        self.arquivo.write(texto)

    def flush(self):
        self.tela.flush()
        self.arquivo.flush()

    def fecha(self):
        sys.stdout = self.tela
        self.arquivo.close()


def main():
    ap = argparse.ArgumentParser(description="Roteiro turistico: experiencia x equidade.")
    ap.add_argument("--modo", default="pe", choices=list(MODOS))
    ap.add_argument("--pois", default=os.path.join("dados", "pois_pirenopolis.csv"))
    ap.add_argument("--dados", default="dados")
    ap.add_argument("--figuras", default="figuras")
    ap.add_argument("--resultados", default="resultados")
    ap.add_argument("--perfis", type=int, default=300)
    ap.add_argument("--tempo", type=float, default=None, help="minutos disponiveis")
    ap.add_argument("--permanencia", type=float, default=T_VISITA, help="minutos por ponto")
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--replicas", type=int, default=30,
                    help="replicas do campo de rede (posicoes das estacoes)")
    ap.add_argument("--sem-osm", action="store_true")
    ap.add_argument("--excluir-sem-avaliacao", action="store_true",
                    help="analise de sensibilidade: retira pontos com 0 avaliacoes")
    args = ap.parse_args()
    for d in (args.figuras, args.resultados):
        os.makedirs(d, exist_ok=True)
    sufixo = (f"_t{args.tempo:g}" if args.tempo else "") + \
             ("_sem_avaliacao" if args.excluir_sem_avaliacao else "")
    caminho_relatorio = os.path.join(args.resultados,
                                     f"relatorio_caso_uso_{args.modo}{sufixo}.txt")
    sys.stdout = Relatorio(caminho_relatorio)

    _, _, vel, t_padrao = MODOS[args.modo]
    tempo_disp = args.tempo or t_padrao

    pois, comentarios = carrega_pois(args.pois)
    if args.excluir_sem_avaliacao:
        fora = [p["nome"] for p in pois[1:] if p["avaliacoes"] == 0]
        pois = [pois[0]] + [p for p in pois[1:] if p["avaliacoes"] > 0]
        comentarios = comentarios + [f"# Sensibilidade: excluidos sem avaliacoes: {', '.join(fora)}"]
    nomes = [p["nome"] for p in pois]
    pop = popularidade(pois)
    G, xs, ys, origem = carrega_mapa(args.modo, pois, args.dados, not args.sem_osm)
    D, diag_dist = matriz_distancias(G, xs, ys)
    campos, estacoes = [], None
    for r in range(args.replicas):
        dl, lr, est = campo_rede(xs, ys, args.semente * 1000 + r)
        campos.append((dl, lr))
        estacoes = est if estacoes is None else estacoes

    ss = np.random.SeedSequence(args.semente).spawn(2)
    gd, _, _ = carrega_referencia(args.dados)
    K = gera_atencao(ajusta_atencao(gd, S_FATORES), args.perfis,
                     np.random.default_rng(ss[0]))
    obj = objetos_por_posto(gd, pop)
    K_pois = K[:, obj].astype(float)
    pref = gera_preferencia(pop, [p["categoria"] for p in pois], args.perfis,
                            np.random.default_rng(ss[1]))
    roteador = Roteador(D, vel, args.permanencia, inicio=0)

    print("=" * 74)
    for c in comentarios:
        print(c)
    print(f"CASO DE USO — {'a pe' if args.modo == 'pe' else 'de carro'} | malha: {origem}")
    print(f"{len(pois)} pontos | {args.perfis} visitantes | tempo {tempo_disp:.0f} min "
          f"| permanencia {args.permanencia:.0f} min/ponto | {vel:.0f} km/h")
    print(f"partida: {nomes[0]}")
    print(f"distancia pela malha: {D[D > 0].min():.0f} a {D.max():.0f} m | "
          f"percurso de todos os pontos: {roteador.dist.max():.0f} m")
    print(f"diagnostico das distancias: pares sem caminho na malha {diag_dist['sem_caminho']}"
          f" | maior assimetria {diag_dist['assimetria']:.1f} m"
          f" | maior trecho de acesso ponto-malha {diag_dist['acesso_max']:.0f} m (incluido)")
    print("=" * 74)

    print("\n[1] CALIBRACAO DA PREFERENCIA (H1) E CORRESPONDENCIA (H2)")
    r_pref = np.corrcoef(pref.mean(0), pop)[0, 1]
    r_at = np.corrcoef(K_pois.mean(0), pop)[0, 1]
    print(f"    correlacao preferencia media x popularidade: {r_pref:+.3f}")
    print(f"    correlacao atencao media x popularidade:     {r_at:+.3f}")
    print(f"    {'ponto':40s}{'pop':>6s}{'pref':>7s}{'aten':>7s}{'obj':>5s}")
    for i in np.argsort(-pop):
        print(f"    {nomes[i][:40]:40s}{pop[i]:6.2f}{pref[:, i].mean():7.2f}"
              f"{K_pois[:, i].mean():7.2f}{obj[i]:5d}")

    print("\n[2] CURVA DE COMPROMISSO (variando lambda)")
    print(f"    {args.replicas} replicas do campo de rede; IC de 95% entre replicas")
    linhas, rotas_ex = varre_lambda(roteador, pref, pop, K_pois, campos, tempo_disp)
    print(f"    {'lambda':>6s}{'equid.':>8s}{'pontos':>8s}{'min':>6s}{'A aten':>9s}"
          f"{'A unif':>8s}{'ganho':>8s}{'MI aten':>16s}{'QoE-CAC':>16s}")
    for l in linhas:
        g = 100 * (l["A"] / l["A_uniforme"] - 1)
        print(f"    {l['lambda']:6.1f}{l['equidade']:8.3f}{l['pontos']:8.2f}{l['tempo']:6.0f}"
              f"{l['A']:9.3f}{l['A_uniforme']:8.3f}{g:+7.1f}%"
              f"{l['mi']:9.3f}±{l['mi_ic']:.3f}{l['cac']:9.3f}±{l['cac_ic']:.3f}")
    a, b = linhas[0], linhas[-1]
    ic = ic_desconcentrar(a, b)
    f_ic = lambda k: f"(IC 95%: {ic[k][0]:+.1f}% a {ic[k][1]:+.1f}%)"
    print(f"\n    desconcentrar (lambda {a['lambda']:.0f} -> {b['lambda']:.0f});"
          f" IC 95% por bootstrap sobre visitantes e replicas da rede:")
    print(f"      equidade            {a['equidade']:.3f} -> {b['equidade']:.3f}")
    print(f"      componente atencao  {100 * (b['A'] / a['A'] - 1):+.1f}% {f_ic('dA')}")
    mi_m = float(np.mean(100 * (b["mi_rep"] / a["mi_rep"] - 1)))     # pareado por replica
    cac_m = float(np.mean(100 * (b["cac_rep"] / a["cac_rep"] - 1)))  # (como na Figura 3b)
    print(f"      Meta-Imersao        {mi_m:+.1f}% {f_ic('dmi')}")
    print(f"      QoE-CAC             {cac_m:+.1f}% {f_ic('dcac')}")
    print(f"      ganho da alocacao por atencao sobre a uniforme: "
          f"{100 * (a['A'] / a['A_uniforme'] - 1):+.1f}% {f_ic('g0')} -> "
          f"{100 * (b['A'] / b['A_uniforme'] - 1):+.1f}% {f_ic('g1')}")

    print("\n[3] SENSIBILIDADE AO MAPEAMENTO PONTO -> OBJETO (H2)")
    print("    mesmas rotas; so a atencao atribuida aos pontos muda"
          " (alternativos: media e faixa 5-95% de 50 sorteios)")
    print(f"    {'mapeamento':32s}{'Delta A (0->1)':>22s}{'ganho lambda=0':>24s}{'ganho lambda=1':>24s}")
    sens = sensibilidade_mapeamento(roteador, pref, pop, K, obj, tempo_disp, K.shape[1],
                                    nota=np.array([p["nota"] for p in pois]))
    for nome_m, vals in sens.items():
        v = np.array(vals)
        if len(v) == 1:
            cel = [f"{x:+.1f}%" for x in v[0]]
        else:
            cel = [f"{m:+.1f}% [{lo:+.1f}; {hi:+.1f}]" for m, lo, hi in
                   zip(v.mean(0), np.percentile(v, 5, axis=0), np.percentile(v, 95, axis=0))]
        print(f"    {nome_m:32s}{cel[0]:>22s}{cel[1]:>24s}{cel[2]:>24s}")

    fig_rotas(G, xs, ys, rotas_ex, K_pois[0], estacoes, args.modo, args.figuras)
    fig_curva(linhas, args.modo, args.figuras)
    fig_visitas(linhas, nomes, args.modo, args.figuras)

    c = os.path.join(args.resultados, f"caso_uso_{args.modo}.csv")
    campos = [k for k in linhas[0] if k != "visitas" and not k.endswith("_rep")
              and not k.endswith("_vis")]
    with open(c, "w", newline="", encoding="utf-8") as f:
        for linha in comentarios:
            f.write(linha + "\n")
        w = csv.writer(f)
        w.writerow(campos + [f"visitas_{i}" for i in range(len(pois))])
        for l in linhas:
            w.writerow([l[k] for k in campos] + [int(v) for v in l["visitas"]])
    print(f"\nresultados: {c}")
    print(f"relatorio:  {caminho_relatorio}")
    for nome in sorted(os.listdir(args.figuras)):
        if nome.startswith("caso_uso") and args.modo in nome:
            print(f"figura:     {os.path.join(args.figuras, nome)}")


if __name__ == "__main__":
    main()
    if isinstance(sys.stdout, Relatorio):
        sys.stdout.fecha()
