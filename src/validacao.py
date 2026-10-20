"""
Validacao do dataset sintetico.

Dados de referencia: UOAL (Du et al.), 30 participantes x 96 objetos, derivado
do rastreamento ocular do PSM (Xu et al.). A mascara de train.txt separa as
celulas OBSERVADAS (historico, usadas no treino) das OCULTAS (teste). Os valores
usados em ambas vem de gd.txt.

REGRA: nenhuma celula oculta entra no treino. O gerador e os modelos do braco
de referencia sao ajustados so com as celulas observadas.

Ensaios:

    A. ENSAIO INTERNO (diagnostico)
       Mascara parte das atencoes, preve com ALS e ANLS, mede RMSE. Roda na
       referencia e no sintetico.

    B. TSTR x TRTR (teste supervisionado)
       Cada modelo de predicao e treinado de duas formas:
           TSTR  com o corpus sintetico
           TRTR  com as celulas observadas da referencia
       e testado nos mesmos participantes da referencia: prediz a atencao das
       celulas ocultas de cada cena, aloca a renderizacao e mede a metrica com
       a atencao verdadeira. Diferenca relativa:

           d = M(TSTR) / M(TRTR) - 1

       Equivalencia pelo TOST (Schuirmann, 1987), alfa = 5%: o intervalo de
       confianca de 90% de d, por bootstrap sobre participantes e sementes,
       deve caber em +-MARGEM (fixada antes dos experimentos). Inclui o tamanho
       do corpus sintetico e a linha de base sem personalizacao.

    C. ROBUSTEZ NUMERICA
       Atencao nao positiva recusada, inviabilidade detectada, piso respeitado.

    D. REGIME DE RECURSO
       O mesmo teste do bloco B ao longo da capacidade por objeto.

    B.1 cena so de celulas ocultas; B.2 semelhanca direta das alocacoes
        (variacao total e correlacao de postos); B.3 sensibilidade da margem;
        B.4 cenas de popularidade homogenea (proposto x so popularidade).

    E. CONTROLES NEGATIVOS
       O mesmo protocolo com geradores de controle (so popularidade, objetos
       permutados, i.i.d.), na cena original e numa cena so de celulas ocultas.
       Verifica que o teste reprova dados sem estrutura.

    F. VALIDACAO POR PARTICIPANTE
       Dobras por participante: gerador e modelos TRTR reestimados sem os
       participantes de teste, que entram so com o proprio historico.

    G. ESCOLHA DE S
       Analise paralela por permutacao so com as celulas observadas, e o teste
       repetido com outro S.

    I. TESTES DE DISTRIBUICAO
       pMSE (Woo et al. 2009; Snoke et al. 2018), MMD (Gretton et al. 2012),
       distancia ao registro mais proximo (copia) e analise paralela por
       permutacao (Horn 1965; Buja e Eyuboglu 1992) para estrutura individual.

    H. FIDELIDADE ESPECTRAL
       Fracao do residuo nos S primeiros componentes: referencia, proposto,
       so popularidade e nulo por permutacao, com faixa de 5-95%.

MODELOS DE PREDICAO
-------------------
    item-item         corrige a media do objeto pelos desvios do usuario nos
                      objetos mais correlacionados (Sarwar et al., 2001)
    kNN usuario       usuarios do corpus mais correlacionados com o
                      participante (Herlocker et al., 1999)
    NMF               completa o corpus por ANLS (predicao.py) e aplica a
                      correcao por vizinhanca sobre a matriz completada
    MF com vieses     fatoracao com vieses de usuario e objeto, minimos
                      quadrados alternados (Koren et al., 2009)

USO
---
    python3 src/validacao.py
    python3 src/validacao.py --usuarios 300 --repeticoes 12
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gerador import (S_FATORES, _cov_pares, ajusta_atencao, carrega_referencia,
                     discretiza_por_quantis, gera_atencao)
from scipy.stats import rankdata
from metricas import (
    CAP_POR_OBJETO,
    LIM_DOWNLINK,
    P_TH,
    InviavelError,
    aloca,
    aloca_uniforme,
    meta_imersao,
    qoe_cac,
    verifica_viabilidade,
)
from predicao import completa, rmse_mascarado

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "figure.constrained_layout.use": True, "pdf.fonttype": 42,
})
AZUL, LARANJA, VERDE, ROXO, CINZA = "#3b6ea5", "#c2703d", "#4c8055", "#7a4f8a", "#8a8a8a"

DOBRAS = 5            # validacao por participante
MARGENS = (0.5, 1.0, 2.0, 5.0)  # sensibilidade da margem, %
MARGEM = 2.0          # margem de equivalencia, % (TOST), fixada antes dos experimentos
REAMOSTRAS = 2000     # reamostras bootstrap (participantes e sementes)
ALFA = 0.05           # nivel de significancia do TOST
K_VIZ = 20            # vizinhos nos modelos de vizinhanca
PISO, TETO = 0.5, 5.5  # faixa das predicoes (atencao > 0)
TAMANHOS = (10, 30, 100, 300, 1000, 1500)
CAPACIDADES = (15.5, 16, 17, 18, 20, 25, 30, 40)


# --------------------------------------------------------------------------- #
# Dados de referencia
# --------------------------------------------------------------------------- #
def historico_referencia(gd, tr):
    """Mascara de train.txt com os valores de gd.txt: -1 onde a celula e oculta."""
    mascara = tr >= 0
    return np.where(mascara, gd, -1.0), mascara


def media_mascarada(M, mascara):
    soma, cont = (np.where(mascara, M, 0.0)).sum(0), mascara.sum(0)
    geral = M[mascara].mean()
    return np.where(cont > 0, soma / np.maximum(cont, 1), geral)


def correlacao_objetos(M, mascara):
    """Correlacao entre objetos, por pares, so com celulas observadas."""
    if mascara.all():
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.nan_to_num(np.corrcoef(M.T))
    else:
        p = media_mascarada(M, mascara)
        R = np.where(mascara, M - p, 0.0)
        W = mascara.astype(float)
        cont = W.T @ W
        cov = (R.T @ R) / np.maximum(cont - 1, 1)
        var = ((R ** 2).T @ W) / np.maximum(cont - 1, 1)
        den = np.sqrt(var * var.T)
        C = np.where((cont >= 3) & (den > 0), cov / np.where(den > 0, den, 1.0), 0.0)
    np.fill_diagonal(C, 0.0)
    return C


# --------------------------------------------------------------------------- #
# Modelos de predicao: cada um recebe (M, mascara) e devolve prever(hist, objs)
# --------------------------------------------------------------------------- #
def _vizinhanca_itens(perfil, W):
    def prever(hist, objs):
        vistos = np.nonzero(hist >= 0)[0]
        if len(vistos) == 0:
            return perfil[objs]
        desvio = hist[vistos] - perfil[vistos]
        out = np.empty(len(objs))
        for i, n in enumerate(objs):
            w = W[n, vistos]
            top = np.argsort(np.abs(w))[-K_VIZ:]
            den = np.abs(w[top]).sum()
            out[i] = perfil[n] + (w[top] @ desvio[top] / den if den > 1e-9 else 0.0)
        return np.clip(out, PISO, TETO)
    return prever


def p2_item_item(M, mascara):
    return _vizinhanca_itens(media_mascarada(M, mascara), correlacao_objetos(M, mascara))


def p3_fatoracao(M, mascara, S=S_FATORES):
    K, _ = completa(np.where(mascara, M, 0.0).astype(float), mascara, metodo="anls", S=S)
    cheia = np.ones(K.shape, bool)
    return _vizinhanca_itens(K.mean(0), correlacao_objetos(K, cheia))


def knn_usuario(M, mascara):
    medias = np.array([M[v, mascara[v]].mean() for v in range(M.shape[0])])

    def prever(hist, objs):
        o = hist >= 0
        mu = hist[o].mean()
        sims = np.zeros(M.shape[0])
        for v in range(M.shape[0]):
            m = o & mascara[v]
            if m.sum() >= 3:
                x, y = hist[m] - hist[m].mean(), M[v, m] - M[v, m].mean()
                den = np.sqrt((x @ x) * (y @ y))
                sims[v] = (x @ y) / den if den > 0 else 0.0
        out = np.empty(len(objs))
        for i, n in enumerate(objs):
            cand = np.nonzero(mascara[:, n] & (sims > 0))[0]
            top = cand[np.argsort(-sims[cand])][:K_VIZ]
            out[i] = mu if len(top) == 0 else mu + (
                sims[top] @ (M[top, n] - medias[top])) / sims[top].sum()
        return np.clip(out, PISO, TETO)
    return prever


def mf_vieses(M, mascara, k=8, reg=0.5, iteracoes=30, semente=0):
    rng = np.random.default_rng(semente)
    n_u, n_o = M.shape
    mu = M[mascara].mean()
    P = 0.1 * rng.standard_normal((n_u, k + 1))  # ultima coluna: vies do usuario
    Q = 0.1 * rng.standard_normal((n_o, k + 1))  # ultima coluna: vies do objeto
    P[:, k] = 0.0
    Q[:, k] = 0.0
    for _ in range(iteracoes):
        for u in range(n_u):
            o = mascara[u]
            A = np.c_[Q[o, :k], np.ones(o.sum())]
            P[u] = np.linalg.solve(A.T @ A + reg * np.eye(k + 1),
                                   A.T @ (M[u, o] - mu - Q[o, k]))
        for n in range(n_o):
            o = mascara[:, n]
            A = np.c_[P[o, :k], np.ones(o.sum())]
            Q[n] = np.linalg.solve(A.T @ A + reg * np.eye(k + 1),
                                   A.T @ (M[o, n] - mu - P[o, k]))

    def prever(hist, objs):
        o = hist >= 0
        A = np.c_[Q[o, :k], np.ones(o.sum())]  # fold-in: fatores e vies do usuario
        x = np.linalg.solve(A.T @ A + reg * np.eye(k + 1), A.T @ (hist[o] - mu - Q[o, k]))
        return np.clip(mu + Q[objs, :k] @ x[:k] + x[k] + Q[objs, k], PISO, TETO)
    return prever


MODELOS = {
    "item–item": p2_item_item,
    "kNN usuário": knn_usuario,
    "NMF": p3_fatoracao,
    "MF com vieses": mf_vieses,
}
MET = {
    "Meta-Imersao": lambda K, P, dl, bep, lat: meta_imersao(K, P, dl, bep),
    "QoE-CAC": qoe_cac,
}


# --------------------------------------------------------------------------- #
# Avaliacao
# --------------------------------------------------------------------------- #
def condicoes_rede(n, semente=1234):
    """Condicoes de rede fixas por usuario, iguais nos dois bracos."""
    r = np.random.default_rng(semente)
    return {
        "downlink": r.uniform(*LIM_DOWNLINK, size=n),
        "bep": 10.0 ** r.uniform(-8, -2, size=n),
        "latencia": sum(r.uniform(a, b, size=n)
                        for a, b in [(0.5, 2), (1, 3), (2, 10), (5, 10)]),
    }


def atencao_predita(prever, hist, cenas):
    """Atencao usada na alocacao: historico onde observado, predicao no resto."""
    out = []
    for u, objs in enumerate(cenas):
        h = hist[u, objs]
        out.append(np.where(h >= 0, h, prever(hist[u], objs)))
    return out


def avalia_u(K_pred, gd, cenas, rede, metrica, cap=CAP_POR_OBJETO):
    """Metrica por participante (vetor de NU valores)."""
    vals = []
    for u, objs in enumerate(cenas):
        P = aloca(K_pred[u], len(objs) * cap, P_TH)
        vals.append(metrica(gd[u, objs], P, rede["downlink"][u], rede["bep"][u],
                            rede["latencia"][u]))
    return np.array(vals)


def avalia(K_pred, gd, cenas, rede, metrica, cap=CAP_POR_OBJETO):
    return float(avalia_u(K_pred, gd, cenas, rede, metrica, cap).mean())


def avalia_uniforme(gd, cenas, rede, metrica, cap=CAP_POR_OBJETO):
    vals = []
    for u, objs in enumerate(cenas):
        P = aloca_uniforme(len(objs), len(objs) * cap, P_TH)
        vals.append(metrica(gd[u, objs], P, rede["downlink"][u], rede["bep"][u],
                            rede["latencia"][u]))
    return float(np.mean(vals))


def tost(trtr_u, tstr_su, reamostras=REAMOSTRAS, semente=2024):
    """TOST com IC por bootstrap sobre participantes E sementes.

    trtr_u  : (NU,)    metrica por participante, braco TRTR
    tstr_su : (S, NU)  metrica por semente e participante, braco TSTR
    d = media(TSTR) / media(TRTR) - 1, em %. O IC de (1 - 2 alfa) vem dos
    percentis da distribuicao bootstrap; equivalente se cabe em +-MARGEM.
    O gerador nao e reestimado a cada reamostra (fica fixo).
    """
    trtr_u, tstr_su = np.asarray(trtr_u), np.atleast_2d(tstr_su)
    S, NU = tstr_su.shape
    ponto = 100 * (tstr_su.mean() / trtr_u.mean() - 1)
    rng = np.random.default_rng(semente)
    iu = rng.integers(0, NU, size=(reamostras, NU))
    js = rng.integers(0, S, size=(reamostras, S))
    num = np.array([tstr_su[js[b]][:, iu[b]].mean() for b in range(reamostras)])
    den = trtr_u[iu].mean(axis=1)
    d = 100 * (num / den - 1)
    lo, hi = np.percentile(d, [100 * ALFA, 100 * (1 - ALFA)])
    return float(ponto), float(lo), float(hi), bool(lo > -MARGEM and hi < MARGEM)


# --------------------------------------------------------------------------- #
# Geradores de controle (bloco E)
# --------------------------------------------------------------------------- #
def controle_so_popularidade(modelo, n, rng):
    p, B, lam, sigma = modelo[:4]
    a = modelo[5] if len(modelo) > 5 else 1.0
    peso = modelo[6] if len(modelo) > 6 else 1.0
    dp = np.sqrt((lam ** 2) @ (B ** 2) + (sigma * peso) ** 2)  # mesma variancia, sem estrutura
    from gerador import discretiza_por_quantis
    return discretiza_por_quantis(a * p + dp * rng.standard_normal((n, len(p))))


def controle_permutado(modelo, n, rng):
    K = gera_atencao(modelo, n, rng)
    return K[:, rng.permutation(K.shape[1])]


def controle_iid(modelo, n, rng):
    from gerador import discretiza_por_quantis
    return discretiza_por_quantis(rng.standard_normal((n, len(modelo[0]))))


# --------------------------------------------------------------------------- #
# Analises adicionais
# --------------------------------------------------------------------------- #
def similaridade_alocacao(Ka, Kb, cenas, cap=CAP_POR_OBJETO):
    """Por participante: distancia de variacao total entre as fracoes de
    capacidade alocadas (0 = identicas, 1 = disjuntas) e correlacao de postos
    entre as alocacoes."""
    vt, rho = [], []
    for u, objs in enumerate(cenas):
        Pa = aloca(np.asarray(Ka[u], float), len(objs) * cap, P_TH)
        Pb = aloca(np.asarray(Kb[u], float), len(objs) * cap, P_TH)
        vt.append(0.5 * np.abs(Pa / Pa.sum() - Pb / Pb.sum()).sum())
        x, y = rankdata(Pa), rankdata(Pb)
        x, y = x - x.mean(), y - y.mean()
        den = np.sqrt((x @ x) * (y @ y))
        rho.append((x @ y) / den if den > 0 else np.nan)
    return np.array(vt), np.array(rho)


def fracao_residual(A, S):
    """Fracao da variancia residual (apos a media por objeto) nos S primeiros
    componentes, matriz completa."""
    s = np.linalg.svd(A - A.mean(0), compute_uv=False) ** 2
    return float(s[:S].sum() / s.sum())


def permuta_colunas(M, mascara, rng):
    """Embaralha, em cada objeto, os valores observados entre os participantes:
    preserva a distribuicao de cada objeto e destroi a estrutura entre eles."""
    Z = M.copy()
    for n in range(M.shape[1]):
        idx = np.nonzero(mascara[:, n])[0]
        Z[idx, n] = M[rng.permutation(idx), n]
    return Z


def analise_paralela(M, mascara, reps=300, semente=11):
    """Analise paralela por permutacao sobre a covariancia por pares.
    Devolve: componentes consecutivos acima do percentil 97,5 do nulo, fracao
    nos 4 primeiros (referencia) e mediana do nulo."""
    def frac(Z):
        _, C = _cov_pares(Z, mascara)
        w = np.sort(np.linalg.eigvalsh(C))[::-1]
        return w / np.trace(C)
    rng = np.random.default_rng(semente)
    fr = frac(M)
    nulo = np.array([frac(permuta_colunas(M, mascara, rng)) for _ in range(reps)])
    acima = fr[:20] > np.percentile(nulo, 97.5, axis=0)[:20]
    k = int(np.argmin(acima)) if not acima.all() else 20
    return k, float(fr[:4].sum()), float(np.median(nulo[:, :4].sum(1)))


def _logistica_ridge(X, y, lam=10.0, it=50):
    """Regressao logistica com penalizacao L2, por Newton (sem dependencias)."""
    X = np.c_[np.ones(len(X)), X]
    w = np.zeros(X.shape[1])
    pen = lam * np.eye(X.shape[1]); pen[0, 0] = 0.0
    for _ in range(it):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + pen @ w
        H = (X * (p * (1 - p))[:, None]).T @ X + pen
        w -= np.linalg.solve(H, g)
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))


def pmse_razao(Xr, Xs, rng, perm=40):
    """pMSE (Woo et al. 2009; Snoke et al. 2018) dividido pela media sob
    permutacao dos rotulos: ~1 = indistinguivel, >1 = distinguivel."""
    def pm(A, B):
        X = np.vstack([A, B]); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        y = np.r_[np.zeros(len(A)), np.ones(len(B))]
        return np.mean((_logistica_ridge(X, y) - len(B) / len(X)) ** 2)
    obs = pm(Xr, Xs); X = np.vstack([Xr, Xs]); n = len(Xr)
    nulo = [pm(*np.split(X[rng.permutation(len(X))], [n])) for _ in range(perm)]
    return obs / np.mean(nulo)


def mmd_valor_p(X, Y, rng, perm=300):
    """Teste de duas amostras por MMD com nucleo gaussiano e permutacao
    (Gretton et al. 2012); largura pela mediana das distancias."""
    Z = np.vstack([X, Y]); D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    K = np.exp(-D / np.median(D[D > 0])); n = len(X)
    f = lambda i: (K[np.ix_(i[:n], i[:n])].mean() + K[np.ix_(i[n:], i[n:])].mean()
                   - 2 * K[np.ix_(i[:n], i[n:])].mean())
    obs = f(np.arange(len(Z)))
    return (1 + sum(f(rng.permutation(len(Z))) >= obs for _ in range(perm))) / (1 + perm)


def dcr(Xr, Xs):
    """Distancia ao registro mais proximo (sintetico -> referencia) e a mesma
    distancia entre participantes da referencia, para comparacao."""
    d = np.sqrt(((Xs[:, None, :] - Xr[None, :, :]) ** 2).sum(-1)).min(1)
    dr = np.sqrt(((Xr[:, None, :] - Xr[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dr, np.inf)
    return float(d.mean()), float(dr.min(1).mean())


def estrutura_paralela(M, rng, reps=100):
    """Analise paralela por permutacao (Horn 1965; Buja e Eyuboglu 1992) na
    matriz completa: componentes consecutivos acima do percentil 97,5 do nulo
    e se a fracao nos 4 primeiros excede o percentil 97,5 do nulo."""
    cheia = np.ones(M.shape, bool)
    def frac(Z):
        s = np.linalg.svd(Z - Z.mean(0), compute_uv=False) ** 2
        return s / s.sum()
    fr = frac(M)
    nulo = np.array([frac(permuta_colunas(M, cheia, rng)) for _ in range(reps)])
    acima = fr[:10] > np.percentile(nulo, 97.5, axis=0)[:10]
    k = int(np.argmin(acima)) if not acima.all() else 10
    return k, bool(fr[:4].sum() > np.percentile(nulo[:, :4].sum(1), 97.5))


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
CORES = dict(zip(MODELOS, (AZUL, LARANJA, VERDE, ROXO)))


def _faixa(ax, x, linhas, rotulo, cor):
    m = np.array([l[0] for l in linhas])
    ax.plot(x, m, "o-", color=cor, ms=3, label=rotulo)
    ax.fill_between(x, [l[1] for l in linhas], [l[2] for l in linhas], color=cor, alpha=0.15)


def fig_robustez(corpus, escassez, ganho, saida):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for nome in MODELOS:
        _faixa(ax[0], TAMANHOS, corpus[nome], nome, CORES[nome])
        _faixa(ax[1], CAPACIDADES, escassez[nome], nome, CORES[nome])
    for a in ax:
        a.axhline(MARGEM, ls=":", c="k", lw=0.8); a.axhline(-MARGEM, ls=":", c="k", lw=0.8)
        a.axhline(0, c=CINZA, lw=0.5); a.set_ylim(-1.25 * MARGEM, 1.25 * MARGEM)
    ax[0].set_xscale("log"); ax[0].set_xlabel("usuários no corpus sintético")
    ax[0].set_ylabel("TSTR − TRTR (%)"); ax[0].set_title("(a)", fontsize=9)
    ax[0].legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    ax[1].set_xlabel("capacidade por objeto (K)"); ax[1].set_title("(b)", fontsize=9)
    a2 = ax[1].twinx()
    a2.plot(CAPACIDADES, ganho, "--", color=CINZA, lw=1)
    a2.set_ylabel("ganho sobre a uniforme (%)", color=CINZA)
    a2.tick_params(axis="y", colors=CINZA); a2.grid(False)
    fig.savefig(os.path.join(saida, "robustez.pdf")); plt.close(fig)


def fig_espectro(gd, modelo, saida, replicas=200):
    def frac(A):
        s = np.linalg.svd(A - A.mean(0), compute_uv=False) ** 2
        return s[:10] / s.sum()
    rng = np.random.default_rng(7)
    er = frac(gd)
    rep = np.array([frac(gera_atencao(modelo, gd.shape[0], rng)) for _ in range(replicas)])
    es = rep.mean(0)
    lo, hi = np.percentile(rep, [5, 95], axis=0)
    k = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ax.bar(k - 0.2, er, 0.4, label="dados de referência", color=AZUL)
    ax.bar(k + 0.2, es, 0.4, yerr=[es - lo, hi - es], capsize=2, ecolor="k",
           error_kw={"lw": 0.7}, label="sintético (faixa 5–95%)", color=LARANJA)
    ax.set_xticks(k); ax.set_xlabel("componente do resíduo")
    ax.set_ylabel("fração da variância"); ax.legend(frameon=False)
    fig.savefig(os.path.join(saida, "espectro_residuo.pdf")); plt.close(fig)
    return er, es


def fig_atencao(gd, sint, saida):
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.8))
    ordem = np.argsort(gd.mean(0))
    ax[0].plot(gd.mean(0)[ordem], lw=1.2, color=AZUL, label="referência")
    ax[0].plot(sint.mean(0)[ordem], lw=1.2, color=LARANJA, alpha=0.85, label="sintético")
    ax[0].set_xlabel("objeto (ordenado pela popularidade na referência)")
    ax[0].set_ylabel("atenção média"); ax[0].legend(frameon=False)
    niv = np.arange(1, 6)
    ax[1].bar(niv - 0.2, [(gd == v).mean() for v in niv], 0.4, color=AZUL)
    ax[1].bar(niv + 0.2, [(sint == v).mean() for v in niv], 0.4, color=LARANJA)
    ax[1].set_xticks(niv); ax[1].set_xlabel("nível de atenção")
    ax[1].set_ylabel("frequência")
    fig.savefig(os.path.join(saida, "distribuicao_atencao.pdf")); plt.close(fig)


def fig_rmse(dados, saida):
    rot = [d["rot"] for d in dados]
    x = np.arange(len(rot))
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(x - 0.2, [d["rmse_real"] for d in dados], 0.4, label="referência", color=AZUL)
    ax.bar(x + 0.2, [d["rmse_sint"] for d in dados], 0.4, label="sintético", color=LARANJA)
    ax.set_xticks(x); ax.set_xticklabels(rot)
    ax.set_ylabel("RMSE da atenção predita"); ax.legend(frameon=False)
    fig.savefig(os.path.join(saida, "rmse_mascarado.pdf")); plt.close(fig)


def fig_convergencia(hist_als, hist_anls, saida):
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(np.arange(1, len(hist_als) + 1), hist_als, "-", color=AZUL, label="ALS")
    ax.plot(np.arange(1, len(hist_anls) + 1), hist_anls, "-", color=VERDE, label="ANLS (não negativa)")
    ax.set_yscale("log"); ax.set_xlabel("iteração")
    ax.set_ylabel("função objetivo"); ax.legend(frameon=False)
    fig.savefig(os.path.join(saida, "convergencia_fatoracao.pdf")); plt.close(fig)


# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Valida o dataset sintetico.")
    ap.add_argument("--dados", default="dados")
    ap.add_argument("--figuras", default="figuras")
    ap.add_argument("--resultados", default="resultados")
    ap.add_argument("--usuarios", type=int, default=150)
    ap.add_argument("--repeticoes", type=int, default=8)
    ap.add_argument("--fatores", type=int, default=S_FATORES)
    args = ap.parse_args()

    for d in (args.figuras, args.resultados):
        os.makedirs(d, exist_ok=True)
    caminho_relatorio = os.path.join(args.resultados, "relatorio_validacao.txt")
    relatorio = Relatorio(caminho_relatorio)
    sys.stdout = relatorio

    gd, tr, cenas = carrega_referencia(args.dados)
    NU, NO = gd.shape
    hist, mascara = historico_referencia(gd, tr)
    rede = condicoes_rede(NU)
    modelo = ajusta_atencao(gd, args.fatores, mascara=mascara)  # so celulas observadas
    sintetico = lambda n, s: gera_atencao(modelo, n, np.random.default_rng(s))
    reg = []

    print("=" * 72)
    print("VALIDACAO DO DATASET SINTETICO")
    print(f"referencia: {NU} x {NO} | observadas {mascara.sum()} | ocultas {(~mascara).sum()}"
          f" | corpus sintetico: {args.usuarios} | {args.repeticoes} sementes")
    print(f"gerador estimado so nas celulas observadas: sigma {modelo[3]:.3f}, a {modelo[5]:.3f},"
          f" ruido por objeto de {modelo[6].min():.2f} a {modelo[6].max():.2f}")
    print("=" * 72)

    # ---- A. ensaio interno (diagnostico) ----------------------------------- #
    print("\n[A] ENSAIO INTERNO — mascarar, prever, medir (diagnostico; sintetico com o mesmo"
          " numero de usuarios da referencia)")
    print(f"    {'condicao':22s}{'RMSE ref':>11s}{'RMSE sint':>11s}{'MAE ref':>10s}"
          f"{'exato ref':>12s}")
    dados_rmse = []
    sint_ref = sintetico(args.usuarios, 0)
    sint_a = [sintetico(NU, 10 + s) for s in range(3)]   # mesmo numero de usuarios da referencia
    for metodo in ("als", "anls"):
        for frac in (0.2, 0.4):
            rr = [rmse_mascarado(gd, frac, metodo, semente=s, S=args.fatores)
                  for s in range(3)]
            rs = [rmse_mascarado(sint_a[s].astype(float), frac, metodo, semente=s,
                                 S=args.fatores) for s in range(3)]
            d = {"rot": f"{metodo.upper()}\n{int(frac * 100)}%",
                 "rmse_real": np.mean([x[0] for x in rr]),
                 "rmse_sint": np.mean([x[0] for x in rs])}
            dados_rmse.append(d)
            print(f"    {metodo.upper() + f' mascara {int(frac*100)}%':22s}"
                  f"{d['rmse_real']:11.4f}{d['rmse_sint']:11.4f}"
                  f"{np.mean([x[1] for x in rr]):10.4f}"
                  f"{100 * np.mean([x[2] for x in rr]):11.1f}%")
            reg.append(("ensaio", metodo, f"mascara={frac}", d["rmse_real"], d["rmse_sint"]))
    fig_rmse(dados_rmse, args.figuras)

    from predicao import als, anls
    mk = np.ones(gd.shape, bool)
    _, _, i_als = als(gd, mk, S=args.fatores)
    _, _, i_anls = anls(gd, mk, S=args.fatores)
    fig_convergencia(i_als["historico"], i_anls["historico"], args.figuras)
    for i in (i_als, i_anls):
        estado = "convergiu" if i["convergiu"] else "TETO ATINGIDO"
        print(f"    {i['metodo']:5s} {i['iteracoes']:4d} iteracoes | objetivo {i['objetivo']:.4f} | delta {i['delta_relativo']:.2e} | {estado}")
        reg.append(("convergencia", i["metodo"], estado, i["iteracoes"], i["objetivo"]))

    # ---- B. TSTR x TRTR ----------------------------------------------------- #
    print(f"\n[B] TSTR x TRTR — treina no sintetico, testa na referencia"
          f" (TOST, alfa {ALFA:.0%}, margem +-{MARGEM:.0f}%)")
    M_ref = np.where(mascara, gd, 0.0)
    pred_trtr = {m: atencao_predita(f(M_ref, mascara), hist, cenas) for m, f in MODELOS.items()}
    pred_tstr = {m: [] for m in MODELOS}           # corpus padrao, por semente
    for s in range(args.repeticoes):
        Ds = sintetico(args.usuarios, 100 + s)
        cheia = np.ones(Ds.shape, bool)
        for m, f in MODELOS.items():
            pred_tstr[m].append(atencao_predita(f(Ds.astype(float), cheia), hist, cenas))
        print(f"    semente {s + 1}/{args.repeticoes}", flush=True)

    ics = []
    for met_nome, met in MET.items():
        unif = avalia_uniforme(gd, cenas, rede, met)
        oraculo = avalia([gd[u, o] for u, o in enumerate(cenas)], gd, cenas, rede, met)
        print(f"\n  {met_nome}  (uniforme = {unif:.4f} | atencao verdadeira = {oraculo:.4f})")
        print(f"    {'modelo':15s}{'TRTR':>9s}{'TSTR':>9s}{'d':>8s}{'IC 90%':>20s}{'ganho':>8s}  TOST")
        for m in MODELOS:
            au = avalia_u(pred_trtr[m], gd, cenas, rede, met)
            bu = np.array([avalia_u(p, gd, cenas, rede, met) for p in pred_tstr[m]])
            a, b = au.mean(), bu.mean()
            md, lo, hi, ok = tost(au, bu)
            print(f"    {m:15s}{a:9.4f}{b:9.4f}{md:+7.2f}%   [{lo:+6.2f}; {hi:+6.2f}]%"
                  f"{100 * (a / unif - 1):7.1f}%  {'equivalente' if ok else 'NAO'}")
            reg.append(("tstr", met_nome, m, md, f"{lo:.4f};{hi:.4f};{int(ok)}"))
            ics.append((met_nome, m, lo, hi))
        perfil = media_mascarada(M_ref, mascara)
        sem_pers = avalia(atencao_predita(lambda h, o: perfil[o], hist, cenas), gd, cenas, rede, met)
        melhor = max(avalia(pred_trtr[m], gd, cenas, rede, met) for m in MODELOS)
        print(f"    {'sem personalizacao (media por objeto)':38s}{sem_pers:9.4f}"
              f"  ganho {100 * (sem_pers / unif - 1):5.1f}% | personalizacao acrescenta"
              f" {100 * (melhor / sem_pers - 1):+.2f}% (melhor modelo)")
        reg.append(("personalizacao", met_nome, "media por objeto", sem_pers,
                    f"acrescimo_melhor_modelo={100 * (melhor / sem_pers - 1):.4f}"))

    # ---- B.1 cena so de celulas ocultas (toda a alocacao depende da predicao)
    print("\n  B.1 cena so de celulas ocultas (toda a alocacao depende da predicao)")
    ocultas = [np.nonzero(~mascara[u])[0] for u in range(NU)]
    pred_trtr_o = {m: atencao_predita(f(M_ref, mascara), hist, ocultas) for m, f in MODELOS.items()}
    pred_tstr_o = {m: [] for m in MODELOS}
    for s in range(args.repeticoes):
        Ds = sintetico(args.usuarios, 100 + s)
        for m, f in MODELOS.items():
            pred_tstr_o[m].append(atencao_predita(f(Ds.astype(float), np.ones(Ds.shape, bool)),
                                                  hist, ocultas))
    perfil = media_mascarada(M_ref, mascara)
    for met_nome, met in MET.items():
        unif = avalia_uniforme(gd, ocultas, rede, met)
        print(f"    {met_nome}  (uniforme = {unif:.4f})")
        for m in MODELOS:
            au = avalia_u(pred_trtr_o[m], gd, ocultas, rede, met)
            bu = np.array([avalia_u(p, gd, ocultas, rede, met) for p in pred_tstr_o[m]])
            md, lo, hi, ok = tost(au, bu)
            print(f"      {m:15s}{au.mean():9.4f}{bu.mean():9.4f}{md:+7.2f}%   [{lo:+6.2f}; {hi:+6.2f}]%"
                  f"{100 * (au.mean() / unif - 1):7.1f}%  {'equivalente' if ok else 'NAO'}")
            reg.append(("tstr_ocultas", met_nome, m, md, f"{lo:.4f};{hi:.4f};{int(ok)}"))
        sp = avalia(atencao_predita(lambda h, o: perfil[o], hist, ocultas), gd, ocultas, rede, met)
        melhor = max(avalia(pred_trtr_o[m], gd, ocultas, rede, met) for m in MODELOS)
        print(f"      sem personalizacao {sp:9.4f} | personalizacao acrescenta "
              f"{100 * (melhor / sp - 1):+.2f}% (melhor modelo)")
        reg.append(("personalizacao_ocultas", met_nome, "media por objeto", sp,
                    f"acrescimo_melhor_modelo={100 * (melhor / sp - 1):.4f}"))

    # ---- B.2 semelhanca direta das alocacoes TSTR x TRTR
    print("\n  B.2 semelhanca direta das alocacoes (por participante; VT: variacao total"
          " entre as fracoes alocadas, 0 = identicas)")
    cenarios_b2 = (("cena original", cenas, pred_trtr, pred_tstr),
                   ("celulas ocultas", ocultas, pred_trtr_o, pred_tstr_o))
    for nome_c, cena_c, p_trtr, p_tstr in cenarios_b2:
        print(f"    {nome_c}")
        print(f"    {'modelo':15s}{'VT media':>10s}{'VT p95':>9s}{'VT max':>9s}{'rho media':>11s}"
              f"{'VT TRTR x verdadeira':>23s}{'VT TRTR x uniforme':>21s}")
        oraculo_K = [gd[u, o] for u, o in enumerate(cena_c)]
        unif_K = [np.ones(len(o)) for o in cena_c]
        for m in MODELOS:
            vts, rhos = [], []
            for p in p_tstr[m]:
                vt, rho = similaridade_alocacao(p, p_trtr[m], cena_c)
                vts.append(vt); rhos.append(rho)
            vts, rhos = np.concatenate(vts), np.concatenate(rhos)
            vt_or, _ = similaridade_alocacao(p_trtr[m], oraculo_K, cena_c)
            vt_un, _ = similaridade_alocacao(p_trtr[m], unif_K, cena_c)
            print(f"    {m:15s}{vts.mean():10.4f}{np.percentile(vts, 95):9.4f}{vts.max():9.4f}"
                  f"{np.nanmean(rhos):11.3f}{vt_or.mean():23.4f}{vt_un.mean():21.4f}")
            reg.append(("alocacao", nome_c, m, float(vts.mean()),
                        f"p95={np.percentile(vts, 95):.4f};max={vts.max():.4f};rho={np.nanmean(rhos):.4f};"
                        f"vt_verdadeira={vt_or.mean():.4f};vt_uniforme={vt_un.mean():.4f}"))

    # ---- B.4 cenas de popularidade homogenea --------------------------------- #
    print("\n  B.4 cenas de popularidade homogenea (celulas ocultas de cada participante,"
          " agrupadas em 5 faixas de popularidade; a popularidade quase nao decide)")
    faixas = np.array_split(np.argsort(perfil), 5)
    cen_h, usu_h = [], []
    for u in range(NU):
        for fx in faixas:
            o = np.array([n for n in fx if not mascara[u, n]])
            if len(o) >= 4:
                cen_h.append(o); usu_h.append(u)
    usu_h = np.array(usu_h)
    gd_h, hist_h = gd[usu_h], hist[usu_h]
    rede_h = {k: v[usu_h] for k, v in rede.items()}
    met = MET["Meta-Imersao"]
    un_h = avalia_uniforme(gd_h, cen_h, rede_h, met)
    or_h = avalia([gd_h[i, o] for i, o in enumerate(cen_h)], gd_h, cen_h, rede_h, met)
    sp_h = avalia(atencao_predita(lambda h, o: perfil[o], hist_h, cen_h), gd_h, cen_h, rede_h, met)
    print(f"    {len(cen_h)} cenas, {np.mean([len(o) for o in cen_h]):.1f} objetos em media | ganho sobre a"
          f" uniforme: atencao verdadeira {100 * (or_h / un_h - 1):+.1f}%, media por objeto"
          f" {100 * (sp_h / un_h - 1):+.1f}%")
    reg.append(("homogenea", "Meta-Imersao", "verdadeira;media", 100 * (or_h / un_h - 1),
                f"{100 * (sp_h / un_h - 1):.4f}"))
    for m, f in MODELOS.items():
        au = avalia_u(atencao_predita(f(M_ref, mascara), hist_h, cen_h), gd_h, cen_h, rede_h, met)
        linha = []
        for g_nome, g in (("proposto", gera_atencao), ("so popularidade", controle_so_popularidade)):
            bu = []
            for s in range(max(3, args.repeticoes // 2)):
                Ds = g(modelo, args.usuarios, np.random.default_rng(100 + s))
                p = atencao_predita(f(Ds.astype(float), np.ones(Ds.shape, bool)), hist_h, cen_h)
                bu.append(avalia_u(p, gd_h, cen_h, rede_h, met))
            md, lo, hi, ok = tost(au, np.array(bu))
            linha.append(f"{g_nome} {md:+.2f}% [{lo:+.2f}; {hi:+.2f}] {'EQ' if ok else 'NAO'}")
            reg.append(("homogenea", g_nome, m, md, f"{lo:.4f};{hi:.4f};{int(ok)}"))
        print(f"    {m:15s} personalizacao {100 * (au.mean() / sp_h - 1):+.2f}% | " + " | ".join(linha))

    # ---- B.3 sensibilidade da margem
    print("\n  B.3 sensibilidade da margem (modelos x metricas equivalentes, de 8)")
    pior = max(max(abs(lo), abs(hi)) for _, _, lo, hi in ics)
    print("    " + " | ".join(f"+-{d:.1f}%: {sum(lo > -d and hi < d for _, _, lo, hi in ics)}/8"
                             for d in MARGENS) + f" | pior extremo de IC: {pior:.2f}%")
    reg.append(("margem", "-", "pior extremo de IC", pior, ""))

    print("\n    tamanho do corpus sintetico (Meta-Imersao)")
    met = MET["Meta-Imersao"]
    base = {m: avalia_u(pred_trtr[m], gd, cenas, rede, met) for m in MODELOS}
    corpus = {m: [] for m in MODELOS}
    for n in TAMANHOS:
        linha = []
        for m, f in MODELOS.items():
            b = []
            for s in range(args.repeticoes):
                Ds = sintetico(n, 200 + s)
                p = atencao_predita(f(Ds.astype(float), np.ones(Ds.shape, bool)), hist, cenas)
                b.append(avalia_u(p, gd, cenas, rede, met))
            r = tost(base[m], np.array(b))
            corpus[m].append(r)
            linha.append(f"{m.split()[0]} {r[0]:+.2f}")
            reg.append(("corpus", "Meta-Imersao", f"{m};N={n}", r[0], f"{r[1]:.4f};{r[2]:.4f};{int(r[3])}"))
        print(f"      N={n:5d}: " + " | ".join(linha), flush=True)

    # ---- C. robustez -------------------------------------------------------- #
    print("\n[C] ROBUSTEZ NUMERICA")
    testes = []
    try:
        aloca(np.array([1.0, 0.0, 3.0]), 100.0)
        testes.append(("atencao nula recusada", False))
    except ValueError:
        testes.append(("atencao nula recusada", True))
    try:
        aloca(np.array([1.0, np.nan]), 100.0)
        testes.append(("NaN recusado", False))
    except ValueError:
        testes.append(("NaN recusado", True))
    try:
        aloca(np.array([1.0, 2.0, 3.0]), 40.0)
        testes.append(("inviabilidade detectada", False))
    except InviavelError:
        testes.append(("inviabilidade detectada", True))
    P = aloca(np.array([1.0, 2.0, 3.0]), 40.0, ajustar_se_inviavel=True)
    testes.append(("ajuste automatico do orcamento", bool((P >= P_TH - 1e-9).all())))
    Kt = np.array([5.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    P = aloca(Kt, len(Kt) * CAP_POR_OBJETO)
    testes.append(("piso respeitado com atencao desigual", bool((P >= P_TH - 1e-9).all())))
    testes.append(("orcamento respeitado", bool(P.sum() <= len(Kt) * CAP_POR_OBJETO + 1e-6)))
    K_anls, _ = completa(gd, np.random.default_rng(0).random(gd.shape) < 0.6, metodo="anls")
    testes.append(("ANLS sem atencao <= 0", bool((K_anls > 0).all())))
    v = verifica_viabilidade(10, 100.0)
    testes.append(("diagnostico de viabilidade", v["minimo"] == 150.0 and not v["viavel"]))
    for nome, ok in testes:
        print(f"    [{'OK ' if ok else 'FALHA'}] {nome}")
        reg.append(("robustez", "-", nome, float(ok), 0.0))

    # ---- D. regime de recurso ----------------------------------------------- #
    print("\n[D] REGIME DE RECURSO (Meta-Imersao, mesmos corpora do bloco B)")
    escassez, ganho = {m: [] for m in MODELOS}, []
    for cap in CAPACIDADES:
        unif = avalia_uniforme(gd, cenas, rede, met, cap)
        linha = []
        for m in MODELOS:
            au = avalia_u(pred_trtr[m], gd, cenas, rede, met, cap)
            bu = np.array([avalia_u(p, gd, cenas, rede, met, cap) for p in pred_tstr[m]])
            a = au.mean()
            r = tost(au, bu)
            escassez[m].append(r)
            if m == "item–item":
                ganho.append(100 * (a / unif - 1))
            linha.append(f"{m.split()[0]} {r[0]:+.2f}")
            reg.append(("escassez", "Meta-Imersao", f"{m};cap={cap}", r[0],
                        f"{r[1]:.4f};{r[2]:.4f};{int(r[3])};ganho={100 * (a / unif - 1):.4f}"))
        print(f"    cap={cap:5.1f}K  ganho {ganho[-1]:5.1f}% | " + " | ".join(linha))
    fig_robustez(corpus, escassez, ganho, args.figuras)

    # ---- E. controles negativos --------------------------------------------- #
    print(f"\n[E] CONTROLES NEGATIVOS (Meta-Imersao; TOST por modelo, margem +-{MARGEM:.0f}%)")
    geradores = {
        "proposto": gera_atencao,
        "so popularidade": controle_so_popularidade,
        "objetos permutados": controle_permutado,
        "i.i.d.": controle_iid,
    }
    for cena_nome, cena in (("cena original", cenas), ("celulas ocultas", ocultas)):
        print(f"  {cena_nome}")
        trtr_c = {m: avalia_u(atencao_predita(f(M_ref, mascara), hist, cena), gd, cena, rede, met)
                  for m, f in MODELOS.items()}
        for g_nome, g in geradores.items():
            res_g = []
            for m, f in MODELOS.items():
                bu = []
                for s in range(max(3, args.repeticoes // 2)):
                    Ds = g(modelo, args.usuarios, np.random.default_rng(500 + s))
                    p = atencao_predita(f(Ds.astype(float), np.ones(Ds.shape, bool)), hist, cena)
                    bu.append(avalia_u(p, gd, cena, rede, met))
                md, lo, hi, ok = tost(trtr_c[m], np.array(bu))
                res_g.append((md, lo, hi, ok))
                reg.append(("controle", cena_nome, f"{g_nome};{m}", md, f"{lo:.4f};{hi:.4f};{int(ok)}"))
            ds = [r[0] for r in res_g]
            pior = max(max(abs(r[1]), abs(r[2])) for r in res_g)
            print(f"    {g_nome:20s} d de {min(ds):+7.2f}% a {max(ds):+7.2f}% | equivalentes"
                  f" {sum(r[3] for r in res_g)}/4 | pior extremo de IC {pior:.2f}%", flush=True)

    # ---- F. validacao por participante ------------------------------------- #
    print(f"\n[F] VALIDACAO POR PARTICIPANTE ({DOBRAS} dobras: gerador e modelos TRTR"
          f" reestimados sem os participantes de teste)")
    dobras = np.array_split(np.random.default_rng(77).permutation(NU), DOBRAS)
    trtr_f = {(mt, m): np.zeros(NU) for mt in MET for m in MODELOS}
    tstr_f = {(mt, m): np.zeros((args.repeticoes, NU)) for mt in MET for m in MODELOS}
    for i, te in enumerate(dobras):
        tr_ = np.setdiff1d(np.arange(NU), te)
        mod_f = ajusta_atencao(gd[tr_], args.fatores, mascara=mascara[tr_])
        Mf = np.where(mascara[tr_], gd[tr_], 0.0)
        cen_te = [cenas[u] for u in te]
        rede_te = {k: v[te] for k, v in rede.items()}
        for m, f in MODELOS.items():
            pa = atencao_predita(f(Mf, mascara[tr_]), hist[te], cen_te)
            for mt, met_ in MET.items():
                trtr_f[(mt, m)][te] = avalia_u(pa, gd[te], cen_te, rede_te, met_)
        for s in range(args.repeticoes):
            Ds = gera_atencao(mod_f, args.usuarios, np.random.default_rng(300 + s))
            for m, f in MODELOS.items():
                pb = atencao_predita(f(Ds.astype(float), np.ones(Ds.shape, bool)), hist[te], cen_te)
                for mt, met_ in MET.items():
                    tstr_f[(mt, m)][s, te] = avalia_u(pb, gd[te], cen_te, rede_te, met_)
        print(f"    dobra {i + 1}/{DOBRAS} (teste: {len(te)} participantes)", flush=True)
    for mt in MET:
        print(f"  {mt}")
        for m in MODELOS:
            md, lo, hi, ok = tost(trtr_f[(mt, m)], tstr_f[(mt, m)])
            print(f"    {m:15s}{md:+7.2f}%   [{lo:+6.2f}; {hi:+6.2f}]%  {'equivalente' if ok else 'NAO'}")
            reg.append(("por_participante", mt, m, md, f"{lo:.4f};{hi:.4f};{int(ok)}"))

    # ---- G. escolha de S ------------------------------------------------------ #
    print("\n[G] ESCOLHA DE S (dimensao da estrutura individual)")
    k_obs, f_obs, n_obs = analise_paralela(gd, mascara)
    print(f"    so celulas observadas: {k_obs} componente(s) acima do percentil 97,5 do nulo"
          f" por permutacao | 4 primeiros: {100 * f_obs:.1f}% (nulo {100 * n_obs:.1f}%)")
    k_cmp, f_cmp, n_cmp = analise_paralela(gd, np.ones(gd.shape, bool))
    print(f"    matriz completa (so descritivo; inclui celulas de teste): {k_cmp} componente(s)"
          f" | 4 primeiros: {100 * f_cmp:.1f}% (nulo {100 * n_cmp:.1f}%)")
    reg.append(("escolha_S", "observadas", "componentes acima do nulo", k_obs, f"{f_obs:.4f};{n_obs:.4f}"))
    reg.append(("escolha_S", "completa", "componentes acima do nulo", k_cmp, f"{f_cmp:.4f};{n_cmp:.4f}"))
    print("    sensibilidade do teste a S (Meta-Imersao, d [IC 90%]):")
    variantes = {f"S={S_}": ajusta_atencao(gd, S_, mascara=mascara) for S_ in (2,) if S_ != args.fatores}
    variantes[f"S={args.fatores}"] = modelo
    for nome_v in sorted(variantes):
        mod_v, linha = variantes[nome_v], []
        for m, f in MODELOS.items():
            bu = []
            for s in range(args.repeticoes):
                Ds = gera_atencao(mod_v, args.usuarios, np.random.default_rng(100 + s))
                p = atencao_predita(f(Ds.astype(float), np.ones(Ds.shape, bool)), hist, cenas)
                bu.append(avalia_u(p, gd, cenas, rede, met))
            md, lo, hi, ok = tost(base[m], np.array(bu))
            linha.append(f"{m.split()[0]} {md:+.2f} [{lo:+.2f}; {hi:+.2f}]")
            reg.append(("sensibilidade_S", nome_v, m, md, f"{lo:.4f};{hi:.4f};{int(ok)}"))
        print(f"      {nome_v}: " + " | ".join(linha))
    print("      S=0 corresponde ao controle so popularidade (bloco E)")

    # ---- H. fidelidade espectral ---------------------------------------------- #
    print("\n[H] FIDELIDADE ESPECTRAL (matriz completa; fracao do residuo nos S primeiros"
          " componentes, media e faixa 5-95% de 200 replicas de 30 usuarios)")
    er, es = fig_espectro(gd, modelo, args.figuras)
    rng_h = np.random.default_rng(13)
    for S_ in sorted({2, args.fatores}):
        ref = fracao_residual(gd, S_)
        sint = [fracao_residual(gera_atencao(modelo, NU, rng_h), S_) for _ in range(200)]
        pop = [fracao_residual(controle_so_popularidade(modelo, NU, rng_h), S_) for _ in range(200)]
        nul = [fracao_residual(permuta_colunas(gd, np.ones(gd.shape, bool), rng_h), S_) for _ in range(200)]
        fx = lambda v: f"{100 * np.mean(v):5.1f}% [{100 * np.percentile(v, 5):4.1f}; {100 * np.percentile(v, 95):4.1f}]"
        print(f"    S={S_}: referencia {100 * ref:5.1f}% | proposto {fx(sint)} | so popularidade {fx(pop)}"
              f" | nulo por permutacao {fx(nul)}")
        reg.append(("espectro", f"S={S_}", "referencia;proposto;so_pop;nulo", ref,
                    f"{np.mean(sint):.4f};{np.mean(pop):.4f};{np.mean(nul):.4f}"))
    fig_atencao(gd, sint_ref, args.figuras)

    # ---- I. testes de distribuicao ------------------------------------------- #
    print("\n[I] TESTES DE DISTRIBUICAO (30 usuarios sinteticos x matriz de referencia completa)")
    rng_i = np.random.default_rng(21)
    geradores_i = {"proposto": gera_atencao, "so popularidade": controle_so_popularidade,
                   "i.i.d.": controle_iid}
    k_ref, ag_ref = estrutura_paralela(gd.astype(float), rng_i)
    print(f"    analise paralela na referencia: {k_ref} componente(s); estrutura agregada"
          f" {'detectada' if ag_ref else 'nao detectada'}")
    print(f"    {'gerador':16s}{'pMSE razao':>24s}{'MMD rejeita':>14s}{'DCR':>8s}"
          f"{'estrutura detectada':>22s}{'componentes':>13s}")
    for g_nome, g in geradores_i.items():
        pr, pv, dd, ks, ag = [], [], [], [], []
        for k in range(20):
            Xs = g(modelo, NU, rng_i).astype(float)
            pr.append(pmse_razao(gd, Xs, rng_i)); pv.append(mmd_valor_p(gd, Xs, rng_i))
            dd.append(dcr(gd, Xs)[0])
            kk, aa = estrutura_paralela(Xs, rng_i); ks.append(kk); ag.append(aa)
        pr, ks = np.array(pr), np.array(ks)
        print(f"    {g_nome:16s}{pr.mean():8.2f} [{np.percentile(pr, 5):.2f}; {np.percentile(pr, 95):.2f}]"
              f"{100 * np.mean(np.array(pv) < 0.05):12.0f}%{np.mean(dd):8.2f}"
              f"{100 * np.mean(ag):20.0f}%{ks.mean():13.1f}", flush=True)
        reg.append(("distribuicao", g_nome, "pmse;mmd_rej;dcr;estrutura;componentes", float(pr.mean()),
                    f"{np.mean(np.array(pv) < 0.05):.4f};{np.mean(dd):.4f};{np.mean(ag):.4f};{ks.mean():.4f}"))
    print(f"    DCR entre participantes da referencia: {dcr(gd, gd.astype(float))[1]:.2f}"
          " (DCR do sintetico menor que isso indicaria copia)")

    c = os.path.join(args.resultados, "validacao.csv")
    with open(c, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bloco", "metrica", "condicao", "valor", "valor_2"])
        w.writerows(reg)
    print(f"\nresultados: {c}")
    print(f"relatorio:  {caminho_relatorio}")
    for nome in sorted(os.listdir(args.figuras)):
        if nome.endswith(".pdf"):
            print(f"figura:     {os.path.join(args.figuras, nome)}")


if __name__ == "__main__":
    main()
    if isinstance(sys.stdout, Relatorio):
        sys.stdout.fecha()
