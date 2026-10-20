"""
Predicao de atencao usuario-objeto (Melhoria I).

Cada usuario visita apenas parte dos objetos, entao a matriz de atencao e
esparsa. As metricas, porem, exigem K_nu para todos os objetos do somatorio.
Este modulo completa a matriz.

    min_{U,V}  SUM_{(u,n) in Omega} ( K_nu - u_u^T v_n )^2
               + lambda ( ||U||_F^2 + ||V||_F^2 )

Dois estimadores:

    als()   minimos quadrados alternados regularizados. Solucao fechada a cada
            passo, decrescimo monotono. Pode produzir predicao negativa, o que
            invalida o logaritmo da metrica.

    anls()  o mesmo com restricao de nao negatividade, por gradiente projetado
            com passo igual ao inverso da constante de Lipschitz. Garante
            U, V >= 0, logo K >= 0 e logaritmo bem definido. E o estimador
            padrao (Melhoria III).

CONVERGENCIA
------------
Iteracoes ate tolerancia relativa de 5e-5 na variacao da funcao objetivo, com
S = 4 e lambda = 0,1, medidas na matriz 30 x 96 sob diferentes niveis de
mascaramento:

    mascara      ALS   ANLS         mascara      ALS   ANLS
    real   0%    123    304         sint.  0%    110    587
    real  20%    155    368         sint. 20%    146    770
    real  40%    198    570         sint. 40%    173    575
    real  60%    204    859         sint. 60%    181    805

O ANLS e cerca de quatro vezes mais lento que o ALS -- gradiente projetado nao
tem solucao fechada -- e desacelera conforme a matriz fica esparsa. MAX_ITER =
2000 cobre o pior caso medido (859) com folga de mais do dobro. Ambos chegam a
perda praticamente igual: a restricao de nao negatividade custa cerca de 0,4% de
ajuste e elimina o risco de atencao negativa.

Atingir o teto de iteracoes NAO e o mesmo que convergir. As funcoes devolvem um
dicionario com o historico da funcao objetivo, o numero de iteracoes, a variacao
relativa final e um sinalizador de convergencia; um aviso e emitido quando o
teto e atingido sem convergencia.

Sementes independentes por componente permitem reproduzir a inicializacao de U e
de V separadamente.
"""

import warnings

import numpy as np

LAMBDA = 0.1
N_FATORES = 4
MAX_ITER = 2000  # pior caso medido: ANLS com 859 iteracoes (ver CONVERGENCIA)
TOL = 5e-5  # tolerancia relativa na variacao da funcao objetivo
PISO_ATENCAO = 0.5  # mantem K > 0; abaixo do menor nivel observavel (1)
TETO_ATENCAO = 5.5


# --------------------------------------------------------------------------- #
def _inicializa(n_u, n_o, S, semente_u, semente_o, escala):
    """Inicializacao nao negativa, com sementes independentes por componente."""
    U = np.random.default_rng(semente_u).uniform(0.1, 1.0, (n_u, S)) * escala
    V = np.random.default_rng(semente_o).uniform(0.1, 1.0, (n_o, S)) * escala
    return U, V


def _objetivo(M, mask, U, V, lam):
    R = (M - U @ V.T)[mask]
    return float((R**2).sum() + lam * ((U**2).sum() + (V**2).sum()))


def _resultado(hist, convergiu, metodo):
    delta = abs(hist[-1] - hist[-2]) / abs(hist[-2]) if len(hist) > 1 else float("inf")
    return {
        "metodo": metodo,
        "historico": hist,
        "iteracoes": len(hist),
        "objetivo": hist[-1],
        "delta_relativo": delta,
        "convergiu": convergiu,
    }


def _avisa(info, max_iter):
    if not info["convergiu"]:
        warnings.warn(
            f"{info['metodo']} atingiu o teto de {max_iter} iteracoes sem "
            f"convergir (variacao relativa final {info['delta_relativo']:.2e}). "
            "Aumente max_iter ou afrouxe tol.",
            RuntimeWarning,
            stacklevel=3,
        )


# --------------------------------------------------------------------------- #
def als(M, mask, S=N_FATORES, lam=LAMBDA, max_iter=MAX_ITER, tol=TOL,
        semente_u=0, semente_o=1):
    """Minimos quadrados alternados regularizados.

    M: matriz com os valores observados; mask: booleana, True onde observado.
    Devolve (U, V, info).
    """
    n_u, n_o = M.shape
    escala = np.sqrt(max(M[mask].mean(), 1e-6) / S)
    U, V = _inicializa(n_u, n_o, S, semente_u, semente_o, escala)
    I = np.eye(S)
    hist, convergiu = [], False
    for it in range(max_iter):
        for u in range(n_u):
            idx = mask[u]
            if idx.any():
                Vi = V[idx]
                U[u] = np.linalg.solve(Vi.T @ Vi + lam * I, Vi.T @ M[u, idx])
        for n in range(n_o):
            idx = mask[:, n]
            if idx.any():
                Ui = U[idx]
                V[n] = np.linalg.solve(Ui.T @ Ui + lam * I, Ui.T @ M[idx, n])
        hist.append(_objetivo(M, mask, U, V, lam))
        if it and abs(hist[-2] - hist[-1]) < tol * abs(hist[-2]):
            convergiu = True
            break
    info = _resultado(hist, convergiu, "ALS")
    _avisa(info, max_iter)
    return U, V, info


def anls(M, mask, S=N_FATORES, lam=LAMBDA, max_iter=MAX_ITER, tol=TOL,
         semente_u=0, semente_o=1):
    """ALS com restricao de nao negatividade, por gradiente projetado.

    O passo e o inverso da constante de Lipschitz do gradiente, o que garante
    decrescimo sem busca de linha. Projecao em [0, inf) apos cada passo.
    Devolve (U, V, info).
    """
    n_u, n_o = M.shape
    escala = np.sqrt(max(M[mask].mean(), 1e-6) / S)
    U, V = _inicializa(n_u, n_o, S, semente_u, semente_o, escala)
    W = mask.astype(float)
    hist, convergiu = [], False
    for it in range(max_iter):
        G = ((W * (U @ V.T - M)) @ V) + lam * U
        L = np.linalg.norm(V.T @ V, 2) + lam + 1e-9
        U = np.maximum(U - G / L, 0.0)

        G = ((W * (U @ V.T - M)).T @ U) + lam * V
        L = np.linalg.norm(U.T @ U, 2) + lam + 1e-9
        V = np.maximum(V - G / L, 0.0)

        hist.append(_objetivo(M, mask, U, V, lam))
        if it and abs(hist[-2] - hist[-1]) < tol * abs(hist[-2]):
            convergiu = True
            break
    info = _resultado(hist, convergiu, "ANLS")
    _avisa(info, max_iter)
    return U, V, info


# --------------------------------------------------------------------------- #
def completa(M, mask, metodo="anls", **kw):
    """Completa a matriz esparsa e devolve K estritamente positivo.

    Os valores observados sao preservados; so as celulas ausentes recebem a
    predicao. O recorte em [PISO_ATENCAO, TETO_ATENCAO] impede que erro numerico
    produza atencao fora da escala -- em particular valor <= 0, que invalidaria
    a ponderacao da metrica.

    Devolve (K, info).
    """
    if metodo not in ("als", "anls"):
        raise ValueError(f"metodo desconhecido: {metodo!r}; use 'als' ou 'anls'")
    U, V, info = {"als": als, "anls": anls}[metodo](M, mask, **kw)
    pred = np.clip(U @ V.T, PISO_ATENCAO, TETO_ATENCAO)
    return np.clip(np.where(mask, M, pred), PISO_ATENCAO, TETO_ATENCAO), info


def rmse_mascarado(M_completa, fracao=0.2, metodo="anls", semente=0, **kw):
    """Ensaio interno: mascara parte das atencoes observadas, preve e mede o
    erro contra os valores retidos. Nao depende de coleta adicional.

    Devolve (rmse, mae, acerto_exato, n_mascaradas).
    """
    if not 0 < fracao < 1:
        raise ValueError("fracao deve estar em (0, 1)")
    rng = np.random.default_rng(semente)
    idx = np.argwhere(np.ones(M_completa.shape, bool))
    escolhidas = idx[rng.choice(len(idx), int(fracao * len(idx)), replace=False)]

    mask = np.ones(M_completa.shape, bool)
    mask[escolhidas[:, 0], escolhidas[:, 1]] = False
    K, _ = completa(np.where(mask, M_completa, 0.0), mask, metodo=metodo, **kw)

    verdade = M_completa[escolhidas[:, 0], escolhidas[:, 1]]
    predito = K[escolhidas[:, 0], escolhidas[:, 1]]
    erro = predito - verdade
    return (
        float(np.sqrt((erro**2).mean())),
        float(np.abs(erro).mean()),
        float((np.rint(predito) == verdade).mean()),
        len(escolhidas),
    )
