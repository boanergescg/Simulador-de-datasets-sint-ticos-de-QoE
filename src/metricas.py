"""
Metricas de QoE e alocacao de recursos.

Duas metricas DISTINTAS, ambas publicadas e revisadas. Aqui sao instrumentos de
medida, nao objeto de estudo: servem para avaliar se o dataset sintetico
substitui o real.

    Meta-Imersao (Du et al., IEEE JSAC 41(1), 2023)

        MI(u) = T(R_u) * T(1 - E_u) * SUM_n K_nu * ln( P_nu / P_th )

    QoE-CAC (Santos et al., SBrT 2025)

        QoE-CAC(u) = T(1 - I_u) * T(R_u) * T(1 - E_u)
                     * SUM_n K_nu * ln( P_nu / P_th )

A diferenca esta no fator de latencia T(1 - I_u), presente so na QoE-CAC. As
duas consomem o mesmo dataset; rodar as duas da evidencia independente.

NORMALIZACAO
------------
T(.) e min-max com LIMITES EXTERNOS FIXOS, nunca calculados dos dados. Limites
derivados dos dados fariam real e sintetico caírem na mesma escala por
construcao, tornando a comparacao tautologica.

ROBUSTEZ NUMERICA (Melhoria III)
--------------------------------
  - atencao negativa ou nula invalida a ponderacao e faz a metrica falhar em
    silencio: `valida_atencao` recusa, `aloca` trata o piso;
  - orcamento insuficiente para o piso torna o problema inviavel:
    `verifica_viabilidade` detecta e `aloca` oferece ajuste iterativo do
    orcamento em vez de correcao em passada unica;
  - fatores de normalizacao sao saturados em [0,1]: valor fora da faixa nao
    inverte o sinal da metrica.
"""

import numpy as np

# --------------------------------------------------------------------------- #
# Limites de normalizacao. Externos e fixos.
# --------------------------------------------------------------------------- #
LIM_LATENCIA = (7.0, 20.0)  # ms, motion-to-photon
LIM_DOWNLINK = (10.0, 100.0)  # Mbit/s
LIM_BEP = (1e-8, 1e-2)  # probabilidade de erro de bit no uplink
P_TH = 15.0  # piso de renderizacao (K)
CAP_POR_OBJETO = 20.0  # orcamento por objeto (K)


class InviavelError(ValueError):
    """Orcamento insuficiente para atender o piso em todos os objetos."""


def tau(t, lim):
    """Min-max com limites externos, saturada em [0, 1]."""
    lo, hi = lim
    return float(np.clip((np.asarray(t, float) - lo) / (hi - lo), 0.0, 1.0))


def tau_bep(e):
    """Normalizacao da BEP em escala logaritmica.

    A BEP varre seis ordens de grandeza; em escala linear quase toda a massa
    ficaria colada no extremo superior.
    """
    lo, hi = LIM_BEP
    x = (np.log10(np.clip(e, lo, hi)) - np.log10(hi)) / (np.log10(lo) - np.log10(hi))
    return float(np.clip(x, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Robustez
# --------------------------------------------------------------------------- #
def valida_atencao(K, nome="K"):
    """Recusa atencao nao positiva, NaN ou infinita.

    Atencao <= 0 zera ou inverte o peso do objeto no somatorio; NaN se propaga
    silenciosamente ate a metrica. Falhar aqui e preferivel a devolver numero.
    """
    K = np.asarray(K, float)
    if not np.isfinite(K).all():
        raise ValueError(f"{nome} contem NaN ou infinito")
    if (K <= 0).any():
        raise ValueError(f"{nome} contem {int((K <= 0).sum())} valor(es) <= 0")
    return K


def verifica_viabilidade(n_objetos, orcamento, p_th=P_TH):
    """Orcamento minimo para viabilidade e folga disponivel."""
    minimo = p_th * n_objetos
    return {
        "viavel": orcamento >= minimo,
        "minimo": minimo,
        "folga": orcamento - minimo,
        "orcamento_sugerido": max(orcamento, minimo * 1.05),
    }


# --------------------------------------------------------------------------- #
# Alocacao (Melhoria II)
# --------------------------------------------------------------------------- #
def aloca(K, orcamento, p_th=P_TH, ajustar_se_inviavel=False):
    """Alocacao otima: P_n = max(K_n / mu*, P_th).

    Maximiza SUM_n K_n ln(P_n / P_th) sob SUM_n P_n <= orcamento e P_n >= P_th.
    Objetivo concavo, restricoes lineares: water-filling, otimo global.

    O ajuste e ITERATIVO, nao em passada unica: a cada rodada os objetos que
    ficariam abaixo do piso sao fixados nele e o multiplicador e recalculado
    sobre os restantes. Uma passada unica deixaria objetos abaixo do piso.

    ajustar_se_inviavel: se o orcamento nao cobre o piso, eleva ao minimo com
    5% de folga em vez de levantar excecao.
    """
    K = valida_atencao(K)
    n = len(K)
    v = verifica_viabilidade(n, orcamento, p_th)
    if not v["viavel"]:
        if not ajustar_se_inviavel:
            raise InviavelError(
                f"orcamento {orcamento:.2f} < minimo {v['minimo']:.2f} "
                f"({n} objetos x piso {p_th})"
            )
        orcamento = v["orcamento_sugerido"]

    no_piso = np.zeros(n, bool)
    for _ in range(n + 1):
        livres = ~no_piso
        if not livres.any():
            return np.full(n, p_th)
        mu = K[livres].sum() / (orcamento - p_th * no_piso.sum())
        P = np.where(no_piso, p_th, K / mu)
        if P.min() >= p_th - 1e-9:
            return P
        no_piso[np.argmin(np.where(livres, P, np.inf))] = True
    return P


def aloca_uniforme(n, orcamento, p_th=P_TH):
    """Linha de base: divide o orcamento igualmente, ignorando a atencao."""
    if not verifica_viabilidade(n, orcamento, p_th)["viavel"]:
        raise InviavelError(f"orcamento {orcamento:.2f} insuficiente para {n} objetos")
    return np.full(n, orcamento / n)


# --------------------------------------------------------------------------- #
# Metricas
# --------------------------------------------------------------------------- #
def somatorio_atencao(K, P, p_th=P_TH):
    """SUM_n K_n ln(P_n / P_th). Nucleo comum as duas metricas."""
    K = valida_atencao(K)
    P = np.asarray(P, float)
    if (P < p_th - 1e-9).any():
        raise ValueError("alocacao abaixo do piso: logaritmo negativo")
    return float((K * np.log(P / p_th)).sum())


def meta_imersao(K, P, downlink, bep, p_th=P_TH):
    """Meta-Imersao (Du et al.). Sem termo de latencia."""
    return tau(downlink, LIM_DOWNLINK) * tau_bep(bep) * somatorio_atencao(K, P, p_th)


def qoe_cac(K, P, downlink, bep, latencia, p_th=P_TH):
    """QoE-CAC (Santos et al.). Inclui o fator de latencia.

    T(1 - I) implementado como 1 - T(I): latencia alta reduz a QoE.
    """
    return (
        (1.0 - tau(latencia, LIM_LATENCIA))
        * tau(downlink, LIM_DOWNLINK)
        * tau_bep(bep)
        * somatorio_atencao(K, P, p_th)
    )


METRICAS = {
    "Meta-Imersao": lambda K, P, dl, bep, lat: meta_imersao(K, P, dl, bep),
    "QoE-CAC": qoe_cac,
}
