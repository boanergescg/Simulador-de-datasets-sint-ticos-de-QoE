"""
Gerador de datasets sinteticos de QoE para ambientes imersivos.

Produz o registro completo exigido pelas duas metricas: atencao usuario-objeto,
latencia motion-to-photon, taxas de rede, confiabilidade do enlace e capacidade
de renderizacao. Cada variavel tem faixa ancorada em recomendacao ITU-T ou em
literatura de referencia (ver FAIXAS).

MODELO DE ATENCAO
-----------------
Dados de referencia: UOAL (Du et al.), 30 participantes x 96 objetos, niveis
derivados de rastreamento ocular (PSM, Xu et al.). Perfil global forte (91,9% da
variancia no 1o componente, Spearman medio +0,916) e estrutura individual de
baixa dimensao: quatro componentes do residuo acima do percentil 97,5 de 500
replicas de ruido, com 46,1% da variancia residual contra 29,3% sob ruido.

    s[k,n] = a p[n] + SUM_j f[k,j] B[j,n] + sigma w[n] eps[k,n]
    f[k,j] ~ N(0, lam_j^2),  eps ~ N(0, 1)

ESTIMACAO -- so com as celulas indicadas por `mascara`:
    p        media por objeto sobre as celulas observadas
    B, lam   autovetores e autovalores da covariancia residual calculada por
             pares de objetos (cada par so com quem observou os dois); nada e
             imputado
    w        intensidade do ruido por objeto: raiz da variancia idiossincratica
             (diagonal da covariancia menos a parte explicada pelos S fatores),
             normalizada para media quadratica 1; ha objetos em que os
             participantes concordam muito e outros em que divergem
    a, sigma calibrados por momentos simulados sobre a mesma mascara, para
             reproduzir o Spearman medio entre usuarios e a fracao da variancia
             residual nos S primeiros componentes

    validacao:   mascara = celulas observadas; as ocultas (teste) ficam de fora
    caso de uso: mascara = None (matriz completa); nao ha teste a proteger

p e B descrevem os objetos e ficam FIXOS; so os escores f dos usuarios sao
sorteados, o que permite escalar de 30 para N usuarios.

Discretizacao em 5 niveis POR QUANTIS, por usuario. Largura igual estaria
errada: os escores brutos sao assimetricos a direita e empilhariam cerca de 66%
dos objetos no nivel 1.

A atencao e EXOGENA: propriedade do par (usuario, objeto), obtida de
rastreamento ocular. Nao depende de latencia nem de FPS.

LATENCIA
--------
Motion-to-photon decomposta por componente, nao sorteada de uma faixa unica e
nao derivada como inverso da taxa de transmissao (dimensionalmente incoerente):

    I = t_sensor + t_processamento + t_renderizacao + t_rede

Isso da orcamento explicito e permite varrer cenarios por componente.

FAIXAS
------
    atencao ............. 1 a 5        escala de 5 niveis (P.800.1)
    MTP ................. 7 a 20 ms    limite de conforto em AR/VR
      sensor ............ 0,5 a 2 ms
      processamento ..... 1 a 3 ms
      renderizacao ...... 2 a 10 ms
      rede .............. 5 a 10 ms    (5G/Edge)
    downlink ............ 10 a 100 Mbit/s
    BEP uplink .......... 1e-8 a 1e-2  (amostrada em escala log)
    FPS ................. 60 a 120     (>= 90 Hz recomendado)
    dispositivos ........ 30% high, 50% mid, 20% low

O tipo de dispositivo condiciona FPS e orcamento de renderizacao; sortear as
duas coisas de forma independente tornaria o campo decorativo.

REPRODUTIBILIDADE
-----------------
Sementes acopladas: uma semente mestra deriva sementes independentes para
atencao, mascaras, rede e dispositivo, de modo que alterar um bloco nao desloca
os demais. O arquivo de configuracao registra tudo que foi usado.

USO
---
    python3 src/gerador.py --usuarios 30
    python3 src/gerador.py --usuarios 150 --semente 7 --cenario congestionado
"""

import argparse
import csv
import json
import os

import numpy as np

N_NIVEIS = 5
S_FATORES = 4  # componentes acima do nulo, medido na referencia

MTP_COMPONENTES = {
    "sensor": (0.5, 2.0),
    "processamento": (1.0, 3.0),
    "renderizacao": (2.0, 10.0),
    "rede": (5.0, 10.0),
}
CENARIOS = {
    "normal": {},
    "congestionado": {"rede": (10.0, 25.0)},
    "edge": {"rede": (2.0, 5.0)},
    "render_limitado": {"renderizacao": (10.0, 25.0)},
}
DOWNLINK = (10.0, 100.0)
BEP_LOG = (-8.0, -2.0)
DISPOSITIVOS = {  # tipo: (prob, faixa de FPS, multiplicador de orcamento)
    "high-end": (0.30, (100, 120), 1.20),
    "mid-range": (0.50, (75, 100), 1.00),
    "low-end": (0.20, (60, 75), 0.85),
}
MAPA = 100.0


# --------------------------------------------------------------------------- #
def sementes(mestra):
    """Sementes independentes por bloco, derivadas de uma mestra."""
    ss = np.random.SeedSequence(mestra).spawn(5)
    nomes = ("atencao", "historico", "cena", "rede", "dispositivo")
    return {n: np.random.default_rng(s) for n, s in zip(nomes, ss)}


# --------------------------------------------------------------------------- #
# Atencao
# --------------------------------------------------------------------------- #
def _cov_pares(M, mascara):
    """Perfil por objeto e covariancia residual por pares, so com celulas observadas."""
    n_obj = M.shape[1]
    p = np.array([M[mascara[:, n], n].mean() for n in range(n_obj)])
    R = np.where(mascara, M - p, 0.0)
    W = mascara.astype(float)
    cont = W.T @ W
    C = np.where(cont >= 2, (R.T @ R) / np.maximum(cont - 1, 1), 0.0)
    return p, C


def _fracao_estrutura(M, mascara, S):
    """Fracao da variancia residual nos S primeiros componentes."""
    _, C = _cov_pares(M, mascara)
    w = np.sort(np.linalg.eigvalsh(C))[::-1]
    return float(np.clip(w[:S], 0, None).sum() / np.trace(C))


def _postos_medios(x):
    """Postos medios com empates (mesmo resultado de scipy.stats.rankdata)."""
    ordem = np.argsort(x, kind="mergesort")
    xs = x[ordem]
    inicio = np.r_[True, xs[1:] != xs[:-1]]
    grupo = np.cumsum(inicio) - 1
    pos = np.arange(1, len(x) + 1, dtype=float)
    soma = np.bincount(grupo, weights=pos)
    cont = np.bincount(grupo)
    r = np.empty(len(x))
    r[ordem] = (soma / cont)[grupo]
    return r


def _spearman_pares(M, mascara):
    """Spearman medio entre pares de usuarios, sobre os objetos que ambos observaram."""
    vals = []
    for u in range(M.shape[0]):
        for v in range(u + 1, M.shape[0]):
            m = mascara[u] & mascara[v]
            if m.sum() >= 5:
                x, y = _postos_medios(M[u, m]), _postos_medios(M[v, m])
                x, y = x - x.mean(), y - y.mean()
                den = np.sqrt((x @ x) * (y @ y))
                if den > 0:
                    vals.append((x @ y) / den)
    return float(np.mean(vals))


def ajusta_atencao(gd_ref, n_fatores=S_FATORES, mascara=None, reps=12, semente=0,
                   voltas=4):
    """Estima o modelo de atencao so com as celulas de `mascara`.

    Devolve (p, B, lam, sigma, fracao_estrutura, a, w). Os cinco primeiros
    mantem a posicao da versao anterior (gera_dataset e config os leem por
    indice); w e a intensidade do ruido por objeto.
    """
    if n_fatores >= min(gd_ref.shape):
        raise ValueError(f"n_fatores={n_fatores} alto demais para {gd_ref.shape}")
    gd_ref = np.asarray(gd_ref, float)
    mascara = np.ones(gd_ref.shape, bool) if mascara is None else np.asarray(mascara, bool)
    S = n_fatores

    p, C = _cov_pares(gd_ref, mascara)
    w, V = np.linalg.eigh(C)
    w, V = w[::-1], V[:, ::-1]
    lam, B = np.sqrt(np.clip(w[:S], 0, None)), V[:, :S].T
    psi = np.clip(np.diag(C) - (lam ** 2) @ (B ** 2), 1e-3, None)
    peso = np.sqrt(psi / psi.mean())
    alvo_e = _fracao_estrutura(gd_ref, mascara, S)
    alvo_s = _spearman_pares(gd_ref, mascara)

    def media(fun, sigma, a):
        rng = np.random.default_rng(semente)
        return np.mean([fun(gera_atencao((p, B, lam, sigma, 0.0, a, peso), mascara.shape[0], rng),
                            mascara) for _ in range(reps)])

    def bissecta(f, lo, hi, cresce):
        for _ in range(18):
            meio = 0.5 * (lo + hi)
            if (f(meio) > 0) == cresce:
                hi = meio
            else:
                lo = meio
        return 0.5 * (lo + hi)

    sigma, a = 0.45, 1.0
    for _ in range(voltas):
        sigma = bissecta(lambda x: media(lambda Z, m: _fracao_estrutura(Z, m, S), x, a)
                         - alvo_e, 0.05, 3.0, cresce=False)
        a = bissecta(lambda x: media(_spearman_pares, sigma, x) - alvo_s,
                     0.3, 4.0, cresce=True)
    return p, B, lam, float(sigma), alvo_e, float(a), peso


def discretiza_por_quantis(escores, n_niveis=N_NIVEIS):
    out = np.zeros(escores.shape, dtype=int)
    for k in range(escores.shape[0]):
        for lvl, idx in enumerate(np.array_split(np.argsort(escores[k]), n_niveis)):
            out[k, idx] = lvl + 1
    return out


def gera_atencao(modelo, n_usuarios, rng):
    p, B, lam, sigma = modelo[:4]
    a = modelo[5] if len(modelo) > 5 else 1.0
    peso = modelo[6] if len(modelo) > 6 else 1.0
    f = rng.standard_normal((n_usuarios, len(lam))) * lam
    return discretiza_por_quantis(
        a * p + f @ B + sigma * peso * rng.standard_normal((n_usuarios, len(p)))
    )


# --------------------------------------------------------------------------- #
# Mascaras
# --------------------------------------------------------------------------- #
def marginais(train_ref, cenas_ref):
    """Marginais reais dos dois conjuntos.

    Historico e cena sao recortes DISTINTOS: o historico e o que o usuario ja
    viu (entrada da predicao), a cena e o que ele ve agora (onde se aloca).
    Confundi-los dobra a metrica e infla o ganho aparente.
    """
    obs = train_ref >= 0
    freq_cena = np.zeros(train_ref.shape[1])
    for c in cenas_ref:
        freq_cena[c] += 1
    return (obs.sum(1), obs.mean(0), [len(c) for c in cenas_ref],
            freq_cena / len(cenas_ref))


def amostra_conjunto(n_usuarios, n_objetos, contagens, freq, rng):
    """Conjunto por usuario, reproduzindo duas marginais: objetos por usuario e
    frequencia de aparicao de cada objeto."""
    prob = np.clip(freq, 1e-6, None)
    prob = prob / prob.sum()
    if n_usuarios == len(contagens):
        tamanhos = rng.permutation(contagens)  # evita variancia de reamostragem
    else:
        tamanhos = rng.choice(contagens, size=n_usuarios, replace=True)
    return [
        rng.choice(n_objetos, size=int(min(max(t, 1), n_objetos)), replace=False, p=prob)
        for t in tamanhos
    ]


# --------------------------------------------------------------------------- #
# Rede, dispositivo, ambiente
# --------------------------------------------------------------------------- #
def gera_dispositivos(n_usuarios, rng):
    tipos = list(DISPOSITIVOS)
    escolha = rng.choice(tipos, size=n_usuarios, p=[DISPOSITIVOS[t][0] for t in tipos])
    fps = np.array([rng.uniform(*DISPOSITIVOS[t][1]) for t in escolha])
    mult = np.array([DISPOSITIVOS[t][2] for t in escolha])
    return escolha, fps, mult


def gera_latencia(n_usuarios, rng, cenario="normal"):
    faixas = dict(MTP_COMPONENTES)
    faixas.update(CENARIOS[cenario])
    parcelas = {c: rng.uniform(*f, size=n_usuarios) for c, f in faixas.items()}
    return sum(parcelas.values()), parcelas


def gera_rede(n_usuarios, rng):
    return (
        rng.uniform(*DOWNLINK, size=n_usuarios),
        10.0 ** rng.uniform(*BEP_LOG, size=n_usuarios),  # log-uniforme
    )


# --------------------------------------------------------------------------- #
def gera_dataset(gd_ref, train_ref, cenas_ref, n_usuarios, semente=42,
                 n_fatores=S_FATORES, cenario="normal"):
    rng = sementes(semente)
    n_objetos = gd_ref.shape[1]

    modelo = ajusta_atencao(gd_ref, n_fatores)
    K = gera_atencao(modelo, n_usuarios, rng["atencao"])

    cont_h, freq_h, cont_c, freq_c = marginais(train_ref, cenas_ref)
    historico = amostra_conjunto(n_usuarios, n_objetos, cont_h, freq_h, rng["historico"])
    cenas = amostra_conjunto(n_usuarios, n_objetos, cont_c, freq_c, rng["cena"])

    tipo, fps, mult = gera_dispositivos(n_usuarios, rng["dispositivo"])
    latencia, parcelas = gera_latencia(n_usuarios, rng["rede"], cenario)
    downlink, bep = gera_rede(n_usuarios, rng["rede"])
    poi = rng["cena"].uniform(0, MAPA, size=(n_objetos, 2))

    linhas = []
    for u in range(n_usuarios):
        hist_u = set(historico[u])
        for t, n in enumerate(cenas[u]):
            linhas.append({
                "user_id": u, "poi_id": int(n), "time_unit": t,
                "attention": int(K[u, n]), "device_tier": tipo[u],
                "mtp_latency_ms": round(float(latencia[u]), 4),
                "lat_sensor_ms": round(float(parcelas["sensor"][u]), 4),
                "lat_proc_ms": round(float(parcelas["processamento"][u]), 4),
                "lat_render_ms": round(float(parcelas["renderizacao"][u]), 4),
                "lat_rede_ms": round(float(parcelas["rede"][u]), 4),
                "downlink_mbps": round(float(downlink[u]), 4),
                "uplink_bep": float(f"{bep[u]:.6e}"),
                "fps": round(float(fps[u]), 2),
                "orcamento_mult": float(mult[u]),
                "poi_x": round(float(poi[n, 0]), 3),
                "poi_y": round(float(poi[n, 1]), 3),
                "no_historico": int(n in hist_u),
            })

    config = {
        "n_usuarios": n_usuarios, "n_objetos": n_objetos, "semente": semente,
        "n_fatores": n_fatores, "cenario": cenario,
        "sigma_residual": modelo[3], "variancia_nos_fatores": modelo[4],
        "amplitude_perfil": modelo[5],
        "faixas": {"mtp_componentes": MTP_COMPONENTES, "downlink": DOWNLINK,
                   "bep_log10": BEP_LOG, "dispositivos": {k: v[1] for k, v in DISPOSITIVOS.items()}},
    }
    return linhas, K, historico, cenas, config


def escreve(saida, linhas, K, historico, cenas, config, prefixo="sint"):
    os.makedirs(saida, exist_ok=True)
    caminhos = []

    c = os.path.join(saida, f"{prefixo}_dataset.csv")
    with open(c, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    caminhos.append(c)

    c = os.path.join(saida, f"{prefixo}_atencao.txt")
    np.savetxt(c, K, fmt="%d")
    caminhos.append(c)

    for nome, conj in (("historico", historico), ("cenas", cenas)):
        c = os.path.join(saida, f"{prefixo}_{nome}.txt")
        with open(c, "w", encoding="utf-8") as f:
            for x in conj:
                f.write(",".join(map(str, sorted(x))) + "\n")
        caminhos.append(c)

    c = os.path.join(saida, f"{prefixo}_config.json")
    with open(c, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    caminhos.append(c)
    return caminhos


def carrega_referencia(dados="dados"):
    gd = np.loadtxt(os.path.join(dados, "gd.txt"))
    tr = np.loadtxt(os.path.join(dados, "train.txt"))
    cenas = [np.array([int(x) for x in l.strip().split(",")])
             for l in open(os.path.join(dados, "experiment.txt")) if l.strip()]
    return gd, tr, cenas


def main():
    ap = argparse.ArgumentParser(description="Gera dataset sintetico de QoE.")
    ap.add_argument("--usuarios", type=int, default=30)
    ap.add_argument("--fatores", type=int, default=S_FATORES)
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--cenario", default="normal", choices=list(CENARIOS))
    ap.add_argument("--dados", default="dados")
    ap.add_argument("--saida", default="resultados")
    args = ap.parse_args()

    gd, tr, cenas_ref = carrega_referencia(args.dados)
    linhas, K, hist, cenas, config = gera_dataset(
        gd, tr, cenas_ref, args.usuarios, args.semente, args.fatores, args.cenario
    )
    print(f"{args.usuarios} usuarios x {gd.shape[1]} objetos | S={args.fatores} "
          f"| cenario={args.cenario} | semente={args.semente}")
    print(f"{len(linhas)} registros | cena media "
          f"{np.mean([len(c) for c in cenas]):.1f} objetos "
          f"| variancia nos fatores {100 * config['variancia_nos_fatores']:.1f}%")
    for c in escreve(args.saida, linhas, K, hist, cenas, config):
        print(f"  {c}")


if __name__ == "__main__":
    main()
