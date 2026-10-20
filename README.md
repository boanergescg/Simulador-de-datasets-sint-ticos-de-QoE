# Simulador de datasets sintéticos de QoE para ambientes imersivos

Gerador de dados sintéticos de atenção usuário–objeto, com as variáveis de rede
das métricas, para avaliar esquemas de alocação de recursos em realidade virtual
e aumentada.

Coletar atenção por rastreamento ocular exige laboratório, equipamento e
participantes, e cada novo cenário exige nova coleta. O gerador é estimado sobre
dados de referência (UOAL) e validado contra eles; validado, permite **prever** o
comportamento de esquemas de alocação em cenários ainda não coletados. Não
substitui o experimento com participantes.

## Validação

Teste supervisionado: cada modelo de predição é treinado com o corpus sintético
(TSTR, *train on synthetic, test on real*) e com as células observadas dos dados
de referência (TRTR), e testado nos mesmos participantes da referência. Nenhuma
célula de teste entra no ajuste dos parâmetros do gerador nem no treino dos
modelos. A equivalência é decidida pelo TOST (α = 5%, margem de ±2%), com
intervalos por *bootstrap* sobre participantes e sementes. A validação mede
também a semelhança direta das alocações, a validação por participante (5
dobras), controles negativos e testes de fidelidade (espectro do resíduo,
análise paralela, pMSE, MMD e distância ao registro mais próximo). Os resultados
saem de `src/validacao.py`, em `resultados/validacao.csv` e
`resultados/relatorio_validacao.txt`, e as figuras em `figuras/`.

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 ou superior. O núcleo usa NumPy, SciPy e Matplotlib; o caso de uso
usa também NetworkX, OSMnx e PyProj. Antes de rodar, coloque os dados de
referência em `dados/` (ver [Dados](#dados)).

## Uso

```bash
python src/gerador.py --usuarios 30       # gera o dataset sintético
python src/validacao.py                   # valida e produz as figuras
python src/caso_uso.py --modo pe          # caso de uso (baixa a malha do OSM)
```

O gerador aceita `--usuarios`, `--fatores`, `--semente` e `--cenario`
(`normal`, `congestionado`, `edge`, `render_limitado`). A validação aceita
`--usuarios` (tamanho do corpus sintético) e `--repeticoes` (sementes por
condição).

Saídas em `resultados/` (dataset, matriz de atenção, conjuntos, configuração,
tabelas e relatórios em texto) e `figuras/` (PDFs vetoriais). Ambas as pastas são geradas
pelo código e não são versionadas. A malha viária baixada pelo caso de uso fica
em cache em `dados/*.graphml`, também não versionada.

## Estrutura

```
dados/                   NÃO versionada; colocar aqui os arquivos de Dados
  gd.txt, train.txt,     dados de referência (30 × 96), da fonte original
  experiment.txt
  pois_pirenopolis.csv   pontos turísticos do caso de uso (ver Dados)
src/
  metricas.py   Meta-Imersão, QoE-CAC, alocação por water-filling
  predicao.py   ALS regularizado e fatoração não negativa
  gerador.py    geração do dataset sintético completo
  validacao.py  teste supervisionado (TSTR x TRTR), controles e figuras
  caso_uso.py   roteiros turísticos: experiência × equidade
figuras/      saída, PDF vetorial
resultados/   saída, CSV e JSON
```

## Métricas de avaliação

Duas métricas distintas, ambas publicadas, usadas aqui como **instrumentos de
medida** — o objeto sob teste é o dataset sintético, não as métricas.

**Meta-Imersão** (Du et al., IEEE JSAC 41(7), 2023)

```
MI(u) = T(R_u) · T(1 − E_u) · Σ_n K_nu · ln( P_nu / P_th )
```

**QoE-CAC** (Santos et al., SBrT 2025)

```
QoE-CAC(u) = T(1 − I_u) · T(R_u) · T(1 − E_u) · Σ_n K_nu · ln( P_nu / P_th )
```

A diferença é o fator de latência `T(1 − I_u)`, presente apenas na QoE-CAC.
Cada métrica sonda um conjunto parcialmente distinto de variáveis, de modo que
concordância entre elas é evidência independente.

A normalização `T(·)` usa limites externos fixos, nunca calculados dos dados:
limites derivados dos dados fariam referência e sintético cair na mesma escala
por construção, tornando a comparação tautológica.

## Modelo de geração

**Atenção.** Nos dados de referência, 91,9% da variância está no primeiro
componente e o Spearman médio entre participantes é +0,916: há um perfil global
de popularidade de objetos, e cada participante é esse perfil mais um desvio de
baixa dimensão. Contra um nulo por permutação, a matriz completa tem dois
componentes do resíduo acima do percentil 97,5, e os quatro primeiros
concentram 46,1% da variância residual (nulo: 31,5%).

```
s[k,n] = a · p[n] + Σ_j f[k,j] · B[j,n] + σ · w[n] · ε[k,n]
```

Estimação só com as células indicadas por uma máscara: `p` é a média por objeto
sobre as células observadas; `B` e as variâncias de `f` vêm dos autovetores da
covariância residual calculada por pares de objetos, sem imputação; `w` é a
intensidade do ruído por objeto (raiz da variância idiossincrática,
normalizada); `a` e `σ` são calibrados por momentos simulados sobre a mesma
máscara. Usa-se `S = 4`, com sensibilidade para `S = 2`. Na validação, a
máscara são as células observadas; no caso de uso, a matriz completa. `p` e `B`
descrevem os objetos e ficam fixos, o que permite escalar de 30 para N usuários.
Os escores são discretizados em cinco níveis por quantis, por usuário.

**Latência.** Motion-to-photon decomposta em sensor, processamento,
renderização e rede, cada componente com faixa própria. Não é sorteada de uma
faixa única nem derivada como inverso da taxa de transmissão, o que seria
dimensionalmente incoerente.

**Faixas.** MTP de 7 a 20 ms, downlink de 10 a 100 Mbit/s, BEP de 1e−8 a 1e−2 em
escala logarítmica, FPS de 60 a 120 com 90 Hz como referência, dispositivos em
30/50/20. Ancoradas em recomendação ITU-T e em literatura de referência; a
fundamentação normativa detalhada está no texto do trabalho.

## Limitações

O gerador é estimado e testado sobre os mesmos 30 participantes, em células
distintas; a confirmação definitiva exige uma amostra independente. Só a
atenção é validada: as variáveis de rede seguem faixas da literatura, sem dados
de referência. Os dados de
referência foram coletados com imagens estáticas, conteúdo selecionado e
visualização em monitor; o gerador reproduz a atenção sob esse protocolo. O
teste não distingue o gerador de um que reproduz só a popularidade (controle
negativo), e a validação cobre quatro modelos e duas métricas.

## Caso de uso

Roteiros turísticos a pé no centro histórico de Pirenópolis (GO), sobre a
malha de pedestres do OpenStreetMap, com 11 pontos turísticos. O caso de uso
separa **preferência** do visitante, que escolhe a rota, de **atenção** visual,
que reparte a renderização entre os pontos visitados. Um parâmetro λ ∈ [0, 1]
desloca a escolha dos pontos mais procurados para os menos procurados, e o
código mede o custo dessa desconcentração em experiência imersiva.

Os resultados saem de `src/caso_uso.py`, em `resultados/` e `figuras/`. As
conclusões são verificadas com 60 e 120 minutos e sem os pontos que não têm
avaliações.

A popularidade dos pontos vem do número de avaliações públicas no Google Maps
(coleta de 13/10/2026), usado como *proxy* de visitação. O campo de rede é
simulado, por ausência de dado público de cobertura por município, e os
resultados são promediados sobre 30 réplicas dele.

```bash
python src/caso_uso.py --modo pe --tempo 60          # outro tempo disponível
python src/caso_uso.py --modo pe --excluir-sem-avaliacao
```

Use `--resultados` e `--figuras` para gravar análises de sensibilidade em
pastas próprias, sem sobrescrever o cenário principal.

## Dados

Os dados de referência derivam do conjunto UOAL (Du et al.), construído a
partir do dataset PSM (Xu et al.). **Esses arquivos não são redistribuídos
neste repositório.** Para reproduzir os resultados, obtenha-os do código
oficial de Du et al. (*Attention-aware Resource Allocation and QoE Analysis
for Metaverse xURLLC Services*) e copie para `dados/`:

| Arquivo em `dados/` | Origem no código de Du et al. |
|---|---|
| `gd.txt` | `Seg2Rating/gd.txt` (matriz completa 30 × 96) |
| `train.txt` | `Main/train.txt` (matriz esparsa) |
| `experiment.txt` | `Algorithm2/experiment.txt` (objetos de cena por usuário) |

O conjunto UOAL completo está disponível a partir do repositório
`HongyangDu/User-Object-Attention-Level` no GitHub. Consulte os termos de uso
das fontes originais.

O arquivo `dados/pois_pirenopolis.csv`, usado no caso de uso, também não é
redistribuído, porque contém dados do Google Maps. Ele tem as colunas `nome`,
`categoria`, `avaliacoes`, `lat`, `lon` e `nota`, separadas por ponto e vírgula;
linhas iniciadas por `#` são comentários. Os 11 pontos do centro histórico de
Pirenópolis e a data da coleta (13/10/2026) estão descritos no artigo.

## Referências

- Xu, Y.; Gao, S.; Wu, J.; Li, N.; Yu, J. *Personalized saliency and its
  prediction*. IEEE TPAMI 41(12), 2019. — rastreamento ocular de origem.
- Du, H. et al. *Attention-aware Resource Allocation and QoE Analysis for
  Metaverse xURLLC Services*. IEEE JSAC 41(7):2158–2175, 2023. — métrica Meta-Imersão.
- Du, H. et al. *Exploring Attention-Aware Network Resource Allocation for
  Customized Metaverse Services*. IEEE Network 37(6):166–175, 2023. — conjunto UOAL.
- Santos, R. O.; Gomes Jr., B. C.; Santos, C. E. S.; Macedo, C. J. A.;
  Oliveira Jr., A. *QoE-CAC*. SBrT 2025. — métrica QoE-CAC.
- Zhang, W. et al. *K-Net: towards unified image segmentation*. NeurIPS 2021. —
  rótulos de segmentação do UOAL.
- Esteban, C.; Hyland, S. L.; Rätsch, G. *Real-valued (medical) time series
  generation with recurrent conditional GANs*. arXiv:1706.02633, 2017. — TSTR.
- Schuirmann, D. J. *A comparison of the two one-sided tests procedure and the
  power approach for assessing the equivalence of average bioavailability*.
  J. Pharmacokinetics and Biopharmaceutics 15(6), 1987. — TOST.
- Sarwar, B. et al. *Item-based collaborative filtering recommendation
  algorithms*. WWW 2001. — modelo item–item.
- Herlocker, J. L. et al. *An algorithmic framework for performing collaborative
  filtering*. SIGIR 1999. — modelo kNN usuário.
- Koren, Y.; Bell, R.; Volinsky, C. *Matrix factorization techniques for
  recommender systems*. Computer 42(8), 2009. — modelo MF com vieses.
- ITU-T G.1035, *Influencing factors on quality of experience for virtual
  reality services*, 2021.
- ITU-T P.10/G.100, *Vocabulary for performance, quality of service and quality
  of experience*, 2017.
- ITU-T P.800.1, *Mean opinion score (MOS) terminology*, 2006.
