# Q2: benchmark de extração de argumentos

Gerado por `uv run q2 bench`. Run `q2-bench-3e5fb45870`, gold `b139d00606a5`.

> **PROVISÓRIO.** O gold ainda contém 30 rascunho(s) LLM (`gpt-5`) não revisados por humano (0 revisados). Números medem concordância com o rascunho; modelos da mesma família do rascunho tendem a ser favorecidos. Reexecutar após a revisão.

Split test: 20 sentenças, 163 argumentos no gold. Limiar do merge
calibrado no dev (10 sentenças): **1.01**.

Pareamento extraído × gold: juiz `gpt-4.1` (`q2-match-v2`) decide se tratam da
mesma questão; one-to-one forçado em código. P = extraídos pareados / extraídos; R = gold
pareados / gold. Campos (tipo, decisão) e "citação confere" (alguma citação extraída sobrepõe
uma citação do gold no texto) são medidos sobre os pares. "Brutos sem âncora" = argumentos
devolvidos pelo LLM sem nenhuma citação encontrada no texto (proxy de invenção), antes do
filtro. Latência = chamadas originais, chunks em paralelo. IC95 = bootstrap sobre sentenças.

## Resultados

### Contexto longo

| config | modelo | modo | P | R | F1 (IC95) | citações ancoradas | brutos sem âncora | tipo | decisão | citação confere | US$/sentença | latência ms p50/p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mini` | gpt-4.1-mini | full | 94.0 | 57.7 | 71.5 (66.4–77.7) | 92.1 | 0.0 | 86.2 | 87.2 | 90.4 | 0.0048 | 15443 / 21475 |
| `truncado_mini` | gpt-4.1-mini | truncate | 87.8 | 39.9 | 54.9 (47.7–63.6) | 94.8 | 1.3 | 78.5 | 78.5 | 83.1 | 0.0028 | 9640 / 12952 |
| `chunk_mini` | gpt-4.1-mini | chunk | 78.2 | 70.6 | 74.2 (69.9–78.7) | 96.1 | 0.0 | 84.3 | 80.9 | 93.0 | 0.0074 | 15935 / 23713 |
### Modelos

| config | modelo | modo | P | R | F1 (IC95) | citações ancoradas | brutos sem âncora | tipo | decisão | citação confere | US$/sentença | latência ms p50/p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `chunk_nano` | gpt-4.1-nano | chunk | 84.1 | 32.5 | 46.9 (37.0–55.3) | 93.2 | 1.6 | 69.8 | 75.5 | 52.8 | 0.0012 | 9176 / 11285 |
| `chunk_mini` | gpt-4.1-mini | chunk | 78.2 | 70.6 | 74.2 (69.9–78.7) | 96.1 | 0.0 | 84.3 | 80.9 | 93.0 | 0.0074 | 15935 / 23713 |
| `chunk_5mini` | gpt-5-mini | chunk | 71.1 | 81.6 | 76.0 (71.7–81.3) | 98.4 | 2.4 | 85.0 | 82.7 | 93.2 | 0.0221 | 53938 / 72565 |
| `chunk_41` | gpt-4.1 | chunk | 80.1 | 76.7 | 78.4 (73.3–82.4) | 98.6 | 2.3 | 95.2 | 85.6 | 91.2 | 0.0384 | 11782 / 16708 |
| `full_41` | gpt-4.1 | full | 99.1 | 70.6 | 82.4 (77.1–87.1) | 99.6 | 0.8 | 93.0 | 83.5 | 95.7 | 0.0268 | 13394 / 17808 |
### Prompt e ablações

| config | modelo | modo | P | R | F1 (IC95) | citações ancoradas | brutos sem âncora | tipo | decisão | citação confere | US$/sentença | latência ms p50/p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `chunk_mini` | gpt-4.1-mini | chunk | 78.2 | 70.6 | 74.2 (69.9–78.7) | 96.1 | 0.0 | 84.3 | 80.9 | 93.0 | 0.0074 | 15935 / 23713 |
| `chunk_mini_basico` | gpt-4.1-mini | chunk | 70.3 | 71.2 | 70.7 (66.5–75.5) | 90.6 | 3.4 | 75.9 | 63.8 | 92.2 | 0.0072 | 14712 / 19989 |
| `chunk_mini_sem_filtro` | gpt-4.1-mini | chunk | 78.2 | 70.6 | 74.2 (69.9–78.7) | 96.1 | 0.0 | 84.3 | 80.9 | 93.0 | 0.0074 | 15935 / 23713 |
| `chunk_mini_sem_merge` | gpt-4.1-mini | chunk | 75.0 | 71.8 | 73.4 (68.9–78.4) | 96.1 | 0.0 | 82.9 | 79.5 | 91.5 | 0.0074 | 15935 / 23713 |
### Verificador

| config | modelo | modo | P | R | F1 (IC95) | citações ancoradas | brutos sem âncora | tipo | decisão | citação confere | US$/sentença | latência ms p50/p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `chunk_mini` | gpt-4.1-mini | chunk | 78.2 | 70.6 | 74.2 (69.9–78.7) | 96.1 | 0.0 | 84.3 | 80.9 | 93.0 | 0.0074 | 15935 / 23713 |
| `chunk_mini_verif` | gpt-4.1-mini | chunk | 78.2 | 70.6 | 74.2 (69.9–78.7) | 96.1 | 0.0 | 84.3 | 80.9 | 93.0 | 0.0101 | 22607 / 34308 |
| `chunk_mini_verif_drop` | gpt-4.1-mini | chunk | 78.1 | 69.9 | 73.8 (69.2–78.5) | 96.1 | 0.0 | 84.2 | 80.7 | 93.9 | 0.0101 | 22607 / 34308 |
### Versão do prompt (v1 × v2)

| config | modelo | modo | P | R | F1 (IC95) | citações ancoradas | brutos sem âncora | tipo | decisão | citação confere | US$/sentença | latência ms p50/p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `chunk_mini` | gpt-4.1-mini | chunk | 78.2 | 70.6 | 74.2 (69.9–78.7) | 96.1 | 0.0 | 84.3 | 80.9 | 93.0 | 0.0074 | 15935 / 23713 |
| `chunk_mini_v2` | gpt-4.1-mini | chunk | 77.2 | 74.8 | 76.0 (71.6–80.6) | 97.1 | 0.6 | 82.0 | 77.9 | 91.8 | 0.0074 | 16243 / 23003 |
| `full_41` | gpt-4.1 | full | 99.1 | 70.6 | 82.4 (77.1–87.1) | 99.6 | 0.8 | 93.0 | 83.5 | 95.7 | 0.0268 | 13394 / 17808 |
| `full_41_v2` | gpt-4.1 | full | 93.1 | 74.2 | 82.6 (76.5–87.7) | 99.2 | 0.0 | 96.7 | 86.8 | 92.6 | 0.0270 | 15561 / 22790 |

## Por faixa de tamanho (recall / precisão %)

Sentenças no test por faixa: curto 2, medio 9, longo 5, muito_longo 4. Faixas com poucas
sentenças indicam tendência, não prova.

| config | curto | medio | longo | muito_longo |
|---|---|---|---|---|
| `full_mini` | 100.0 / 100.0 | 66.1 / 94.9 | 56.1 / 91.4 | 46.8 / 95.7 |
| `truncado_mini` | 100.0 / 100.0 | 58.9 / 84.6 | 29.8 / 94.4 | 25.5 / 85.7 |
| `chunk_mini` | 100.0 / 100.0 | 69.6 / 88.6 | 71.9 / 75.9 | 68.1 / 69.6 |
| `chunk_41` | 66.7 / 100.0 | 69.6 / 90.7 | 80.7 / 80.7 | 80.9 / 70.4 |

## Confiança calibrada? (precisão % por nível)

Confiança = média do score de ancoragem × fator do verificador (sem verificador, teto 0,8:
nada sai "alta").

| config | alta | média | baixa |
|---|---|---|---|
| `chunk_mini` | — | 78.2 (n=147) | — |
| `chunk_mini_verif` | 78.1 (n=146) | — | 100.0 (n=1) |

## Calibração do merge (dev, `chunk_mini`)

| limiar cosseno | P | R | F1 |
|---|---|---|---|
| 0.8 | 90.2 | 65.7 | 76.0 |
| 0.85 | 90.4 | 67.1 | 77.0 |
| 0.88 | 90.4 | 67.1 | 77.0 |
| 0.92 | 90.4 | 67.1 | 77.0 |
| 1.01 | 90.4 | 67.1 | 77.0 |

## Outros números

| config | extraídos | fundidos no merge | descartados pelo verificador | chunks falhos / com retry | extraídos em sentença sem argumento | chunks/sentença | tokens entrada/sentença | concordância pareadores % | F1 com pareador embeddings |
|---|---|---|---|---|---|---|---|---|---|
| `full_mini` | 100 | 0 | 0 | 0 / 0 | 0 | 1.0 | 6891 | 83.0 | 67.7 |
| `truncado_mini` | 74 | 1 | 0 | 0 / 0 | 0 | 1.0 | 3532 | 71.8 | 48.1 |
| `chunk_mini` | 147 | 9 | 0 | 0 / 0 | 0 | 2.3 | 9195 | 82.3 | 71.6 |
| `chunk_mini_basico` | 165 | 8 | 0 | 0 / 0 | 0 | 2.3 | 8181 | 77.3 | 67.7 |
| `chunk_mini_v2` | 158 | 11 | 0 | 1 / 0 | 0 | 2.3 | 9247 | 77.8 | 73.5 |
| `full_41_v2` | 130 | 2 | 0 | 0 / 0 | 0 | 1.0 | 7016 | 89.7 | 80.5 |
| `chunk_mini_sem_filtro` | 147 | 9 | 0 | 0 / 0 | 0 | 2.3 | 9195 | 82.3 | 71.6 |
| `chunk_mini_sem_merge` | 156 | 0 | 0 | 0 / 0 | 0 | 2.3 | 9195 | 76.2 | 70.2 |
| `chunk_mini_verif` | 147 | 9 | 0 | 0 / 0 | 0 | 2.3 | 14431 | 82.3 | 71.6 |
| `chunk_mini_verif_drop` | 146 | 9 | 1 | 0 / 0 | 0 | 2.3 | 14431 | 82.9 | 71.8 |
| `chunk_nano` | 63 | 0 | 0 | 0 / 0 | 0 | 2.3 | 9195 | 53.4 | 31.9 |
| `chunk_5mini` | 187 | 17 | 0 | 0 / 0 | 0 | 2.3 | 9191 | 78.9 | 74.3 |
| `chunk_41` | 156 | 11 | 0 | 0 / 0 | 0 | 2.3 | 9195 | 86.5 | 77.1 |
| `full_41` | 116 | 3 | 0 | 0 / 0 | 0 | 1.0 | 6891 | 89.2 | 80.3 |

## Configurações

- `full_mini`: run `q2-extract-99ddecfabe`, prompt `q2-extract-guia-v1` (sha `45eabfd25126781a`), schema `q2-extracted-v2`, params `{"mode": "full", "model": "gpt-4.1-mini", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `truncado_mini`: run `q2-extract-5641ecf694`, prompt `q2-extract-guia-v1` (sha `45eabfd25126781a`), schema `q2-extracted-v2`, params `{"mode": "truncate", "model": "gpt-4.1-mini", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `chunk_mini`: run `q2-extract-eda2728692`, prompt `q2-extract-guia-v1+chunk` (sha `8f5f6e5815dbedb8`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-4.1-mini", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `chunk_mini_basico`: run `q2-extract-26c1267bd7`, prompt `q2-extract-basico-v1+chunk` (sha `9b8fb73777d36262`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-4.1-mini", "prompt": "basico", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `chunk_mini_v2`: run `q2-extract-bad888d8b5`, prompt `q2-extract-guia-v2+chunk` (sha `974faab14f6cbcdf`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-4.1-mini", "prompt": "guia2", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `full_41_v2`: run `q2-extract-760976f574`, prompt `q2-extract-guia-v2` (sha `6fc6e3e4600de1b5`), schema `q2-extracted-v2`, params `{"mode": "full", "model": "gpt-4.1", "prompt": "guia2", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `chunk_mini_sem_filtro`: run `q2-extract-ecf1ff2997`, prompt `q2-extract-guia-v1+chunk` (sha `8f5f6e5815dbedb8`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-4.1-mini", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": false, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `chunk_mini_sem_merge`: run `q2-extract-ce0be780df`, prompt `q2-extract-guia-v1+chunk` (sha `8f5f6e5815dbedb8`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-4.1-mini", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": false, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `chunk_mini_verif`: run `q2-extract-671339d5ea`, prompt `q2-extract-guia-v1+chunk` (sha `fd454892f3218804`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-4.1-mini", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": "gpt-4.1-mini", "verifier_drop": false}`
- `chunk_mini_verif_drop`: run `q2-extract-719d140a58`, prompt `q2-extract-guia-v1+chunk` (sha `fd454892f3218804`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-4.1-mini", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": "gpt-4.1-mini", "verifier_drop": true}`
- `chunk_nano`: run `q2-extract-b562d60ec3`, prompt `q2-extract-guia-v1+chunk` (sha `8f5f6e5815dbedb8`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-4.1-nano", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `chunk_5mini`: run `q2-extract-7b6641e690`, prompt `q2-extract-guia-v1+chunk` (sha `8f5f6e5815dbedb8`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-5-mini", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `chunk_41`: run `q2-extract-b79dd7e679`, prompt `q2-extract-guia-v1+chunk` (sha `8f5f6e5815dbedb8`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-4.1", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `full_41`: run `q2-extract-c9823d5f00`, prompt `q2-extract-guia-v1` (sha `45eabfd25126781a`), schema `q2-extracted-v2`, params `{"mode": "full", "model": "gpt-4.1", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
