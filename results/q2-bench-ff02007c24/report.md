# Q2: benchmark de extração de argumentos

Gerado por `uv run q2 bench`. Run `q2-bench-ff02007c24`, gold `48e9b5006148`.

Gold: 6 sentenças revisadas por humano.

Split test: 6 sentenças, 36 argumentos no gold. Limiar do merge
calibrado no dev (0 sentenças): **1.01**.

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
| `full_mini` | gpt-4.1-mini | full | 88.0 | 61.1 | 72.1 (60.8–82.2) | 94.2 | 0.0 | 77.3 | 90.9 | 95.5 | 0.0043 | 14203 / 20491 |
| `truncado_mini` | gpt-4.1-mini | truncate | 80.0 | 44.4 | 57.1 (43.1–76.2) | 95.7 | 0.0 | 81.2 | 81.2 | 87.5 | 0.0027 | 8358 / 10105 |
| `chunk_mini` | gpt-4.1-mini | chunk | 72.2 | 72.2 | 72.2 (65.3–80.0) | 99.0 | 0.0 | 76.9 | 92.3 | 96.2 | 0.0062 | 13065 / 17396 |
### Modelos

| config | modelo | modo | P | R | F1 (IC95) | citações ancoradas | brutos sem âncora | tipo | decisão | citação confere | US$/sentença | latência ms p50/p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `chunk_nano` | gpt-4.1-nano | chunk | 78.6 | 30.6 | 44.0 (18.2–54.3) | 92.3 | 0.0 | 54.5 | 72.7 | 54.5 | 0.0011 | 8355 / 9825 |
| `chunk_mini` | gpt-4.1-mini | chunk | 72.2 | 72.2 | 72.2 (65.3–80.0) | 99.0 | 0.0 | 76.9 | 92.3 | 96.2 | 0.0062 | 13065 / 17396 |
| `chunk_5mini` | gpt-5-mini | chunk | 64.4 | 80.6 | 71.6 (63.0–79.3) | 98.0 | 2.0 | 79.3 | 89.7 | 93.1 | 0.0184 | 43469 / 68633 |
| `chunk_41` | gpt-4.1 | chunk | 81.8 | 75.0 | 78.3 (66.7–82.8) | 100.0 | 0.0 | 85.2 | 92.6 | 85.2 | 0.0317 | 10280 / 14130 |
| `full_41` | gpt-4.1 | full | 96.0 | 66.7 | 78.7 (63.1–90.9) | 100.0 | 0.0 | 87.5 | 87.5 | 87.5 | 0.0221 | 12061 / 16757 |
### Prompt e ablações

| config | modelo | modo | P | R | F1 (IC95) | citações ancoradas | brutos sem âncora | tipo | decisão | citação confere | US$/sentença | latência ms p50/p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `chunk_mini` | gpt-4.1-mini | chunk | 72.2 | 72.2 | 72.2 (65.3–80.0) | 99.0 | 0.0 | 76.9 | 92.3 | 96.2 | 0.0062 | 13065 / 17396 |
| `chunk_mini_basico` | gpt-4.1-mini | chunk | 65.8 | 69.4 | 67.6 (57.1–74.2) | 91.6 | 0.0 | 68.0 | 88.0 | 92.0 | 0.0059 | 13758 / 15375 |
| `chunk_mini_sem_filtro` | gpt-4.1-mini | chunk | 72.2 | 72.2 | 72.2 (65.3–80.0) | 99.0 | 0.0 | 76.9 | 92.3 | 96.2 | 0.0062 | 13065 / 17396 |
| `chunk_mini_sem_merge` | gpt-4.1-mini | chunk | 71.1 | 75.0 | 73.0 (64.0–81.1) | 99.0 | 0.0 | 74.1 | 88.9 | 88.9 | 0.0062 | 13065 / 17396 |
### Verificador

| config | modelo | modo | P | R | F1 (IC95) | citações ancoradas | brutos sem âncora | tipo | decisão | citação confere | US$/sentença | latência ms p50/p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `chunk_mini` | gpt-4.1-mini | chunk | 72.2 | 72.2 | 72.2 (65.3–80.0) | 99.0 | 0.0 | 76.9 | 92.3 | 96.2 | 0.0062 | 13065 / 17396 |
| `chunk_mini_verif` | gpt-4.1-mini | chunk | 72.2 | 72.2 | 72.2 (65.3–80.0) | 99.0 | 0.0 | 76.9 | 92.3 | 96.2 | 0.0085 | 19515 / 28510 |
| `chunk_mini_verif_drop` | gpt-4.1-mini | chunk | 72.2 | 72.2 | 72.2 (65.3–80.0) | 99.0 | 0.0 | 76.9 | 92.3 | 96.2 | 0.0085 | 19515 / 28510 |
### Versão do prompt (v1 × v2)

| config | modelo | modo | P | R | F1 (IC95) | citações ancoradas | brutos sem âncora | tipo | decisão | citação confere | US$/sentença | latência ms p50/p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `chunk_mini` | gpt-4.1-mini | chunk | 72.2 | 72.2 | 72.2 (65.3–80.0) | 99.0 | 0.0 | 76.9 | 92.3 | 96.2 | 0.0062 | 13065 / 17396 |
| `chunk_mini_v2` | gpt-4.1-mini | chunk | 73.7 | 77.8 | 75.7 (66.6–82.1) | 98.0 | 0.0 | 67.9 | 89.3 | 89.3 | 0.0064 | 12446 / 21560 |
| `full_41` | gpt-4.1 | full | 96.0 | 66.7 | 78.7 (63.1–90.9) | 100.0 | 0.0 | 87.5 | 87.5 | 87.5 | 0.0221 | 12061 / 16757 |
| `full_41_v2` | gpt-4.1 | full | 90.0 | 75.0 | 81.8 (66.7–88.4) | 100.0 | 0.0 | 88.9 | 85.2 | 85.2 | 0.0239 | 14572 / 20369 |

## Por faixa de tamanho (recall / precisão %)

Sentenças no test por faixa: curto 1, medio 3, longo 1, muito_longo 1. Faixas com poucas
sentenças indicam tendência, não prova.

| config | curto | medio | longo | muito_longo |
|---|---|---|---|---|
| `full_mini` | 100.0 / 100.0 | 58.3 / 77.8 | 63.6 / 87.5 | 50.0 / 100.0 |
| `truncado_mini` | 100.0 / 100.0 | 58.3 / 70.0 | 27.3 / 100.0 | 30.0 / 75.0 |
| `chunk_mini` | 100.0 / 100.0 | 58.3 / 77.8 | 81.8 / 64.3 | 70.0 / 70.0 |
| `chunk_41` | 66.7 / 100.0 | 58.3 / 87.5 | 100.0 / 73.3 | 70.0 / 87.5 |

## Confiança calibrada? (precisão % por nível)

Confiança = média do score de ancoragem × fator do verificador (sem verificador, teto 0,8:
nada sai "alta").

| config | alta | média | baixa |
|---|---|---|---|
| `chunk_mini` | — | 72.2 (n=36) | — |
| `chunk_mini_verif` | 72.2 (n=36) | — | — |

## Calibração do merge (dev, `chunk_mini`)

| limiar cosseno | P | R | F1 |
|---|---|---|---|
| 0.8 | 0.0 | 0.0 | 0.0 |
| 0.85 | 0.0 | 0.0 | 0.0 |
| 0.88 | 0.0 | 0.0 | 0.0 |
| 0.92 | 0.0 | 0.0 | 0.0 |
| 1.01 | 0.0 | 0.0 | 0.0 |

## Outros números

| config | extraídos | fundidos no merge | descartados pelo verificador | chunks falhos / com retry | extraídos em sentença sem argumento | chunks/sentença | tokens entrada/sentença | concordância pareadores % | F1 com pareador embeddings |
|---|---|---|---|---|---|---|---|---|---|
| `full_mini` | 25 | 0 | 0 | 0 / 0 | 0 | 1.0 | 6176 | 91.3 | 72.1 |
| `truncado_mini` | 20 | 0 | 0 | 0 / 0 | 0 | 1.0 | 3493 | 57.9 | 50.0 |
| `chunk_mini` | 36 | 2 | 0 | 0 / 0 | 0 | 2.2 | 8409 | 79.3 | 72.2 |
| `chunk_mini_basico` | 38 | 1 | 0 | 0 / 0 | 0 | 2.2 | 7453 | 81.5 | 64.9 |
| `chunk_mini_v2` | 38 | 2 | 0 | 0 / 0 | 0 | 2.2 | 8680 | 71.9 | 73.0 |
| `full_41_v2` | 30 | 0 | 0 | 0 / 0 | 0 | 1.0 | 6301 | 92.9 | 81.8 |
| `chunk_mini_sem_filtro` | 36 | 2 | 0 | 0 / 0 | 0 | 2.2 | 8409 | 79.3 | 72.2 |
| `chunk_mini_sem_merge` | 38 | 0 | 0 | 0 / 0 | 0 | 2.2 | 8409 | 71.0 | 70.3 |
| `chunk_mini_verif` | 36 | 2 | 0 | 0 / 0 | 0 | 2.2 | 12977 | 79.3 | 72.2 |
| `chunk_mini_verif_drop` | 36 | 2 | 0 | 0 / 0 | 0 | 2.2 | 12977 | 79.3 | 72.2 |
| `chunk_nano` | 14 | 0 | 0 | 0 / 0 | 0 | 2.2 | 8409 | 63.6 | 28.0 |
| `chunk_5mini` | 45 | 3 | 0 | 0 / 0 | 0 | 2.2 | 8404 | 78.1 | 69.1 |
| `chunk_41` | 33 | 2 | 0 | 0 / 0 | 0 | 2.2 | 8409 | 89.3 | 75.4 |
| `full_41` | 25 | 1 | 0 | 0 / 0 | 0 | 1.0 | 6176 | 80.8 | 75.4 |

## Configurações

- `full_mini`: run `q2-extract-dbc71eacf3`, prompt `q2-extract-guia-v1` (sha `45eabfd25126781a`), schema `q2-extracted-v2`, params `{"mode": "full", "model": "gpt-4.1-mini", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `truncado_mini`: run `q2-extract-3651d4f67e`, prompt `q2-extract-guia-v1` (sha `45eabfd25126781a`), schema `q2-extracted-v2`, params `{"mode": "truncate", "model": "gpt-4.1-mini", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `chunk_mini`: run `q2-extract-1fc1179879`, prompt `q2-extract-guia-v1+chunk` (sha `8f5f6e5815dbedb8`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-4.1-mini", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `chunk_mini_basico`: run `q2-extract-9e206b9088`, prompt `q2-extract-basico-v1+chunk` (sha `9b8fb73777d36262`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-4.1-mini", "prompt": "basico", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `chunk_mini_v2`: run `q2-extract-1adea39d90`, prompt `q2-extract-guia-v2+chunk` (sha `974faab14f6cbcdf`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-4.1-mini", "prompt": "guia2", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `full_41_v2`: run `q2-extract-927f16f03d`, prompt `q2-extract-guia-v2` (sha `6fc6e3e4600de1b5`), schema `q2-extracted-v2`, params `{"mode": "full", "model": "gpt-4.1", "prompt": "guia2", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `chunk_mini_sem_filtro`: run `q2-extract-d81ce1689d`, prompt `q2-extract-guia-v1+chunk` (sha `8f5f6e5815dbedb8`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-4.1-mini", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": false, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `chunk_mini_sem_merge`: run `q2-extract-2d52b637d9`, prompt `q2-extract-guia-v1+chunk` (sha `8f5f6e5815dbedb8`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-4.1-mini", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": false, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `chunk_mini_verif`: run `q2-extract-932da52732`, prompt `q2-extract-guia-v1+chunk` (sha `fd454892f3218804`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-4.1-mini", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": "gpt-4.1-mini", "verifier_drop": false}`
- `chunk_mini_verif_drop`: run `q2-extract-d0899be0f6`, prompt `q2-extract-guia-v1+chunk` (sha `fd454892f3218804`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-4.1-mini", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": "gpt-4.1-mini", "verifier_drop": true}`
- `chunk_nano`: run `q2-extract-89eacf7475`, prompt `q2-extract-guia-v1+chunk` (sha `8f5f6e5815dbedb8`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-4.1-nano", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `chunk_5mini`: run `q2-extract-42db9b406d`, prompt `q2-extract-guia-v1+chunk` (sha `8f5f6e5815dbedb8`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-5-mini", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `chunk_41`: run `q2-extract-9ddf560e55`, prompt `q2-extract-guia-v1+chunk` (sha `8f5f6e5815dbedb8`), schema `q2-extracted-v2`, params `{"mode": "chunk", "model": "gpt-4.1", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
- `full_41`: run `q2-extract-e74e314fd5`, prompt `q2-extract-guia-v1` (sha `45eabfd25126781a`), schema `q2-extracted-v2`, params `{"mode": "full", "model": "gpt-4.1", "prompt": "guia", "budget": 3000, "overlap": 300, "head": 800, "tail": 600, "ground_filter": true, "merge": true, "merge_threshold": 1.01, "verifier": null, "verifier_drop": false}`
