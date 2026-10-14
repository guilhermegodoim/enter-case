# Dataset sintético Q1: estatísticas

Gerado por `uv run q1 build`. Não editar à mão.

| campo | valor |
|---|---|
| versão | `q1-synth-v1` |
| seed | 13 |
| fingerprint (docs + pares) | `d4c0f411b0ec` |
| fingerprint dos templates | `225836f76519` |
| casos | 60 (30 dev, 30 test) |
| documentos | 373 |
| documentos por caso | min 2, mediana 6, max 10 |
| caracteres por documento | mediana 11521, max 23624 |

Pares = cada documento comparado com todos os anteriores do mesmo caso, na ordem de chegada.

## Pares por rótulo

| rótulo | dev | test | total | % |
|---|---|---|---|---|
| copia | 92 | 92 | 184 | 17.3 |
| versao | 180 | 142 | 322 | 30.2 |
| diferente | 256 | 303 | 559 | 52.5 |
| **total** | 528 | 537 | 1065 | 100 |

## Pares por relação

Distância de blocos = 1 − Jaccard ponderado por palavras entre os blocos de origem dos dois
documentos (0 = mesmo conteúdo). Ruído de superfície e troca de entidades não alteram blocos.

| rótulo | relação | pares | distância de blocos média |
|---|---|---|---|
| copia | ruido_scan_media | 51 | 0.000 |
| copia | ruido_scan_forte | 42 | 0.000 |
| copia | troca_entidades | 28 | 0.000 |
| copia | ruido_pdf | 26 | 0.000 |
| copia | ruido_scan_leve | 26 | 0.000 |
| copia | identica | 11 | 0.000 |
| versao | versao_forte | 172 | 0.395 |
| versao | versao_media | 98 | 0.198 |
| versao | versao_leve | 52 | 0.080 |
| diferente | tipo_diferente | 278 | 1.000 |
| diferente | tese_vizinha_mesmo_escritorio | 129 | 0.783 |
| diferente | mesma_tese_outro_escritorio | 85 | 0.982 |
| diferente | outra_tese_mesmo_escritorio | 49 | 0.777 |
| diferente | outra_tese_outro_escritorio | 18 | 1.000 |

## Tamanho dos documentos (palavras)

| tipo | escritório | docs | palavras min | mediana | p90 | max |
|---|---|---|---|---|---|---|
| contestacao | ré | 53 | 904 | 1009 | 1062 | 1071 |
| peticao_inicial | firm_a | 142 | 2582 | 3218 | 3464 | 3547 |
| peticao_inicial | firm_b | 178 | 1438 | 1668 | 1790 | 1925 |

## Documentos por operação

| operação | docs |
|---|---|
| original | 60 |
| contestacao | 44 |
| neg:tese_vizinha_mesmo_escritorio | 30 |
| versao:leve | 30 |
| versao:media | 29 |
| versao:forte | 25 |
| neg:mesma_tese_outro_escritorio | 21 |
| copia:scan_forte | 20 |
| copia:pdf | 19 |
| copia:scan_media | 19 |
| copia:scan_leve | 17 |
| copia:de_versao | 16 |
| copia:troca_entidades | 13 |
| copia:identica | 11 |
| neg:outra_tese_mesmo_escritorio | 10 |
| copia:contestacao | 9 |

## Documentos por superfície

| superfície | docs |
|---|---|
| digital | 124 |
| pdf | 150 |
| scan_leve | 42 |
| scan_media | 37 |
| scan_forte | 20 |
