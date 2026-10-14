# Q1: benchmark de similaridade (cópia / versão / diferente)

Gerado por `uv run q1 bench`. Dataset `q1-synth-v1` (fingerprint
`d4c0f411b0ec`). Calibração no split dev, números no split test
(186 documentos, 537 pares).

Cada documento de teste é ingerido na ordem de chegada do caso e comparado com os anteriores.
Métricas por par e por documento (ação tomada: cópia, versão ou novo). Latência por documento =
CPU local + latência original das chamadas de API. IC95 = bootstrap sobre casos.

## Métodos

| método | macro-F1 % (IC95) | P cópia | R cópia | F1 versão | F1 dif. | acurácia ação (doc) | precisão ação cópia | revisão humana % | latência ms p50/p95 | US$/doc |
|---|---|---|---|---|---|---|---|---|---|---|
| `jaccard_bruto` | 69.9 (65.6–73.3) | 94.4 | 18.5 | 78.8 | 100.0 | 75.3 | 94.4 | 0.0 | 6 / 12 | 0.00000 |
| `minhash` | 73.1 (67.7–78.2) | 85.7 | 26.1 | 79.3 | 100.0 | 76.3 | 87.5 | 0.0 | 9 / 18 | 0.00000 |
| `embeddings` | 65.5 (60.2–69.9) | 100.0 | 17.4 | 72.3 | 94.6 | 73.7 | 100.0 | 0.0 | 50 / 852 | 0.00008 |
| `alinhamento` | 98.5 (95.6–100.0) | 98.9 | 95.7 | 98.3 | 100.0 | 98.9 | 98.4 | 0.0 | 13 / 30 | 0.00000 |
| `logreg` | 98.8 (97.1–100.0) | 95.8 | 100.0 | 98.6 | 100.0 | 98.9 | 96.9 | 2.7 | 55 / 858 | 0.00008 |
| `cascata` | 98.5 (95.6–100.0) | 98.9 | 95.7 | 98.3 | 100.0 | 98.9 | 98.4 | 0.0 | 11 / 25 | 0.00000 |
| `cascata_llm` | 99.4 (97.9–100.0) | 100.0 | 97.8 | 99.3 | 100.0 | 100.0 | 100.0 | 5.4 | 11 / 6596 | 0.00010 |

## Ablações

| método | macro-F1 % (IC95) | P cópia | R cópia | F1 versão | F1 dif. | acurácia ação (doc) | precisão ação cópia | revisão humana % | latência ms p50/p95 | US$/doc |
|---|---|---|---|---|---|---|---|---|---|---|
| `sem_mascara` | 97.0 (93.5–99.4) | 98.8 | 90.2 | 96.6 | 100.0 | 97.3 | 98.3 | 0.0 | 10 / 25 | 0.00000 |
| `sem_layout` | 97.6 (95.6–99.4) | 93.8 | 97.8 | 97.1 | 100.0 | 98.4 | 95.4 | 0.0 | 14 / 35 | 0.00000 |
| `minhash_boilerplate` | 51.0 (44.0–57.6) | 100.0 | 17.4 | 54.7 | 68.7 | 54.8 | 100.0 | 0.0 | 10 / 20 | 0.00000 |
| `alinhamento_sem_hash` | 98.5 (95.6–100.0) | 98.9 | 95.7 | 98.3 | 100.0 | 98.9 | 98.4 | 0.0 | 14 / 32 | 0.00000 |

## Acerto por relação (% dos pares)

Linhas `diferente/*` são os negativos difíceis: vocabulário, partes ou boilerplate em comum.

| relação | `jaccard_bruto` | `minhash` | `embeddings` | `alinhamento` | `logreg` | `cascata` | `cascata_llm` |
|---|---|---|---|---|---|---|---|
| copia/identica | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| copia/ruido_pdf | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| copia/ruido_scan_forte | 0.0 | 0.0 | 0.0 | 91.3 | 100.0 | 91.3 | 91.3 |
| copia/ruido_scan_leve | 7.1 | 57.1 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| copia/ruido_scan_media | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| copia/troca_entidades | 0.0 | 0.0 | 0.0 | 77.8 | 100.0 | 77.8 | 100.0 |
| diferente/mesma_tese_outro_escritorio | 100.0 | 100.0 | 93.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| diferente/outra_tese_mesmo_escritorio | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| diferente/outra_tese_outro_escritorio | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| diferente/tese_vizinha_mesmo_escritorio | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| diferente/tipo_diferente | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| versao/versao_forte | 100.0 | 100.0 | 80.6 | 100.0 | 100.0 | 100.0 | 100.0 |
| versao/versao_leve | 93.8 | 75.0 | 93.8 | 93.8 | 75.0 | 93.8 | 100.0 |
| versao/versao_media | 100.0 | 100.0 | 85.9 | 100.0 | 100.0 | 100.0 | 100.0 |

## Matriz de confusão: `cascata_llm` (linhas = verdade, colunas = predição)

|  | copia | versao | diferente |
|---|---|---|---|
| **copia** | 90 | 2 | 0 |
| **versao** | 0 | 142 | 0 |
| **diferente** | 0 | 0 | 303 |

Estágios que decidiram os pares: `{"prefiltro": 229, "regra": 272, "hash": 16, "llm": 20}`.
Pares na zona cinzenta enviados ao LLM: 20.
Juiz: `gpt-4.1-mini`, prompt `q1-judge-v1`,
20 chamadas, 26103 tokens de entrada no total.

## Escala: MinHash-LSH sobre o corpus inteiro

Fora do escopo da Q1 (comparação é dentro do caso), mas mostra que a mesma assinatura serve
para deduplicação entre casos: limiar 0.5, 373 documentos.

| métrica | valor |
|---|---|
| recall de pares da mesma família | 92.1% |
| candidatos por documento | 21.5 (5.8% do corpus) |
| consulta por documento | 0.02 ms |

## Configuração

- `jaccard_bruto`: Jaccard de shingles sobre texto só normalizado. Parâmetros: `{"tipo": "regra", "sim": "jaccard", "t_diff": 0.317474, "change": "-jaccard", "t_copy": -0.925716, "fast_path": false, "gray_factor": 0.0, "prefiltro_minhash": 0.0, "llm": false}`
- `minhash`: MinHash (128 perm.) sobre texto limpo e mascarado. Parâmetros: `{"tipo": "regra", "sim": "minhash", "t_diff": 0.3125, "change": "-minhash", "t_copy": -0.957031, "fast_path": true, "gray_factor": 0.0, "prefiltro_minhash": 0.0, "llm": false}`
- `embeddings`: Cosseno médio de embeddings por trecho (text-embedding-3-small). Parâmetros: `{"tipo": "regra", "sim": "emb_cos", "t_diff": 0.977434, "change": "-emb_cos", "t_copy": -0.999601, "fast_path": false, "gray_factor": 0.0, "prefiltro_minhash": 0.0, "llm": false}`
- `alinhamento`: Hash + alinhamento de tokens com trechos classificados (regra). Parâmetros: `{"tipo": "regra", "sim": "align_cov", "t_diff": 0.632786, "change": "content_ratio", "t_copy": 0.011338, "fast_path": true, "gray_factor": 0.0, "prefiltro_minhash": 0.0, "llm": false}`
- `logreg`: Regressão logística sobre todas as features léxicas + embeddings. Parâmetros: `{"tipo": "logreg", "features": ["exact", "skeleton", "jaccard", "contain_b", "contain_a", "minhash", "align_sim", "align_cov", "content_ratio", "data_ratio", "noise_ratio", "moved_ratio", "n_content_regions", "max_content_region", "len_ratio", "emb_cos", "emb_chunk_cov"], "min_conf": 0.8, "prefiltro_minhash": 0.0, "llm": false}`
- `cascata`: Hash → pré-filtro MinHash → alinhamento (regra). Parâmetros: `{"tipo": "regra", "sim": "align_cov", "t_diff": 0.632786, "change": "content_ratio", "t_copy": 0.011338, "fast_path": true, "gray_factor": 0.0, "prefiltro_minhash": 0.1758, "llm": false}`
- `cascata_llm`: Cascata + juiz LLM na zona cinzenta cópia × versão. Parâmetros: `{"tipo": "regra", "sim": "align_cov", "t_diff": 0.632786, "change": "content_ratio", "t_copy": 0.011338, "fast_path": true, "gray_factor": 2.0, "prefiltro_minhash": 0.1758, "llm": true}`
- `sem_mascara`: Alinhamento sem máscara de entidades. Parâmetros: `{"tipo": "regra", "sim": "align_cov", "t_diff": 0.627098, "change": "content_ratio", "t_copy": 0.011599, "fast_path": true, "gray_factor": 0.0, "prefiltro_minhash": 0.0, "llm": false}`
- `sem_layout`: Alinhamento sem limpeza de cabeçalho/paginação. Parâmetros: `{"tipo": "regra", "sim": "align_cov", "t_diff": 0.628097, "change": "content_ratio", "t_copy": 0.01576, "fast_path": true, "gray_factor": 0.0, "prefiltro_minhash": 0.0, "llm": false}`
- `minhash_boilerplate`: MinHash com shingles de boilerplate removidos. Parâmetros: `{"tipo": "regra", "sim": "minhash_f", "t_diff": 0.042969, "change": "-minhash_f", "t_copy": -1.0, "fast_path": true, "gray_factor": 0.0, "prefiltro_minhash": 0.0, "llm": false}`
- `alinhamento_sem_hash`: Alinhamento sem atalho de hash. Parâmetros: `{"tipo": "regra", "sim": "align_cov", "t_diff": 0.632786, "change": "content_ratio", "t_copy": 0.011338, "fast_path": false, "gray_factor": 0.0, "prefiltro_minhash": 0.0, "llm": false}`
