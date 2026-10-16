# Q2: comparação entre execuções

Gerado por `uv run q2 compare q2-bench-4d5403af77 q2-bench-3e5fb45870`.

Gold e juiz iguais em todas as linhas: números diretamente comparáveis.
> **PROVISÓRIO**: gold ainda contém rascunhos LLM não revisados.

O `run_id` da extração é o hash de modelo, versão e hash do prompt, schema, parâmetros e código:
mudar o schema ou o código gera run nova. Mesma config (modelo, modo, versão do prompt)
reexecutada em benchmarks diferentes, com o cache LLM, deve dar o mesmo F1:
`chunk_mini` (2 runs, schemas q2-extracted-v1, q2-extracted-v2): F1 74.2 / 74.2; `full_41` (2 runs, schemas q2-extracted-v1, q2-extracted-v2): F1 82.4 / 82.4.

| bench | config | run extração | modelo | modo | prompt | sha prompt | schema | gold | P | R | F1 (IC95) | US$/sentença | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `q2-bench-4d5403af77` | `chunk_mini` | `q2-extract-12db7f9f31` | gpt-4.1-mini | chunk | `q2-extract-guia-v1+chunk` | `—` | `q2-extracted-v1` | `b139d00606a5` | 78.2 | 70.6 | 74.2 (69.9–78.7) | 0.0074 | 15935 |
| `q2-bench-4d5403af77` | `full_41` | `q2-extract-a7716daf2f` | gpt-4.1 | full | `q2-extract-guia-v1` | `—` | `q2-extracted-v1` | `b139d00606a5` | 99.1 | 70.6 | 82.4 (77.1–87.1) | 0.0268 | 13394 |
| `q2-bench-3e5fb45870` | `chunk_mini` | `q2-extract-eda2728692` | gpt-4.1-mini | chunk | `q2-extract-guia-v1+chunk` | `8f5f6e5815dbedb8` | `q2-extracted-v2` | `b139d00606a5` | 78.2 | 70.6 | 74.2 (69.9–78.7) | 0.0074 | 15935 |
| `q2-bench-3e5fb45870` | `chunk_mini_v2` | `q2-extract-bad888d8b5` | gpt-4.1-mini | chunk | `q2-extract-guia-v2+chunk` | `974faab14f6cbcdf` | `q2-extracted-v2` | `b139d00606a5` | 77.2 | 74.8 | 76.0 (71.6–80.6) | 0.0074 | 16243 |
| `q2-bench-3e5fb45870` | `full_41_v2` | `q2-extract-760976f574` | gpt-4.1 | full | `q2-extract-guia-v2` | `6fc6e3e4600de1b5` | `q2-extracted-v2` | `b139d00606a5` | 93.1 | 74.2 | 82.6 (76.5–87.7) | 0.0270 | 15561 |
| `q2-bench-3e5fb45870` | `full_41` | `q2-extract-c9823d5f00` | gpt-4.1 | full | `q2-extract-guia-v1` | `45eabfd25126781a` | `q2-extracted-v2` | `b139d00606a5` | 99.1 | 70.6 | 82.4 (77.1–87.1) | 0.0268 | 13394 |
