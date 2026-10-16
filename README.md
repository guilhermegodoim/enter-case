# Case técnico Enter

Respostas às duas perguntas do case, cada uma sustentada por um mini projeto e um benchmark:

- **Q1**: similaridade entre documentos do mesmo caso (cópia / versão modificada / documento diferente).
- **Q2**: extração estruturada de argumentos de sentenças, com revisão humana e versionamento.

Dados usados: apenas documentos sintéticos (Q1) e decisões judiciais públicas (Q2). Nenhum dado de cliente.

## Setup

Requer [uv](https://docs.astral.sh/uv/) (Python 3.12 é instalado automaticamente).

```bash
uv sync
cp .env.example .env   # preencher OPENAI_API_KEY
```

## Comandos

```bash
uv run pytest              # testes offline (cliente OpenAI simulado)
uv run pytest -m live      # teste contra a API real (requer chave)
uv run enter smoke         # 1 chamada estruturada + 1 embedding reais
uv run enter mask arq.txt  # normaliza e mascara entidades de um texto
uv run q1 build            # monta o dataset sintético da Q1 (determinístico, sem LLM)
uv run q1 bench            # benchmark Q1: calibra no dev, mede no test, grava results/<run_id>/
uv run q1 gen-templates    # (já executado) gera os templates via LLM; só refaz o que falta
uv run q2 scrape           # coleta sentenças públicas do TJSP CJPG (cache local, 1 req/2 s)
uv run q2 build            # seleciona e anonimiza o corpus da Q2 em data/rulings
uv run q2 draft-gold       # rascunhos do gold da Q2 via LLM (nunca sobrescreve revisados)
uv run q2 review           # editor do gold (Streamlit): corrigir, aprovar, ancorar citações
uv run q2 gold-status      # quantas sentenças revisadas, citações ancoradas
uv run q2 extract <ruling_id> --config chunk_mini_verif   # extrai e mostra argumentos de 1 sentença
uv run q2 bench            # benchmark Q2: calibra merge no dev, mede no test, grava results/<run_id>/
uv run q2 bench --gold revisado   # só sentenças com gold revisado por humano
uv run q2 ingest --config chunk_mini --split dev   # extrai e grava no store (data/store/q2.sqlite)
uv run q2 import-bench q2-bench-4d5403af77        # importa predições antigas (schema v1, upcast)
uv run q2 review-args      # revisão dos argumentos extraídos (Streamlit): aceitar/editar/rejeitar/adicionar
uv run q2 runs <ruling_id>                 # runs gravadas para a sentença (modelo, prompt, schema)
uv run q2 diff <ruling_id> <run_a> <run_b> # o que mudou entre duas runs
uv run q2 compare <bench_a> <bench_b>      # tabela única entre benchmarks (confere gold e juiz)
uv run ruff check .
```

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `common/llm.py` | Wrapper OpenAI (Responses API): saída estruturada, cache em disco, ledger de custo/latência |
| `common/text.py` | Normalização de OCR/PDF e mascaramento de entidades (nomes, CPF/CNPJ, valores, datas, nº CNJ) |
| `common/runs.py` | `run_id` determinístico (modelo, prompt, schema, dataset, código) e artefatos em `results/<run_id>/` |
| `common/pricing.py` | Tabela de preços por modelo para custo por documento |
| `q1_similarity/` | Dataset sintético, pipeline de similaridade (hash → MinHash → alinhamento → juiz LLM) e benchmark |
| `data/synthetic/q1/` | Casos, documentos e pares rotulados da Q1 ([README](data/synthetic/q1/README.md), [estatísticas](data/synthetic/q1/STATS.md)) |
| `q2_arguments/` | Coleta TJSP, corpus anonimizado, schema, ancoragem de citações, gold + editor, segmentação/chunking, extração, merge, verificador, benchmark, store + revisão + versionamento |
| `q2_arguments/store.py` | SQLite (DDL formato Postgres): sentenças, runs de extração, argumentos e revisões append-only; diff e herança entre runs; exportação para o gold |
| `q2_arguments/versions.py` | Upcasters de schema: linhas antigas nunca reescritas, lidas na versão atual |
| `q2_arguments/review_app.py` | Revisão humana dos argumentos extraídos (Streamlit) |
| `data/rulings/` | Corpus da Q2: 30 sentenças públicas anonimizadas ([README](data/rulings/README.md), [estatísticas](data/rulings/STATS.md)) |
| `data/gold/` | Gold da Q2: argumentos por sentença, rascunho LLM corrigido por humano ([guia](data/gold/GUIDE.md)) |
| `docs/` | *(fase 6)* respostas escritas |

## Reprodutibilidade

- Toda chamada LLM passa pelo cache em `.cache/llm/`, endereçado pelo hash de modelo, mensagens, schema e parâmetros. Reexecutar um benchmark devolve os mesmos números a custo zero.
- Em cache hit, custo e latência reportados são os da chamada original.
- Cada execução grava `spec.json`, `metrics.json` e `report.md` em `results/<run_id>/`.
- Preços em `common/pricing.py` são referência; conferir antes de citar custos.
