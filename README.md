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
uv run q1 gen-templates    # (já executado) gera os templates via LLM; só refaz o que falta
uv run ruff check .
```

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `common/llm.py` | Wrapper OpenAI (Responses API): saída estruturada, cache em disco, ledger de custo/latência |
| `common/text.py` | Normalização de OCR/PDF e mascaramento de entidades (nomes, CPF/CNPJ, valores, datas, nº CNJ) |
| `common/runs.py` | `run_id` determinístico (modelo, prompt, schema, dataset, código) e artefatos em `results/<run_id>/` |
| `common/pricing.py` | Tabela de preços por modelo para custo por documento |
| `q1_similarity/` | Gerador do dataset sintético; *(fase 2)* pipeline e benchmark |
| `data/synthetic/q1/` | Casos, documentos e pares rotulados da Q1 ([README](data/synthetic/q1/README.md), [estatísticas](data/synthetic/q1/STATS.md)) |
| `q2_arguments/` | *(fase 3-5)* coleta de sentenças, extração, ancoragem, revisão |
| `docs/` | *(fase 6)* respostas escritas |

## Reprodutibilidade

- Toda chamada LLM passa pelo cache em `.cache/llm/`, endereçado pelo hash de modelo, mensagens, schema e parâmetros. Reexecutar um benchmark devolve os mesmos números a custo zero.
- Em cache hit, custo e latência reportados são os da chamada original.
- Cada execução grava `spec.json`, `metrics.json` e `report.md` em `results/<run_id>/`.
- Preços em `common/pricing.py` são referência; conferir antes de citar custos.
