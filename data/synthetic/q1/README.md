# Dataset sintético da Q1

Documentos 100% fictícios (petições iniciais e contestações em pt-BR) agrupados em casos,
com pares rotulados como **cópia**, **versão** ou **diferente**. Nenhum dado real: empresas,
escritórios, pessoas, CPFs e CNPJs são inventados.

Estatísticas: [STATS.md](STATS.md).

## Como é gerado

1. **Templates (LLM, uma vez).** `uv run q1 gen-templates` pede ao `gpt-4.1` os textos com
   placeholders (`{AUTOR_NOME}`, `{VALOR_CAUSA}`...) e grava em `../templates/`:
   - `boilerplate/`: trechos que cada escritório cola em todas as petições de uma área
     (endereçamento, qualificação, justiça gratuita, aplicabilidade do CDC/CLT, fecho).
   - `families/`: conteúdo específico de cada tese × escritório, mais uma paráfrase de cada
     parágrafo e 8 parágrafos extras (matéria-prima das versões).
   - `ementas/`, `contestacoes/`: jurisprudência fictícia e a defesa padrão da ré por tese.

   Saídas com placeholder desconhecido ou dado concreto (R$, data, CPF) são rejeitadas e
   refeitas. Os templates ficam versionados: o resto do pipeline não chama LLM.
2. **Build (determinístico).** `uv run q1 build` (seed 13) monta 60 casos. Mesmo seed e mesmos
   templates geram arquivos byte a byte idênticos (`meta.json` guarda os fingerprints).

## O que há em cada caso

Um réu (cliente), uma tese, um escritório autor. Documentos em ordem de chegada:

| operação | o que muda | rótulo vs. origem |
|---|---|---|
| `original` | nada | — |
| `copia:identica` | nada (mesmos bytes) | cópia |
| `copia:pdf`, `copia:scan_*` | quebras de linha, paginação, hifenização, erros de OCR | cópia |
| `copia:troca_entidades` | outra parte autora, valores, datas, voo, foro | cópia |
| `versao:leve/media/forte` | parágrafos reescritos, adicionados, removidos; ementa trocada (≈3-10%, 12-25%, 28-45% do texto) | versão |
| `contestacao` | peça da ré: mesmas partes, valores e tema | diferente |
| `neg:tese_vizinha_mesmo_escritorio` | mesma ré e mesmo boilerplate, tese vizinha (atraso × cancelamento) | diferente |
| `neg:mesma_tese_outro_escritorio` | mesma tese e mesmos fatos, redação de outro escritório | diferente |
| `neg:outra_tese_mesmo_escritorio` | mesmo boilerplate de consumidor, outro setor | diferente |

Os negativos difíceis existem porque vocabulário jurídico parecido não é sinal de cópia.

## Regra de rótulo

Cada documento é uma sequência de blocos com id de origem. O rótulo de um par vem só dessa
proveniência, nunca de uma métrica de texto:

- **cópia**: mesma sequência de blocos. Valores de entidades e superfície podem mudar.
- **versão**: mesma linhagem de template, sequência de blocos diferente.
- **diferente**: linhagens distintas (outro escritório, outra tese ou outro tipo de peça).

`block_distance` (1 − Jaccard ponderado por palavras dos ids de bloco) mede o tamanho da
edição. É metadado de análise, não entra no rótulo.

## Arquivos

| arquivo | conteúdo |
|---|---|
| `docs/<doc_id>.txt` | texto como chegaria à plataforma (com ruído de superfície) |
| `docs.jsonl` | metadados por documento: caso, tipo, operação, pai, superfície, blocos |
| `pairs.jsonl` | um par por (documento novo, documento anterior do mesmo caso) com rótulo e relação |
| `cases.jsonl` | casos e split `dev`/`test` (estratificado por tese, metade cada) |
| `meta.json` | versão, seed e fingerprints |

Use `dev` para calibrar limiares e `test` só para o número final.
