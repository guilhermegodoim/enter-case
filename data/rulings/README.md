# Corpus da Q2: sentenças públicas do TJSP

30 sentenças cíveis de 1º grau (consumidor: aéreo, bancário, varejo/tecnologia), coletadas da
[Consulta de Julgados de 1º Grau (CJPG)](https://esaj.tjsp.jus.br/cjpg/) do TJSP. Decisões
públicas; mesmo assim, nomes de pessoas físicas privadas e identificadores foram anonimizados.

Estatísticas: [STATS.md](STATS.md).

## Arquivos

| arquivo | conteúdo |
|---|---|
| `queries.json` | consultas (termo + janela de datas fixa no passado) e nº de páginas por consulta |
| `rulings.jsonl` | metadados por sentença: processo, classe, assunto, magistrado, vara, data, domínio, faixa de tamanho, split, hashes |
| `text/<ruling_id>.txt` | texto limpo e anonimizado. **É o texto canônico**: offsets de citações referem-se a ele |
| `meta.json` | versão do corpus, contagens, modelo/prompt usados na anonimização |
| `raw/` | HTML bruto da coleta (fora do git, `.gitignore`) |

## Como é gerado

```bash
uv run q2 scrape   # 10 consultas x 3 páginas, 1 requisição a cada 2 s, cache em raw/
uv run q2 build    # filtra, seleciona, anonimiza, grava text/ + rulings.jsonl + STATS.md
```

1. **Coleta.** A lista de resultados do CJPG já traz o inteiro teor de cada sentença, então
   não há download de PDF. A paginação depende do cookie da sessão (um cliente por consulta).
   Páginas que voltam sem tabela de resultados (erro transitório do e-SAJ) são refeitas.
2. **Filtro.** Classes cíveis de conhecimento (sem penal, sem cumprimento de sentença) e texto
   inline < 49 mil caracteres: o CJPG corta o inteiro teor inline perto de 50 mil, e sentença
   truncada não serve de gold. Isso excluiu 5 sentenças cíveis do pool.
3. **Seleção determinística** (sem sorteio dependente de ordem): ordem por hash do nº do
   processo; cotas por faixa de tamanho em tokens (curto < 3k, médio < 6k, longo < 9k,
   muito longo ≥ 9k), alternando domínios; no máximo 2 sentenças por magistrado; mais 2
   sentenças sem mérito (extinção), que testam se o extrator inventa argumentos.
   As faixas longas existem para testar chunking na fase 4 (orçamento de chunk menor que a
   sentença inteira).
4. **Split.** 1 em cada 3 sentenças (por nº do processo) vai para `dev` (ajuste de prompt);
   o resto é `test`.
5. **Anonimização.** Ver docstring de `q2_arguments/corpus.py`:
   - partes do cabeçalho + pessoas listadas por LLM (`gpt-4.1`, com papel). O LLM só lista
     nomes; a troca é feita em código, por nome completo e, depois, prenome/sobrenome isolado;
   - CPF, RG, telefone, e-mail, CEP e OAB por regex;
   - mantidos: empresas (os réus), magistrados e relatores, autores de doutrina, comarca/foro.

   Rótulos: `[AUTOR]`, `[REU]`, `[PESSOA_n]`, `[CPF]`... consistentes dentro de cada sentença.
   `STATS.md` reporta quantas palavras de nomes privados sobraram e por quê (todas conferidas:
   magistrados homônimos e nomes de lugar).

## Limites conhecidos

- Só TJSP, só consumidor. A busca nacional da Justiça do Trabalho (jurisprudencia.jt.jus.br)
  bloqueou acesso automatizado (HTTP 403), então não há sentenças trabalhistas.
- A busca do CJPG é por texto livre: o assunto real vem do campo `assunto` do tribunal.
- A janela de datas é fixa, mas o tribunal pode republicar ou retirar decisões; `source_sha256`
  em `rulings.jsonl` permite verificar se o texto coletado mudou.
