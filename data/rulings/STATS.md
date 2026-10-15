# Corpus Q2 (q2-tjsp-v1)

Gerado por `uv run q2 build`. Não editar à mão.

- Pool coletado: 300 resultados, 268 processos únicos.
- Candidatos (classe cível permitida, texto inline completo): 222.
- Selecionadas: 30 (10 dev, 20 test).

## Tamanho (tokens o200k)

| mín | p25 | mediana | p75 | máx |
|---|---|---|---|---|
| 270 | 3043 | 4357 | 7284 | 12654 |

| faixa | n |
|---|---|
| curto | 7 |
| longo | 8 |
| medio | 11 |
| muito_longo | 4 |

| domínio | n |
|---|---|
| aereo | 10 |
| bancario | 13 |
| varejo_tech | 7 |

| resultado (heurística) | n |
|---|---|
| extinto | 2 |
| improcedente | 11 |
| parcial | 10 |
| procedente | 7 |

| classe | n |
|---|---|
| Busca e Apreensão em Alienação Fiduciária | 1 |
| Procedimento Comum Cível | 28 |
| Procedimento de Repactuação de Dívidas (Superendividamento) | 1 |

## Anonimização (trocas no corpus selecionado)

| rótulo | trocas |
|---|---|
| AUTOR | 76 |
| OAB | 3 |
| PESSOA | 51 |
| REU | 1 |
| TELEFONE | 3 |

Palavras de nomes privados que sobraram (protegidas por regra de lugar/nome público): 10 no corpus todo.
