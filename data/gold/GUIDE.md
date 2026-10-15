# Guia de rotulagem: argumentos de sentenças (Q2)

Gold = o que um(a) advogado(a) da defesa quer encontrar depois no banco de argumentos: **cada
questão que o juiz decidiu, com a razão da decisão e o trecho que a sustenta**.

Arquivos: `data/gold/<ruling_id>.json`, um por sentença (texto em `data/rulings/text/`).
Schema: `q2_arguments/schema.py` (`q2-args-v1`).

## Fluxo

1. `uv run q2 draft-gold` gera rascunhos com LLM (`status: "rascunho"`). Citações já vêm
   ancoradas no texto em código; as que não acharam o trecho ficam marcadas.
2. `uv run q2 review` abre o editor (Streamlit). Para cada sentença: corrigir, apagar,
   acrescentar argumentos; conferir resultado; **Marcar revisado**.
3. Só arquivos `revisado` contam como gold no benchmark. Reexecutar o rascunho nunca
   sobrescreve um arquivo revisado.

## O que é UM argumento

Uma **tese sobre uma questão** que o juiz enfrentou e decidiu, com fundamentação própria.

- Uma preliminar = um argumento (ex.: ilegitimidade passiva; falta de interesse de agir;
  incompetência; inépcia; impugnação à gratuidade).
- Uma prejudicial = um argumento (prescrição, decadência).
- No mérito, um argumento por questão decidida: existência da relação/contratação, falha na
  prestação do serviço, excludente (fortuito externo, culpa exclusiva de terceiro/vítima),
  dano moral configurado ou não, dano material, repetição em dobro, inversão do ônus da prova
  quando discutida, abusividade de cláusula/taxa etc.
- Valor da indenização = argumento `quantum` separado do "há dano moral?" quando o juiz
  fundamenta o valor (critérios, redução do pedido).
- **Não** dividir uma mesma questão em vários argumentos só porque o juiz usou vários
  parágrafos ou várias citações de jurisprudência. Junte as citações no mesmo argumento.
- **Não** juntar questões diferentes num argumento só (ex.: "responsabilidade e dano moral").

## O que NÃO é argumento

- Resumo das alegações das partes no relatório, se o juiz não decidir sobre elas.
- Fórmulas padrão do dispositivo sem fundamentação: custas, honorários, correção e juros
  aplicados mecanicamente, "P.R.I.", trânsito em julgado.
- Julgamento antecipado (art. 355 CPC) só quando for mera formalidade. Se o juiz rejeitar
  pedido de prova de uma parte com fundamentação, é argumento `preliminar`.
- Sentença sem mérito (extinção por abandono, cumprimento, homologação): só o fundamento da
  extinção, se houver um real; pode ter zero argumentos.

## Campos

| campo | regra |
|---|---|
| `titulo` | A tese em uma linha, do ponto de vista de quem a levantou. Ex.: "Ilegitimidade passiva da plataforma". |
| `resumo` | 1 a 3 frases: o que o juiz decidiu e o porquê, sem copiar a sentença. |
| `tipo` | `preliminar` (processual), `prejudicial` (prescrição/decadência), `merito`, `quantum` (valor de indenização/repetição), `acessorio` (juros, correção, honorários, gratuidade **quando discutidos**). |
| `tema` | Rótulo curto e reutilizável: "ilegitimidade passiva", "fortuito externo", "dano moral in re ipsa", "repetição em dobro". |
| `suscitado_por` | `autor`, `reu` ou `juizo` (levantado de ofício). |
| `decisao` | O juiz `acolhido`, `rejeitado` ou `parcial` a tese do título. De ofício: `acolhido`. |
| `fundamento_legal` | Dispositivos, súmulas, temas e precedentes citados para esta questão, na forma curta ("art. 14 do CDC", "Súmula 479 do STJ", "Tema 1417 do STF"). |
| `citacoes` | 1 a 3 trechos **literais** da sentença, de preferência da fundamentação, que sustentam a decisão. Uma ou duas frases cada; sem "...". |
| `secao` | Onde a questão é decidida: normalmente `fundamentacao`. |

`resultado` (nível da sentença): `procedente`, `parcialmente_procedente`, `improcedente`,
`extinto_sem_merito`, `homologacao`, `outro`.

## Citações e ancoragem

- O editor ancora cada citação no texto ao salvar. Citação não ancorada = não está no texto;
  corrigir copiando o trecho exato. Um arquivo só pode ser marcado revisado com 100% das
  citações ancoradas.
- O texto já está anonimizado (`[AUTOR]`, `[PESSOA_1]`...): copie do texto mostrado.

## Casos difíceis

- Juiz acolhe a tese em parte (ex.: dano material sim, parte dos valores não): `parcial`.
- Tese do réu rejeitada ⇔ tese do autor acolhida: registrar uma vez, do lado de quem a
  levantou primeiro no processo (geralmente o réu, na contestação, para preliminares e
  excludentes; o autor para o pedido de mérito).
- Embargos de declaração: cada vício alegado (omissão, contradição...) é um argumento.
