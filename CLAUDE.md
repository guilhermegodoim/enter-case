# Enter technical case

Take-home case from Enter. Goal: answer both questions below, and back each answer with a working mini project plus a structured benchmark. Implementation approach not decided yet; don't assume one.

## About Enter

- Brazilian legal-AI startup, founded 2023 by Mateus Costa-Ribeiro (CEO), Henrique Vaz (CPO), Mike Mac-Vicar (CTO).
- Product: EnterOS, "the operating system for legal departments". Customers are in-house legal teams of large companies (Nubank, Mercado Livre, LATAM, Azul, Bradesco, Santander, iFood, Airbnb, 40+ others), not law firms.
- Focus: contencioso de massa (high-volume litigation), consumer/civil and labor (trabalhista). Sectors: aviation, finance, tech, retail.
- Scale: 300k+ lawsuits/year run through EnterOS. AI supports every step of the process, from claim calculation to hearing, with full audit and editing. Client teams and their law firms keep control of legal strategy.
- Series B (May 2026): US$100M+ at US$1.2B valuation, led by Founders Fund, with Sequoia, Ribbit, ONEVC, Atlantico, Kaszek. First AI unicorn in Latin America.
- Runs on OpenAI under Zero Data Retention.

Sources: getenter.ai/sobre-nos, getenter.ai Series B announcement, Brazilian press coverage (Forbes Brasil, Bloomberg Línea, Exame).

## Why this case matters to Enter

- Mass litigation means the same plaintiff firms file thousands of near-identical petições from templates, changing names, values and dates. Spotting copies and versions is core to their workflow, not an edge case.
- Legal text is full of shared boilerplate, so "similar vocabulary" is everywhere and is not a signal on its own.
- An argument database from sentenças feeds defense strategy: which arguments win, with which judges, in which kinds of case.
- Every step is audited and editable by humans, so outputs must be traceable back to the source text.

## What to consider when designing

- Volume: many documents per day. Cost and latency per document matter as much as accuracy.
- Comparison scope in Q1 is documents of the same caso, not the whole corpus.
- Documents are long, in pt-BR, often scanned/OCR'd, and heavily templated.
- Errors have different costs: a missed copy wastes work; a false "copy" can hide a document that matters.
- LLM outputs must be grounded in the source text and reviewable by a human.
- Results must be measurable: every design claim in the written answers should be backed by benchmark numbers.
- Models, prompts and schemas will change over time; results must stay comparable across versions.
- Data is sensitive. Use only public court decisions or synthetic documents, never real client data.

## Domain model

- A client has multiple legal cases; each case belongs to exactly one client.
- A case contains multiple documents; every document belongs to exactly one case.
- The platform processes a large number of legal documents per day.

## The case (translated from Portuguese)

### Question 1: Comparing similar documents

At Enter, we want to identify when two legal documents are very similar, even if they are not identical. For example: two petitions may share the same main legal thesis but have different names, amounts, dates and some different paragraphs.

For each new document added to the platform, we want to compute its similarity to other documents in the same case and decide whether it looks like a copy, a modified version or a different document.

Design a solution to compute and use this similarity between documents.

Your answer should cover:

- How you would represent each document for comparison: raw text, chunks, embeddings, hashes, MinHash, or another approach.
- How you would tell near-identical documents apart from documents that merely use similar legal vocabulary.
- How you would handle long documents that don't fit entirely in an LLM's context.
- How you would evaluate whether the solution is working well.
- What you would do when a new document is identified as a likely copy or version of an existing document.
- What trade-offs your solution makes in terms of cost, latency and precision.

### Question 2: Extracting arguments from court rulings

At Enter, we want a feature that receives a court ruling (sentença) and extracts the main arguments used by the judge. The goal is to build a database of legal arguments. For each new ruling added to the platform, we want to extract and store these arguments in a structured way.

Design a solution for this feature.

Your answer should cover:

- What structured output you would store. Example: argument title, summary, quoted excerpt, position in the document, confidence and argument type.
- Assuming you use an LLM for this task, how you would reduce the risk of the model inventing arguments that are not in the text.
- How you would handle long rulings that don't fit entirely in the model's context.
- How you would let a human review, edit or reject extracted arguments.
- How you would measure extraction quality.
- How you would version the extraction if the prompt, the model or the schema change in the future.
