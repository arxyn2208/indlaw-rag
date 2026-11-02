# indlaw

A small retrieval system for Indian court judgments. You give it a folder of judgment text files. It returns the passages and sentences that best match a question, each with the case name and paragraph number, and it refuses when the corpus does not seem to cover the question.

It runs in two modes, and the interface says which one is active:

- **RAG mode**: a language model writes the answer from the retrieved passages only, citing them as [1], [2]. The citations are checked afterwards. Sentences with no citation, and citation numbers that do not exist, are flagged.
- **Extractive mode** (no model configured): the answer is sentences copied from the judgments, with citations. This is not RAG, because no model generates anything.

## How it works

1. **Split** (`indlaw/text.py`). Each judgment is split on its numbered paragraphs, so a citation can say "para 12". Long paragraphs are cut by sentence. Judgments without numbered paragraphs are split on blank lines.
2. **Index** (`indlaw/bm25.py`, `indlaw/index.py`). Passages are ranked with Okapi BM25 after lower-casing and removing common words.
3. **Optional semantic search** (`indlaw/dense.py`). `build --dense` embeds passages locally with `BAAI/bge-small-en-v1.5` (fastembed, ONNX, CPU) and fuses the result with BM25 by reciprocal rank fusion.
4. **Generate** (`indlaw/generate.py`). The question and numbered passages go to a model with an instruction to use only those passages. Backends: a local GGUF file (`INDLAW_GGUF=path`, needs llama-cpp-python), or any OpenAI-compatible endpoint (`INDLAW_LLM_URL`, `INDLAW_LLM_MODEL`, `INDLAW_LLM_KEY`). Ollama on `localhost:11434` needs no key. Hosted free tiers need your own key.
5. **Abstain and extract** (`indlaw/answer.py`). The top passages are searched for sentences that share the most key terms with the question. If the best passage contains fewer than half of the question's key terms, the system refuses and says why.
6. **Evaluate** (`indlaw/evaluate.py`). Recall@k, MRR and nDCG at document level against a gold file you write, with a bootstrap confidence interval, plus how many out-of-corpus questions were refused.

## Use it

```bash
python -m indlaw build corpus/              # corpus/ holds your .txt judgments
python -m indlaw build corpus/ --dense      # optional: add semantic search
python -m indlaw ask "Can a prisoner complain to a court about ill-treatment in jail?"   # add --no-llm for quotes only
python -m indlaw eval gold.json             # see format below
pip install streamlit && streamlit run app.py   # demo, reads ./corpus or $INDLAW_CORPUS
```

Gold file: a JSON list. Use `{"question": "...", "relevant": ["file.txt"]}` for answerable questions and `"relevant": []` for questions the corpus cannot answer.

The corpus is not part of this repository. Judgment texts have their own terms of use, so put your own files in `corpus/`, which is git-ignored.

## Install on a Mac (Apple silicon, 8 GB is enough for the API route)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[app]"          # core has no dependencies; the app extra adds Streamlit
mkdir corpus                      # put your own judgments here as .txt files
python -m indlaw build corpus/    # writes index.json (git-ignored)
export INDLAW_CORPUS=corpus
export INDLAW_LLM_URL=https://generativelanguage.googleapis.com/v1beta/openai
export INDLAW_LLM_MODEL=gemini-flash-lite-latest
export INDLAW_LLM_KEY=...        # your own key, typed into your shell. Never commit it.
streamlit run app.py
```

Without the three INDLAW_LLM variables the app runs in quotes-only mode and says so. Model names change: check which models your key can use before relying on one. Transient 429 and 5xx replies are retried; if they persist the app shows an error and no answer.

## Tests

```bash
pip install pytest && python -m pytest
```

The tests use short synthetic passages written for them (not real legal material), and a fake model and embedder. They cover splitting, BM25, the index, citations, abstention, the metrics and the command line. They do not measure retrieval quality on real judgments.

## Limits

- Without `--dense`, matching is lexical only. Hybrid search is implemented and unit-tested with a stand-in embedder, but its retrieval quality has not been measured on real judgments.
- A small local model (tested: Qwen2.5-1.5B-Instruct, 4-bit) gives slow (about 40 s on 2 CPUs) and sometimes muddled answers. Use a larger model for better ones. The citation check does not verify that a cited passage supports the claim.
- The abstention threshold (0.5) is a starting value. On one local trial it refused many answerable questions, so calibrate it on your own gold set before relying on it.
- The corpus decides what it can answer. It only covers whatever judgments you load.
- Not legal advice.
