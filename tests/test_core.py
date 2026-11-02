
import pytest

from indlaw.answer import answer
from indlaw.bm25 import BM25
from indlaw.cli import main
from indlaw.evaluate import evaluate, ndcg_at, recall_at, reciprocal_rank
from indlaw.index import Index
from indlaw.text import make_chunks, split_paragraphs, tokens

# Synthetic text written for these tests. It is not a real judgment.
DOC_A = """Test Case A v State
1. The petitioner was detained for forty days without being produced before a magistrate.
2. The court held that detention without producing the accused within twenty four hours violates personal liberty.
3. Compensation was awarded for the unlawful detention of the petitioner.
4. The state was directed to frame guidelines for arrests."""
DOC_B = """Test Case B v Municipal Body
1. The appellant sought a licence to run a bakery near the market.
2. The licensing authority refused the licence citing zoning rules.
3. The court held that zoning rules apply on the date the application is decided.
4. The appeal was dismissed."""


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "a.txt").write_text(DOC_A)
    (tmp_path / "b.txt").write_text(DOC_B)
    return str(tmp_path)


def test_tokens_drop_stopwords():
    assert tokens("The right of the accused") == ["right", "accus"]


def test_split_on_numbered_paragraphs():
    assert [label for label, _ in split_paragraphs(DOC_A)][:2] == ["para 1", "para 2"]


def test_split_falls_back_to_blocks():
    parts = split_paragraphs("first block\n\nsecond block")
    assert [label for label, _ in parts] == ["part 1", "part 2"]


def test_long_paragraph_is_cut_without_losing_text():
    body = " ".join(f"Sentence number {i} is here today." for i in range(100))
    chunks = make_chunks("d", "t", body, max_words=50)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 60 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == body.split()


def test_bm25_prefers_matching_document():
    bm = BM25([["bail", "accused"], ["licence", "bakery"]])
    assert bm.top(["bakery"], 2)[0][0] == 1
    assert bm.top(["unknown"], 2) == []


def test_index_roundtrip_and_search(folder, tmp_path):
    idx = Index.from_folder(folder)
    path = str(tmp_path / "i.json")
    idx.save(path)
    again = Index.load(path)
    assert again.search_docs("detention without magistrate", k=2)[0] == "a.txt"


def test_answer_cites_source(folder):
    result = answer(Index.from_folder(folder), "Is detention without producing the accused before a magistrate lawful?")
    assert not result.abstained
    assert "Test Case A" in result.render() and "para" in result.render()


def test_answer_abstains_off_topic(folder):
    result = answer(Index.from_folder(folder), "What is the corporate tax rate on cryptocurrency mining?")
    assert result.abstained


def test_metrics():
    assert recall_at(["x", "a"], {"a", "b"}, 2) == 0.5
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert ndcg_at(["a"], {"a"}, 10) == 1.0


def test_evaluate_counts_refusals(folder):
    gold = [
        {"question": "detention without magistrate compensation", "relevant": ["a.txt"]},
        {"question": "bakery licence zoning rules", "relevant": ["b.txt"]},
        {"question": "cryptocurrency mining tax", "relevant": []},
    ]
    out = evaluate(Index.from_folder(folder), gold)
    assert out["recall@5"] == 1.0 and out["correctly_refused"] == 1


def test_cli_build_and_ask(folder, tmp_path, capsys):
    out = str(tmp_path / "i.json")
    assert main(["build", folder, "--out", out]) == 0
    assert main(["ask", "zoning rules licence date", "--index", out]) == 0
    assert "Test Case B" in capsys.readouterr().out


def test_empty_folder_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        Index.from_folder(str(tmp_path))


# ---- generation (uses a fake generator: no model is called) ----

from indlaw.generate import answer_rag, build_prompt, parse_statements
from indlaw.text import Chunk


class Fake:
    def __init__(self, reply):
        self.reply, self.prompt = reply, None

    def complete(self, prompt):
        self.prompt = prompt
        return self.reply


def test_prompt_numbers_passages_and_restricts_sources(folder):
    chunks = [c for c, _ in Index.from_folder(folder).search("detention magistrate", k=2)]
    prompt = build_prompt("q?", chunks)
    assert "[1]" in prompt and "[2]" in prompt and "ONLY the numbered passages" in prompt


def test_answer_rag_uses_generator_and_validates(folder):
    q = "Is detention without producing the accused before a magistrate lawful?"
    chunks = [c for c, _ in Index.from_folder(folder).search(q, k=4)]
    words = chunks[0].text.split()[:8]
    quote = " ".join(words)
    fake = Fake(f'- {quote} was stated in the passage. [1] "{quote}"')
    out = answer_rag(Index.from_folder(folder), q, fake)
    assert out.mode == "rag" and len(out.shown) == 1 and out.shown[0].passage == 1
    assert "Passages:" in fake.prompt


def test_answer_rag_abstains_before_calling_generator(folder):
    fake = Fake("should not be used")
    out = answer_rag(Index.from_folder(folder), "cryptocurrency mining tax rate", fake)
    assert out.mode == "abstained" and fake.prompt is None


def test_answer_rag_respects_model_refusal(folder):
    out = answer_rag(Index.from_folder(folder), "detention magistrate accused lawful", Fake("INSUFFICIENT"))
    assert out.mode == "abstained"


# ---- hybrid retrieval with a fake embedder (checks the fusion path, not embedding quality) ----

def test_hybrid_search_fuses_lexical_and_dense(folder, monkeypatch):
    np = pytest.importorskip("numpy")
    from indlaw import dense

    class FakeEmb:
        def embed(self, texts):
            for t in texts:  # "semantic" axis: 1 for bakery-ish text, 0 otherwise
                yield np.array([1.0, 0.0] if any(w in t.lower() for w in ("bakery", "licence", "shop")) else [0.0, 1.0])

    monkeypatch.setattr(dense.DenseIndex, "_load_embedder", staticmethod(lambda model: FakeEmb()))
    idx = Index.from_folder(folder)
    idx.add_dense()
    assert idx.dense is not None
    hits = idx.search("shop permit", k=2, mode="hybrid")  # no lexical overlap; dense finds the bakery case
    assert hits[0][0].doc_id == "b.txt"
    assert idx.search("shop permit", k=2, mode="bm25") == []
    with pytest.raises(ValueError):
        Index.from_folder(folder).search("x", mode="hybrid")


PASSAGE = Chunk("d1", "Case A", "para 5", "The petitioner argued that delay entitles a prisoner to approach this Court. "
                "The Court rejected that argument and held that only subsequent events after the verdict may be challenged.")


def _stmts(reply):
    return {s.status: s for s in parse_statements(reply, [PASSAGE])}


def test_statement_with_quote_from_the_cited_passage_is_shown_with_context_warning():
    st = _stmts('- The Court rejected the argument about delay. [1] "The Court rejected that argument and held that only subsequent events"')
    assert "verified_quote" in st
    assert "rejected view" in st["verified_quote"].warning  # passage reports a rejected argument


def test_regression_stretched_claim_with_invented_quote_is_removed():
    # Known failure: a claim that goes beyond the passage, supported by words the passage does not contain.
    st = _stmts('- A prisoner can always approach the court if rights are infringed. [1] "a prisoner may approach this Court whenever rights are infringed"')
    assert "quote_not_found" in st and "verified_quote" not in st


def test_regression_yes_claim_with_irrelevant_quote_is_removed():
    st = _stmts('- Yes, complaints about prison conditions succeed always. [1] "The Court rejected that argument and held that only subsequent events"')
    assert "off_point" in st


def test_uncited_invalid_and_quoteless_statements_are_removed():
    st = _stmts('- The state appealed the decision later today in court.\n- Compensation was awarded to him later. [7] "Compensation was awarded to him"\n- The Court held something about delay. [1]')
    assert {"uncited", "invalid", "no_quote"} <= set(st)


def test_answer_rag_abstains_when_no_statement_survives(folder):
    out = answer_rag(Index.from_folder(folder), "Is detention without producing the accused before a magistrate lawful?",
                     Fake("Detention without a magistrate is plainly unlawful in every case."))
    assert out.mode == "abstained" and out.statements


def test_chat_endpoint_retries_then_fails_cleanly(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    import pytest

    from indlaw.generate import ChatEndpoint, GenerationError

    calls = {"n": 0}

    def fake_urlopen(req, timeout=0):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, io.BytesIO(b""))
        return io.BytesIO(b'{"choices":[{"message":{"content":" ok "}}]}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert ChatEndpoint("http://x/v1", "m", "k", retries=3, backoff=0).complete("hi") == "ok"
    assert calls["n"] == 3

    def always_503(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, io.BytesIO(b""))

    monkeypatch.setattr(urllib.request, "urlopen", always_503)
    with pytest.raises(GenerationError):
        ChatEndpoint("http://x/v1", "m", "k", retries=1, backoff=0).complete("hi")


def test_stemming_matches_inflections():
    assert tokens("accused") == tokens("accuse") == tokens("accusation") == tokens("accuses")
    assert tokens("detained") == tokens("detaining") == tokens("detain")
    assert tokens("prisoners") == tokens("prisoner")


def test_stemming_does_not_overmerge_or_claim_irregulars():
    assert tokens("detain") != tokens("detention")  # known limit, not matched
    assert tokens("prison") != tokens("prisoner")  # derivational, not merged
    assert tokens("bail") != tokens("bale")
    assert tokens("court") != tokens("count")
    assert tokens("judge") != tokens("judgment")
