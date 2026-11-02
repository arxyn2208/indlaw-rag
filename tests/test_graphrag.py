from indlaw.generate import answer_rag
from indlaw.graphrag import answer_llm_only, graph_expand
from indlaw.index import Index
from indlaw.text import Chunk


def mk(doc, n, text):
    return Chunk(doc_id=doc, title=doc, label=f"para {n}", text=text)


def idx():
    return Index([mk("a.txt", 1, "The bail question under Article 21 arises. Liberty matters."),
                  mk("b.txt", 1, "Delay was held to violate Article 21 of the Constitution."),
                  mk("b.txt", 2, "Unrelated tax matter."),
                  mk("c.txt", 1, "Section 138 of the Act is attracted here."),
                  mk("d.txt", 1, "Section 138 of the Act was also pleaded.")])


def test_expansion_adds_only_other_cases_with_a_literal_shared_provision():
    i = idx()
    added = graph_expand(i, [i.chunks[0]], "bail liberty")
    assert [c.doc_id for c in added] == ["b.txt"]
    assert "added through Article 21" in added[0].label
    assert "Article 21" in added[0].text


def test_unnamed_act_never_links_cases():
    i = idx()
    assert graph_expand(i, [i.chunks[3]], "section 138") == []


class Gen:
    def __init__(self): self.prompts = []
    def complete(self, prompt):
        self.prompts.append(prompt)
        return "- Delay violates liberty [2]. Quote: \"violate Article 21\""


def test_graph_mode_sends_added_passage_and_rag_does_not():
    i = idx()
    g1, g2 = Gen(), Gen()
    r = answer_rag(i, "bail liberty", g1, judge=False)
    gr = answer_rag(i, "bail liberty", g2, judge=False, expand=graph_expand)
    assert "added through" not in g1.prompts[0]
    assert "added through Article 21" in g2.prompts[0]
    assert len(gr.sources) >= len(r.sources)


def test_llm_only_prompt_has_no_passages():
    g = Gen()
    answer_llm_only("What is bail?", g)
    assert "[1]" not in g.prompts[0] and "What is bail?" in g.prompts[0]


def test_comparison_must_fit_all_three_uses_under_every_cap():
    from indlaw.guard import UsageGuard

    g = UsageGuard(per_session=8, per_hour=30, per_day=100, clock=lambda: 1000.0)
    assert g.check("q", 5, 3) is None        # 5 + 3 = 8 fits
    assert g.check("q", 6, 3) is not None    # 6 + 3 = 9 does not
    assert g.check("q", 7, 1) is None and g.check("q", 8, 1) is not None  # single use unchanged
    g.record(28)
    assert g.check("q", 0, 2) is None        # 28 + 2 = 30 fits the hour
    assert g.check("q", 0, 3) is not None    # 28 + 3 = 31 does not
    g2 = UsageGuard(per_session=8, per_hour=1000, per_day=100, clock=lambda: 1000.0)
    g2.record(98)
    assert g2.check("q", 0, 2) is None and g2.check("q", 0, 3) is not None
    g2.record(3)
    assert len(g2._stamps) == 101
