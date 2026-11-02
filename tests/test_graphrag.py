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
