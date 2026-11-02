import pytest

from indlaw.kmap import build_map, find_provisions, to_dot
from indlaw.text import make_chunks


def test_finds_articles_and_sections_with_their_sentence():
    text = ("Article 21 of the Constitution of India protects life. The accused was charged under "
            "Section 302 of the Indian Penal Code, 1860. Section 41 CrPC was also cited.")
    got = dict(find_provisions(text))
    assert "Article 21" in got
    assert "Section 302, Indian Penal Code, 1860" in got
    assert "Section 41, CrPC" in got
    assert got["Article 21"].startswith("Article 21 of")


def test_no_mention_means_no_edge():
    assert find_provisions("The Court held that the claim was allowed. Section alone is not a mention.") == []


def test_edges_carry_paragraph_and_evidence_and_dot_is_valid():
    raw = "Test v Case\n1. Facts.\n2. The petitioner relied on Article 14 of the Constitution.\n3. Held for the petitioner."
    edges = build_map(make_chunks("t.txt", "Test v Case", raw))
    assert len(edges) == 1
    assert edges[0].label == "para 2" and "Article 14" in edges[0].evidence
    dot = to_dot(edges)
    assert dot.startswith("digraph") and "para 2" in dot


def test_sample_corpus_has_no_provisions_so_map_is_empty():
    import os

    from indlaw.index import Index
    folder = os.path.join(os.path.dirname(__file__), "..", "sample_corpus")
    assert build_map(Index.from_folder(folder).chunks) == []


def test_article_lists_give_one_mention_each():
    got = [p for p, _ in find_provisions("Under Articles 14 , 19 and 21 of the Constitution the reasons must be rational.")]
    assert got == ["Article 14", "Article 19", "Article 21"]


def test_code_of_criminal_procedure_is_found():
    got = [p for p, _ in find_provisions("She relied on Section 41 of the Code of Criminal Procedure, 1973.")]
    assert got == ["Section 41, Code of Criminal Procedure, 1973"]


def test_app_provision_map_view_on_fixture(tmp_path):
    import os

    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    (tmp_path / "a.txt").write_text("CASE TITLE: A v. B\n\n1. The court considered Article 21 of the Constitution of India.\n", encoding="utf-8")
    os.environ["INDLAW_CORPUS"] = str(tmp_path)
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "app.py")).run()
    at.radio[0].set_value("Provision map").run()
    assert not at.exception
    assert any("1 mentions" in m.value for m in at.markdown)
    assert "Article 21" in str(at.dataframe[0].value)


def test_bare_the_act_is_not_merged_with_a_named_act():
    got = find_provisions("Section 138 of the Negotiable Instruments Act, 1881 applies. Section 138 of the Act is attracted.")
    labels = [g[0] for g in got]
    assert labels[0] == "Section 138, Negotiable Instruments Act, 1881"
    assert labels[1] == "Section 138 (Act not named in the sentence)"


def test_unnamed_act_sections_do_not_link_different_cases():
    a = make_chunks("a.txt", "A v. B", "1. Section 7 of the Act applies here.\n2. x.\n3. y.", 200)
    b = make_chunks("b.txt", "C v. D", "1. Section 7 of the Act is different.\n2. x.\n3. y.", 200)
    provs = {e.provision for e in build_map(a + b)}
    assert len(provs) == 2


def test_vis_html_escapes_markup_in_labels():
    from indlaw.kmap import Edge, to_vis_html

    html = to_vis_html([Edge("a.txt", "</script><img src=x onerror=alert(1)>", "para 1", "Article 21", "s")])
    assert "</script><img" not in html
    assert html.count("</script>") == 2  # only our own two closing tags


def test_sc_sample_has_forty_distinct_nonempty_judgments():
    import glob
    import hashlib
    import os
    from pathlib import Path

    if not os.path.isdir("sc_sample"):
        pytest.skip("sc_sample not present")
    files = [Path(f) for f in glob.glob("sc_sample/*.txt")]
    assert len(files) == 40
    assert len({hashlib.sha256(f.read_bytes()).hexdigest() for f in files}) == 40
    assert all(len(f.read_text(encoding="utf-8").split()) > 1000 for f in files)
    assert Path("sc_sample/SOURCE.md").read_text(encoding="utf-8").count("_EN.pdf |") == 40
