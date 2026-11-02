"""Streamlit demo. Point INDLAW_CORPUS at a folder of .txt judgments (default ./corpus)."""

import os

import streamlit as st

from indlaw.answer import answer
from indlaw.generate import REASONS, GenerationError, answer_rag, generator_from_env
from indlaw.graphrag import answer_llm_only, graph_expand
from indlaw.guard import CappedGenerator, UsageGuard
from indlaw.index import Index

st.set_page_config(page_title="indlaw", layout="wide")
st.title("Indian case law search")
generator = generator_from_env()
st.caption("Retrieval-augmented answers: a language model is given retrieved passages and asked to quote them. Quotations are checked against the passages. The claims themselves are not verified."
           if generator else
           "Extractive mode: no language model is configured, so this shows quotes from the judgments, "
           "not a generated answer. Set INDLAW_GGUF or INDLAW_LLM_URL to enable generation.")


@st.cache_resource
def load(folder: str) -> Index:
    return Index.from_folder(folder)


@st.cache_resource
def usage_guard() -> UsageGuard:
    return UsageGuard(per_session=int(os.getenv("INDLAW_SESSION_LIMIT", "8")),
                      per_hour=int(os.getenv("INDLAW_HOURLY_LIMIT", "30")),
                      per_day=int(os.getenv("INDLAW_DAILY_LIMIT", "100")))


folder = os.getenv("INDLAW_CORPUS", "corpus")
if not os.path.isdir(folder) and os.path.basename(os.path.normpath(folder)) == "sc_2019_sample" and os.path.isdir("sc_sample"):
    folder = "sc_sample"  # the sample was renamed; keep older INDLAW_CORPUS settings working
if not os.path.isdir(folder):
    st.error(f"Corpus folder '{folder}' not found. Put .txt judgments there, or set INDLAW_CORPUS.")
    st.stop()

index = load(folder)
if os.path.basename(os.path.normpath(folder)).startswith("sc_"):
    st.warning("Prototype on real Supreme Court of India judgments (2016-2022, CC-BY-4.0, Dattam Labs via AWS Open Data, "
               "see sc_sample/SOURCE.md). Text came from PDFs and may contain extraction errors. Paragraph labels "
               "come from this tool's splitter, not the court. Answers can miss the question and are not legal advice.")
if os.path.basename(os.path.normpath(folder)) == "sample_corpus":
    st.warning("Prototype. The three cases loaded here are FICTIONAL, written for this demo. Nothing below is a "
               "real judgment or legal advice, and answers can miss the question.")
docs = len({c.doc_id for c in index.chunks})
st.sidebar.metric("Judgments loaded", docs)
st.sidebar.metric("Passages indexed", len(index.chunks))
min_cov = st.sidebar.slider("Abstention threshold (term coverage)", 0.0, 1.0, 0.5, 0.05,
                            help="The system refuses when the best passage covers less of the question's key terms than this.")

def provision_map_tab() -> None:
    from collections import Counter

    from indlaw.kmap import build_map, to_dot

    edges = build_map(index.chunks)
    st.caption("Literal mentions only: each row is a paragraph whose text contains the provision, with the sentence as "
               "evidence. It does not say the case applies or interprets the provision. Pattern matching can miss or "
               "mis-split mentions, and accuracy has not been measured. Case citations and topics are not extracted.")
    if not edges:
        st.info("No Article or Section mentions found in this corpus.")
        return
    counts = Counter(e.provision for e in edges)
    chosen = st.multiselect("Provisions", [p for p, _ in counts.most_common()], default=[p for p, _ in counts.most_common() if "not named" not in p][:12])
    titles = sorted({e.title for e in edges})
    case = st.selectbox("Case", ["All cases"] + titles)
    shown = [e for e in edges if e.provision in chosen and (case == "All cases" or e.title == case)]
    st.write(f"{len(shown)} mentions in {len({e.doc_id for e in shown})} cases")
    if shown:
        if st.checkbox("Interactive graph (drag, zoom, pan)", value=True):
            import streamlit.components.v1 as components

            from indlaw.kmap import to_vis_html
            components.html(to_vis_html(shown), height=640)
            st.caption("Boxes are cases and ovals are provisions. An edge exists only where a paragraph mentions the provision; "
                       "hover an edge for the paragraph count.")
        else:
            st.graphviz_chart(to_dot(shown, top=len(chosen) or None))
        st.dataframe([{"Case": e.title, "Paragraph": e.label, "Provision": e.provision, "Sentence": e.evidence}
                      for e in shown])


def compare_tab() -> None:
    st.caption("Same question and same model in three modes. LLM only gets no passages. RAG gets the top search "
               "passages. GraphRAG (experimental) gets the same passages plus a few from other cases that name the "
               "same Article or Section, each tagged with the provision that linked it. None of this shows GraphRAG "
               "is better. One comparison counts as 3 model uses, and each mode has its own deadline.")
    if generator is None:
        st.info("A language model is needed for this view and none is configured.")
        return
    q = st.text_input("Question to compare", key="compare_q")
    if not q:
        return
    uses = st.session_state.get("model_uses", 0)
    blocked = usage_guard().check(q, uses + 2)
    if blocked:
        st.info(f"{blocked} A comparison needs 3 model uses, so it was not run.")
        return
    for _ in range(3):
        usage_guard().record()
    st.session_state["model_uses"] = uses + 3
    cols = st.columns(3)
    budget = float(os.getenv("INDLAW_TIME_BUDGET", "60"))
    with cols[0]:
        st.subheader("LLM only")
        st.warning("No sources given. Nothing here can be checked against a judgment, and it may invent cases.")
        try:
            with st.spinner("Asking the model with no passages..."):
                st.write(answer_llm_only(q, CappedGenerator(generator, limit=1, time_budget=budget)))
        except GenerationError as err:
            st.warning(f"No answer in this mode ({err}).")
    for col, name, expand in ((cols[1], "RAG", None), (cols[2], "GraphRAG (experimental)", graph_expand)):
        with col:
            st.subheader(name)
            try:
                with st.spinner("Generating from retrieved passages..."):
                    rag = answer_rag(index, q, CappedGenerator(generator, time_budget=budget), min_coverage=min_cov,
                                     expand=expand)
            except GenerationError as err:
                st.warning(f"No answer in this mode ({err}).")
                continue
            if rag.mode == "abstained":
                st.warning(f"No answer: {rag.text}")
            else:
                for stmt in rag.shown:
                    chunk = rag.sources[stmt.passage - 1]
                    st.markdown(stmt.claim)
                    st.markdown(f"> \"{stmt.quote}\"")
                    st.markdown(f":green[Quote found word for word in {chunk.title}, {chunk.label}. "
                                f"This does not prove the claim follows from it.]")
            added = [c for c in rag.sources if "added through" in c.label]
            with st.expander(f"Passages given to the model ({len(rag.sources)}" + (f", {len(added)} added by the graph)" if expand else ")")):
                for n, c in enumerate(rag.sources, start=1):
                    st.markdown(f"**[{n}] {c.title}, {c.label}**")
                    st.write(c.text)


view = st.radio("View", ["Ask", "Compare 3 modes", "Provision map"], horizontal=True)
if view == "Compare 3 modes":
    compare_tab()
    st.stop()
if view == "Provision map":
    provision_map_tab()
    st.stop()
question = st.text_input("Ask about the loaded judgments")
if question:
    use_model = generator is not None
    if use_model:
        blocked = usage_guard().check(question, st.session_state.get("model_uses", 0))
        if blocked:
            use_model = False
            st.info(f"{blocked} Showing extractive quotes instead (not a generated answer).")
    if use_model:
        usage_guard().record()
        st.session_state["model_uses"] = st.session_state.get("model_uses", 0) + 1
        generator = CappedGenerator(generator, time_budget=float(os.getenv("INDLAW_TIME_BUDGET", "60")))
    if use_model:
        rag = None
        try:
            with st.spinner("Generating from the retrieved passages..."):
                rag = answer_rag(index, question, generator, min_coverage=min_cov)
        except GenerationError as err:
            st.warning(f"The model did not answer ({err}). Showing extractive quotes instead (not a generated answer).")
            use_model = False
    if use_model:
        if rag.mode == "abstained":
            st.warning(f"No answer: {rag.text}")
        else:
            st.subheader("Answer (generated by a language model)")
            st.warning("Machine-generated with retrieved passages as context. Not legal advice, and not verified. "
                       "Check every claim against the quoted source.")
            for stmt in rag.shown:
                chunk = rag.sources[stmt.passage - 1]
                st.markdown(f"{stmt.claim}")
                st.markdown(f"> \"{stmt.quote}\"")
                st.markdown(f":green[Quotation found word for word in {chunk.title}, {chunk.label}. "
                            f"This does not prove the claim above follows from it.]")
                if stmt.warning:
                    st.markdown(f":orange[Check context: {stmt.warning}]")
            if rag.removed:
                with st.expander(f"{len(rag.removed)} statement(s) removed because they could not be checked"):
                    for stmt in rag.removed:
                        st.markdown(f":red[Removed: {REASONS[stmt.status]}.] {stmt.claim}")
            st.markdown("**Retrieved passages**")
            for n, chunk in enumerate(rag.sources, start=1):
                with st.expander(f"[{n}] {chunk.title}, {chunk.label}"):
                    st.write(chunk.text)
    else:
        result = answer(index, question, min_coverage=min_cov)
        if result.abstained:
            st.warning(f"No answer: {result.reason}")
        else:
            st.subheader("Answer (extractive quotes)")
            for sentence, chunk in result.sentences:
                st.markdown(f"- {sentence}  \n  *{chunk.title}, {chunk.label}*")
    with st.expander("Retrieved passages"):
        for chunk, score in index.search(question, k=5):
            st.markdown(f"**{chunk.title}**, {chunk.label} (BM25 {score:.2f})")
            st.write(chunk.text)
