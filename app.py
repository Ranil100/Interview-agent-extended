import io
import itertools
import os
import re

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader

from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_INTERVIEW_QUESTIONS = 3
DIFFICULTY_MIN = 1
DIFFICULTY_MAX = 10
DIFFICULTY_STEP = 2
DIFFICULTY_BASELINE = 5

PORTAL_STUDENT = "Student Career Portal"
PORTAL_RECRUITER = "Corporate Recruiter Suite"

LLM_MODEL = "gemini-2.5-flash"
# text-embedding-004 was deprecated Jan 2026; gemini-embedding-001 is the GA replacement.
EMBEDDING_MODEL = "gemini-embedding-001"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "portal": PORTAL_STUDENT,
        "api_key": os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""),
        "student_chat_history": [],
        "student_resume_text": "",
        "student_jd": "",
        "int_step": 0,
        "int_difficulty": DIFFICULTY_BASELINE,
        "int_history": [],
        "current_question": "",
        "tech_advice_result": "",
        "recruiter_jd": "",
        "batch_resume_texts": {},
        "funnel_results": None,
        "plagiarism_results": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# File & text utilities
# ---------------------------------------------------------------------------
def extract_text_from_resume(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    file_name = uploaded_file.name.lower()
    file_data = uploaded_file.getvalue()

    if file_name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(file_data))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""
    try:
        return file_data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def strip_demographic_identifiers(text: str) -> str:
    header, body = text[:300], text[300:]
    header = re.sub(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b", "[CANDIDATE NAME]", header)
    return header + body


# ---------------------------------------------------------------------------
# Model loading — never assigned to Settings.llm / Settings.embed_model
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Gemini models…")
def get_models(api_key: str):
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key

    llm = GoogleGenAI(
        model=LLM_MODEL,
        api_key=api_key,
        temperature=0.0,
    )
    embed_model = GoogleGenAIEmbedding(
        model_name=EMBEDDING_MODEL,
        api_key=api_key,
    )
    return llm, embed_model


# ---------------------------------------------------------------------------
# LlamaIndex helpers — always pass llm & embed_model explicitly
# ---------------------------------------------------------------------------
def build_single_index(text_content: str, label: str, embed_model) -> VectorStoreIndex:
    doc = Document(text=text_content, doc_id_=label)
    return VectorStoreIndex.from_documents([doc], embed_model=embed_model)


def query_with_index(prompt: str, context_text: str, label: str, llm, embed_model) -> str:
    index = build_single_index(context_text, label, embed_model)
    engine = index.as_query_engine(llm=llm, embed_model=embed_model)
    return str(engine.query(prompt).response)


def query_llm_direct(prompt: str, llm) -> str:
    return str(llm.complete(prompt).text)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_rubric_response(report: str) -> tuple[str, str]:
    score_match = re.search(r"TOTAL OBJECTIVE SCORE:\s*(\d+)", report, re.IGNORECASE)
    status_match = re.search(
        r"RECRUITMENT STATUS:\s*(SELECT FOR PROCESS|HOLD)",
        report,
        re.IGNORECASE,
    )
    score = score_match.group(1) if score_match else "N/A"
    status = status_match.group(1).upper() if status_match else "HOLD"
    if "SELECT" in status:
        status = "SELECT FOR PROCESS"
    return score, status


def parse_plagiarism_response(report: str) -> tuple[str, str]:
    sim_match = re.search(
        r"SIMILARITY MATCH INDEX:\s*(\d+)\s*%",
        report,
        re.IGNORECASE,
    )
    fraud_match = re.search(
        r"FRAUD CONCLUSION:\s*(CRITICAL OVERLAP FLAG|AMBIGUOUS SIMILARITY|CLEAR)",
        report,
        re.IGNORECASE,
    )
    similarity = f"{sim_match.group(1)}%" if sim_match else "N/A"
    conclusion = fraud_match.group(1).upper() if fraud_match else "UNDETERMINED"
    return similarity, conclusion


def evaluate_answer_quality(eval_output: str) -> str:
    upper = eval_output.upper()
    if re.search(r"SCORE:\s*CORRECT", upper):
        return "CORRECT"
    if re.search(r"SCORE:\s*WRONG", upper):
        return "WRONG"
    return "PARTIAL"


# ---------------------------------------------------------------------------
# Corporate evaluation
# ---------------------------------------------------------------------------
def evaluate_blind_rubric(resume_text: str, jd_text: str, llm, embed_model) -> str:
    clean_resume = strip_demographic_identifiers(resume_text)
    prompt = (
        "You are an objective, blind evaluation AI system. Evaluate the candidate on an "
        "anonymized, bias-free basis.\n"
        "Ignore all personal demographics, names, locations, graduation years, and universities.\n"
        "Score the candidate out of 100 based strictly on this structured rubric:\n"
        "1. Technical Alignment — keyword & stack fit (40 points)\n"
        "2. Project Complexity — depth, scale, and relevance (40 points)\n"
        "3. Logic Framework — problem-solving and systems thinking (20 points)\n\n"
        "Output your final decision using the exact structure below:\n"
        "TOTAL OBJECTIVE SCORE: [Score]/100\n"
        "RUBRIC BREAKDOWN:\n"
        "- Technical Alignment: [X]/40\n"
        "- Project Complexity: [Y]/40\n"
        "- Logic Framework: [Z]/20\n"
        "RECRUITMENT STATUS: [SELECT FOR PROCESS / HOLD]\n\n"
        f"Job Description:\n{jd_text}\n\nAnonymized Resume:\n{clean_resume}"
    )
    return query_with_index(prompt, clean_resume, "eval_doc", llm, embed_model)


# ---------------------------------------------------------------------------
# Student portal
# ---------------------------------------------------------------------------
def render_student_qa_tab(llm, embed_model) -> None:
    st.subheader("Interactive Academic & Career Advisor")
    st.caption(
        "Ask about career paths, curriculum, certifications, or technologies. "
        "Upload a resume in the sidebar to personalize answers."
    )

    for role, text in st.session_state.student_chat_history:
        with st.chat_message(role):
            st.write(text)

    user_query = st.chat_input("Ask about paths, industries, certifications, or technologies…")
    if not user_query:
        return

    st.session_state.student_chat_history.append(("user", user_query))
    with st.chat_message("user"):
        st.write(user_query)

    resume_text = st.session_state.student_resume_text
    if resume_text.strip():
        advisor_prompt = (
            "You are an empathetic university academic advisor and career mentor.\n"
            f"Student question: {user_query}\n\n"
            "Use the student's resume context below when relevant. "
            "Answer concisely and actionably."
        )
        reply = query_with_index(
            advisor_prompt, resume_text, "student_resume_ctx", llm, embed_model
        )
    else:
        advisor_prompt = (
            "You are an empathetic university academic advisor and career mentor.\n"
            f"Answer this student question concisely and actionably:\n{user_query}"
        )
        reply = query_llm_direct(advisor_prompt, llm)

    st.session_state.student_chat_history.append(("assistant", reply))
    with st.chat_message("assistant"):
        st.write(reply)


def render_resume_advisory_tab(llm, embed_model) -> None:
    st.subheader("Technical Resume Optimization Engine")
    st.caption(
        "Scan your resume against a target job description and receive a specific "
        "technical roadmap covering stack gaps and project architecture improvements."
    )

    if st.button("Generate Technical Optimization Analysis", key="gen_tech_adv"):
        resume_text = st.session_state.student_resume_text
        target_jd = st.session_state.student_jd

        if not resume_text.strip():
            st.error("Upload your resume in the sidebar before running the analysis.")
        elif not target_jd.strip():
            st.error("Provide a target job description in the sidebar.")
        else:
            with st.spinner("Analyzing technical stack alignment…"):
                prompt = (
                    "You are a Senior Principal Engineer and technical recruiter.\n"
                    "Analyze the student's resume against the target job description.\n"
                    "Deliver a highly specific technical roadmap with these sections:\n"
                    "## 1. Technical Stack Gaps\n"
                    "List missing languages, frameworks, tools, and infra the role expects.\n"
                    "## 2. Project Architecture Improvements\n"
                    "For each project, explain how to rewrite it to demonstrate scale, "
                    "systems design, and measurable impact.\n"
                    "## 3. Priority Learning Path\n"
                    "Rank certifications, libraries, and architectural patterns to master next.\n\n"
                    f"Target Job Description:\n{target_jd}\n\n"
                    f"Student Resume:\n{resume_text}"
                )
                st.session_state.tech_advice_result = query_with_index(
                    prompt, resume_text, "resume_adv", llm, embed_model
                )

    if st.session_state.tech_advice_result:
        st.markdown(st.session_state.tech_advice_result)


def render_adaptive_interview_tab(llm, embed_model) -> None:
    st.subheader("Adaptive Technical Mock Interview")
    st.caption(
        "A mentor-style simulation that adjusts difficulty based on your answers. "
        "Three questions, then a performance summary."
    )

    step = st.session_state.int_step

    if step == 0:
        if st.button("Initialize Mock Interview Session", key="init_interview"):
            if not st.session_state.student_jd.strip():
                st.error("Provide a target job description in the sidebar.")
            else:
                st.session_state.int_step = 1
                st.session_state.int_difficulty = DIFFICULTY_BASELINE
                st.session_state.int_history = []
                init_prompt = (
                    f"Generate one technical interview question at difficulty "
                    f"{DIFFICULTY_BASELINE}/10 aligned to this role:\n"
                    f"{st.session_state.student_jd}\n\n"
                    "Return only the question — no preamble."
                )
                st.session_state.current_question = query_llm_direct(init_prompt, llm)
                st.rerun()
        return

    if step == -1:
        st.success("Mock interview complete.")
        st.subheader("Performance Summary")
        for idx, log in enumerate(st.session_state.int_history, start=1):
            with st.expander(f"Question {idx} — Difficulty {log['difficulty']}/10"):
                st.markdown(f"**Question:** {log['question']}")
                st.markdown(f"**Your answer:** {log['answer']}")
                st.markdown(f"**Evaluation:**\n{log['eval']}")
        st.markdown(
            f"**Final difficulty tier reached:** {st.session_state.int_difficulty}/10"
        )
        if st.button("Reset Interview Session", key="reset_interview"):
            st.session_state.int_step = 0
            st.session_state.int_difficulty = DIFFICULTY_BASELINE
            st.session_state.int_history = []
            st.session_state.current_question = ""
            st.rerun()
        return

    st.info(f"**Difficulty tier:** {st.session_state.int_difficulty}/{DIFFICULTY_MAX}")
    st.markdown(f"### Question {step} of {MAX_INTERVIEW_QUESTIONS}")
    st.markdown(st.session_state.current_question)

    student_answer = st.text_area(
        "Your technical explanation / code answer",
        height=150,
        key=f"interview_ans_{step}",
    )

    if st.button("Submit Answer & Progress", key=f"submit_ans_{step}"):
        if not student_answer.strip():
            st.error("Provide an answer before submitting.")
            return

        with st.spinner("Evaluating your response…"):
            eval_prompt = (
                "You are a demanding technical interviewer.\n"
                f"Question: {st.session_state.current_question}\n"
                f"Candidate answer: {student_answer}\n\n"
                "Determine if the answer is CORRECT, PARTIAL, or WRONG.\n"
                "Respond strictly in this format:\n"
                "SCORE: [CORRECT / PARTIAL / WRONG]\n"
                "FEEDBACK: [Brief mentor-style feedback; include a hint if WRONG]"
            )
            eval_output = query_llm_direct(eval_prompt, llm)
            verdict = evaluate_answer_quality(eval_output)

            if verdict == "CORRECT":
                st.session_state.int_difficulty = min(
                    DIFFICULTY_MAX,
                    st.session_state.int_difficulty + DIFFICULTY_STEP,
                )
                direction = (
                    "The candidate excelled. Ask a significantly harder "
                    "system/architecture question."
                )
            elif verdict == "WRONG":
                st.session_state.int_difficulty = max(
                    DIFFICULTY_MIN,
                    st.session_state.int_difficulty - DIFFICULTY_STEP,
                )
                direction = (
                    "The candidate struggled. Provide a subtle hint in the feedback, "
                    "then ask an easier foundational question."
                )
            else:
                direction = (
                    "The candidate was partially correct. Ask a related variant "
                    "at the same conceptual level."
                )

            st.session_state.int_history.append(
                {
                    "question": st.session_state.current_question,
                    "answer": student_answer,
                    "eval": eval_output,
                    "difficulty": st.session_state.int_difficulty,
                    "verdict": verdict,
                }
            )

            if step >= MAX_INTERVIEW_QUESTIONS:
                st.session_state.int_step = -1
            else:
                next_prompt = (
                    f"Generate the next technical interview question at difficulty "
                    f"{st.session_state.int_difficulty}/10.\n"
                    f"Direction: {direction}\n"
                    f"Role context:\n{st.session_state.student_jd}\n\n"
                    "Return only the question — no preamble."
                )
                st.session_state.current_question = query_llm_direct(next_prompt, llm)
                st.session_state.int_step = step + 1

            st.rerun()


def render_student_portal(llm, embed_model) -> None:
    st.header("Student Career Portal")

    with st.sidebar:
        st.subheader("Student Inputs")
        student_resume = st.file_uploader(
            "Upload your resume",
            type=["pdf", "txt"],
            key="student_resume_uploader",
        )
        st.session_state.student_jd = st.text_area(
            "Target job / internship description",
            value=st.session_state.student_jd,
            height=200,
            key="student_jd_input",
        )
        if student_resume is not None:
            st.session_state.student_resume_text = extract_text_from_resume(student_resume)

    tab_qa, tab_resume, tab_interview = st.tabs(
        [
            "Academic & Career Q&A",
            "Technical Resume Advisory",
            "Adaptive Question Progression",
        ]
    )

    with tab_qa:
        render_student_qa_tab(llm, embed_model)
    with tab_resume:
        render_resume_advisory_tab(llm, embed_model)
    with tab_interview:
        render_adaptive_interview_tab(llm, embed_model)


# ---------------------------------------------------------------------------
# Corporate recruiter suite
# ---------------------------------------------------------------------------
def render_batch_screening_tab(llm, embed_model) -> None:
    st.subheader("Batch Screening & Objective ATS Rubrics")
    st.caption(
        "Demographic-blind scoring: 40 pts Technical Alignment, "
        "40 pts Project Complexity, 20 pts Logic Framework."
    )

    if st.button("Process Recruitment Funnel Batch", key="run_funnel"):
        jd = st.session_state.recruiter_jd
        batch = st.session_state.batch_resume_texts

        if not jd.strip():
            st.error("Enter a job description in the sidebar.")
        elif not batch:
            st.error("Upload at least one resume in the sidebar.")
        else:
            rows = []
            progress = st.progress(0, text="Screening candidates…")

            for idx, (candidate_id, resume_text) in enumerate(batch.items()):
                report = evaluate_blind_rubric(resume_text, jd, llm, embed_model)
                score, status = parse_rubric_response(report)
                rows.append(
                    {
                        "Candidate ID": candidate_id,
                        "Score": f"{score}/100",
                        "Funnel Status": status,
                        "_report": report,
                    }
                )
                progress.progress((idx + 1) / len(batch), text="Screening candidates…")

            st.session_state.funnel_results = rows
            progress.empty()
            st.success(f"Screened {len(rows)} candidate(s).")

    results = st.session_state.funnel_results
    if results:
        display_df = pd.DataFrame(
            [
                {
                    "Candidate ID": r["Candidate ID"],
                    "Score": r["Score"],
                    "Funnel Status": r["Funnel Status"],
                }
                for r in results
            ]
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        with st.expander("Detailed rubric breakdowns"):
            for record in results:
                st.markdown(f"**{record['Candidate ID']}**")
                st.text(record["_report"])
                st.divider()


def render_plagiarism_tab(llm, embed_model) -> None:
    st.subheader("Batch Plagiarism Similarity Radar")
    st.caption(
        "Cross-scan resume pairs for matching project phrasing, copy-paste sentences, "
        "or identical boilerplate code blocks."
    )

    if st.button("Execute Plagiarism Cross-Scan", key="run_plagiarism"):
        batch = st.session_state.batch_resume_texts

        if len(batch) < 2:
            st.error("Upload at least two resumes to run pairwise comparison.")
        else:
            pairs = list(itertools.combinations(batch.keys(), 2))
            rows = []
            progress = st.progress(0, text="Comparing resume pairs…")

            for idx, (id_a, id_b) in enumerate(pairs):
                prompt = (
                    "You are an expert fraud and document verification specialist.\n"
                    "Compare these two candidate resumes side-by-side. Inspect for:\n"
                    "- Matching unique project descriptions\n"
                    "- Identical boilerplate phrasing or essay sentences\n"
                    "- Matching code blocks or structural copy-paste patterns\n\n"
                    "Respond exactly in this format:\n"
                    "SIMILARITY MATCH INDEX: [0% to 100%]\n"
                    "FRAUD CONCLUSION: [CRITICAL OVERLAP FLAG / AMBIGUOUS SIMILARITY / CLEAR]\n"
                    "AUDIT ANALYSIS: [Detail matching signatures found, or state none]\n\n"
                    f"Document A ({id_a}):\n{batch[id_a][:8000]}\n\n"
                    f"Document B ({id_b}):\n{batch[id_b][:8000]}"
                )
                context = batch[id_a] + "\n\n---\n\n" + batch[id_b]
                report = query_with_index(
                    prompt, context, f"plag_{id_a}_{id_b}", llm, embed_model
                )
                similarity, conclusion = parse_plagiarism_response(report)
                rows.append(
                    {
                        "Pair": f"{id_a} ↔ {id_b}",
                        "Similarity %": similarity,
                        "Fraud Conclusion": conclusion,
                        "_report": report,
                    }
                )
                progress.progress((idx + 1) / len(pairs), text="Comparing resume pairs…")

            st.session_state.plagiarism_results = rows
            progress.empty()
            st.success(f"Analyzed {len(pairs)} unique pair(s).")

    results = st.session_state.plagiarism_results
    if results:
        display_df = pd.DataFrame(
            [
                {
                    "Pair": r["Pair"],
                    "Similarity %": r["Similarity %"],
                    "Fraud Conclusion": r["Fraud Conclusion"],
                }
                for r in results
            ]
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        with st.expander("Detailed audit analyses"):
            for record in results:
                st.markdown(
                    f"**{record['Pair']}** — {record['Similarity %']} — "
                    f"{record['Fraud Conclusion']}"
                )
                st.text(record["_report"])
                st.divider()


def render_recruiter_portal(llm, embed_model) -> None:
    st.header("Corporate Recruiter Suite")

    with st.sidebar:
        st.subheader("Recruiter Inputs")
        st.session_state.recruiter_jd = st.text_area(
            "Job description",
            value=st.session_state.recruiter_jd,
            height=200,
            key="recruiter_jd_input",
        )
        batch_files = st.file_uploader(
            "Upload candidate resumes (10–20)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="batch_resume_uploader",
        )
        if batch_files:
            texts = {}
            for f in batch_files:
                content = extract_text_from_resume(f)
                if content.strip():
                    texts[f.name] = content
            st.session_state.batch_resume_texts = texts
            st.caption(f"{len(texts)} resume(s) loaded.")

    tab_screening, tab_plagiarism = st.tabs(
        [
            "Batch Screening & Objective ATS Rubrics",
            "Batch Plagiarism Similarity Radar",
        ]
    )

    with tab_screening:
        render_batch_screening_tab(llm, embed_model)
    with tab_plagiarism:
        render_plagiarism_tab(llm, embed_model)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Enterprise Interview & Talent Suite",
        page_icon="💼",
        layout="wide",
    )
    init_session_state()

    st.title("Enterprise Interview & Talent Hub")

    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input(
        "Gemini API key",
        type="password",
        value=st.session_state.api_key,
        key="api_key_input",
    )
    st.session_state.api_key = api_key

    if not api_key:
        st.sidebar.warning("Provide your Gemini API key to unlock features.")
        return

    llm, embed_model = get_models(api_key)

    st.sidebar.markdown("---")
    portal = st.sidebar.radio(
        "Select portal",
        [PORTAL_STUDENT, PORTAL_RECRUITER],
        index=0 if st.session_state.portal == PORTAL_STUDENT else 1,
        key="portal_radio",
    )
    st.session_state.portal = portal

    if portal == PORTAL_STUDENT:
        render_student_portal(llm, embed_model)
    else:
        render_recruiter_portal(llm, embed_model)


if __name__ == "__main__":
    main()
