"""
app.py
======
ResumeLens Pro - NLP-Based Intelligent Resume Screening, Skill Extraction
and Context-Aware Job Matching System.

Three modes, each mapped to a case-study objective:

  1. Resume <-> Job Match   : preprocessing, NER, education/experience,
                               skill extraction, TF-IDF/Word2Vec/semantic
                               similarity, missing-skill identification.
  2. Rank Candidates        : multiple resumes vs one job -> ranked shortlist.
  3. Rank Jobs for Resume   : one resume vs multiple job descriptions
                               (multiple-job matching) -> best-fit roles.
"""

from html import escape
from io import StringIO

import pandas as pd
import streamlit as st

from src.matcher import match_resume_to_job
from src.preprocessing import extract_text
from src.ranking import rank_candidates, rank_jobs_for_resume

st.set_page_config(
    page_title="ResumeLens Pro",
    page_icon="\u25c9",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #f7f9fc; color: #172033; }
[data-testid="stDecoration"] {
  display: none;
}
[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { max-width: 1400px; padding: 2rem 2rem 3rem; }
.brand { display:flex; align-items:center; gap:12px; margin-bottom:2px; }
.brand-mark { width:38px; height:38px; border:1px solid #2563eb; border-radius:10px;
  display:flex; align-items:center; justify-content:center; color:#2563eb;
  font-size:22px; font-weight:800; background:#eff6ff; }
.brand-name { font-size:30px; font-weight:800; color:#172033; letter-spacing:-1px; }
.subtitle { color:#64748b; font-size:14px; margin:0 0 20px 50px; }
.score-card { background:#ffffff; border:1px solid #dbe3ef; border-radius:18px;
  padding:24px; text-align:center; }
.score-circle { width:148px; height:148px; border-radius:50%; margin:0 auto 14px;
  display:flex; align-items:center; justify-content:center; position:relative; }
.score-circle::before { content:""; position:absolute; width:116px; height:116px;
  background:#ffffff; border-radius:50%; }
.score-value { position:relative; font-size:30px; font-weight:800; color:#172033; }
.score-status { font-size:20px; font-weight:800; color:#2563eb; margin-bottom:5px; }
.score-note { color:#64748b; font-size:14px; line-height:1.5; }
.result-card { background:#ffffff; border:1px solid #dbe3ef; border-radius:16px;
  padding:20px 22px; min-height:150px; }
.card-title { font-size:18px; font-weight:750; color:#172033; margin-bottom:8px; }
.card-copy { color:#64748b; font-size:14px; margin-bottom:14px; }
.skill-item, .missing-item, .extra-item { border-radius:999px; padding:8px 12px;
  font-size:13px; font-weight:600; margin:3px; display:inline-block; }
.skill-item { border:1px solid #a7f3d0; background:#ecfdf5; color:#047857; }
.missing-item { border:1px solid #fed7aa; background:#fff7ed; color:#c2410c; }
.extra-item { border:1px solid #bfdbfe; background:#eff6ff; color:#1d4ed8; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brand">
  <div class="brand-mark">R</div>
  <br>
  <div class="brand-name">ResumeLens Pro</div>
</div>
<div class="subtitle">
Simple resume insights for your next opportunity
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### About")
    st.write(
        "Compare a resume with a job description and get clear, practical "
        "ideas for strengthening the application."
    )
    st.markdown("### Modes")
    st.markdown(
        "- **Resume ↔ Job Match** — single detailed report\n"
        "- **Rank Candidates** — many resumes vs one job\n"
        "- **Rank Jobs** — one resume vs many jobs"
    )
    st.caption("Supported files: PDF, DOCX, TXT")


def _read_upload(file):
    return extract_text(file)


def _skill_pills(skills, css_class, empty_msg):
    if not skills:
        st.info(empty_msg)
        return
    html = "".join(
        f'<span class="{css_class}">{escape(str(skill).title())}</span>'
        for skill in skills
    )
    st.markdown(html, unsafe_allow_html=True)


def _match_status(score):
    if score >= 85:
        return "Excellent Match"
    if score >= 70:
        return "Good Match"
    if score >= 50:
        return "Moderate Match"
    return "Needs Improvement"


def _other_relevant_skills(report):
    matched = {skill.lower() for skill in report["matched_skills"]}
    return [
        skill for skill in report["resume_skills"]
        if skill.lower() not in matched
    ]


def _improvement_recommendations(report):
    recommendations = []
    missing = report["missing_skills"]
    if missing:
        skills = ", ".join(skill.title() for skill in missing[:5])
        recommendations.append(
            f"Add {skills} to your resume only if you genuinely have this experience."
        )
    if report["matched_skills"]:
        recommendations.append(
            "Strengthen your matched skills with a project, responsibility, or result "
            "that shows how you used them."
        )
    recommendations.append(
        "Add measurable results to project and experience descriptions, such as time "
        "saved, revenue supported, or performance improved."
    )
    recommendations.append(
        "Tailor your professional summary so it clearly reflects the target role "
        "and the value you bring."
    )
    if _other_relevant_skills(report):
        recommendations.append(
            "Keep your additional relevant skills, but place the most job-relevant "
            "ones where they are easiest to find."
        )
    return recommendations[:5]


def _report_text(report, status, other_skills, recommendations):
    lines = [
        "ResumeLens Resume Analysis",
        "==========================",
        "",
        f"Compatibility score: {report['final_score']}/100",
        f"Status: {status}",
        "",
        "Skills you already have",
        "-----------------------",
        ", ".join(skill.title() for skill in report["matched_skills"]) or "None identified",
        "",
        "Skills not found in your resume",
        "-------------------------------",
        ", ".join(skill.title() for skill in report["missing_skills"]) or "None identified",
        "",
        "Other relevant skills",
        "----------------------",
        ", ".join(skill.title() for skill in other_skills) or "None identified",
        "",
        "How to improve your resume",
        "--------------------------",
    ]
    lines.extend(f"{index}. {item}" for index, item in enumerate(recommendations, 1))
    lines.extend([
        "",
        "This score is an estimate of compatibility with the supplied job description,",
        "not a guarantee of selection.",
    ])
    return "\n".join(lines)


tab1, tab2, tab3 = st.tabs(
    ["🔍 Resume Analyzer", "🏆 Rank Candidates", "🎯 Rank Jobs for a Resume"]
)

# ===========================================================================
# TAB 1 — SINGLE RESUME vs SINGLE JOB
# ===========================================================================
with tab1:
    st.markdown("# Resume Analyzer")
    st.write("Find out how well your resume matches a job description.")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Upload your resume")
        resume_file = st.file_uploader(
            "PDF, DOCX or TXT", type=["pdf", "docx", "txt"], key="single_resume"
        )
    with right:
        st.markdown("#### Paste the job description")
        job_text_input = st.text_area(
            "Include the responsibilities and skills for the role.",
            height=230,
            key="single_job",
        )

    if st.button("Analyze My Resume", type="primary", key="btn_single"):
        if not resume_file or not job_text_input.strip():
            st.error("Please provide both a resume and a job description.")
        else:
            with st.spinner("Reviewing your resume against the job description..."):
                resume_text = _read_upload(resume_file)
                if not resume_text.strip():
                    st.error("We could not find readable text in that file. Please try another resume.")
                else:
                    report = match_resume_to_job(resume_text, job_text_input)
                    status = _match_status(report["final_score"])
                    other_skills = _other_relevant_skills(report)
                    recommendations = _improvement_recommendations(report)

                    st.success("Analysis complete.")
                    score = max(0.0, min(report["final_score"], 100.0))
                    score_col, explanation_col = st.columns([1, 2], gap="large")
                    with score_col:
                        st.markdown(
                            f'<div class="score-card"><div class="score-circle" '
                            f'style="background:conic-gradient(#2563eb {score}%, #e2e8f0 0)">'
                            f'<div class="score-value">{score:.0f}</div></div>'
                            f'<div class="score-status">{status}</div>'
                            f'<div class="score-note">out of 100</div></div>',
                            unsafe_allow_html=True,
                        )
                    with explanation_col:
                        st.markdown("### Resume Compatibility Score")
                        st.write(
                            "This estimate reflects how closely the resume aligns with "
                            "the supplied job description."
                        )
                        st.info(
                            "Use this score as guidance while tailoring your resume. "
                            "It is not a guarantee of selection."
                        )

                    st.markdown("### Skills You Already Have")
                    with st.container(border=True):
                        _skill_pills(
                            report["matched_skills"],
                            "skill-item",
                            "No matching skills were identified.",
                        )

                    skill_col, extra_col = st.columns(2, gap="large")
                    with skill_col:
                        st.markdown("### Skills Missing From Your Resume")
                        with st.container(border=True):
                            st.caption("These job-related skills were not found in your resume.")
                            _skill_pills(
                                report["missing_skills"],
                                "missing-item",
                                "No additional skills were identified as missing.",
                            )
                    with extra_col:
                        st.markdown("### Other Relevant Skills")
                        with st.container(border=True):
                            st.caption("These skills appear in your resume but were not explicitly required.")
                            _skill_pills(
                                other_skills,
                                "extra-item",
                                "No additional skills were identified.",
                            )

                    st.markdown("### How to Improve Your Resume")
                    with st.container(border=True):
                        for index, recommendation in enumerate(recommendations, 1):
                            st.markdown(f"**{index}.** {recommendation}")

                    st.markdown("### Resume Improvement Summary")
                    with st.container(border=True):
                        for recommendation in recommendations[:3]:
                            st.markdown(f"- {recommendation}")

                    report_data = _report_text(
                        report, status, other_skills, recommendations
                    )
                    st.download_button(
                        "Download Report",
                        data=StringIO(report_data).getvalue(),
                        file_name="resume_analysis.txt",
                        mime="text/plain",
                    )

# ===========================================================================
# TAB 2 — RANK CANDIDATES (many resumes -> one job)
# ===========================================================================
with tab2:
    st.markdown("#### Upload multiple resumes")
    resume_files = st.file_uploader(
        "Resumes (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"],
        accept_multiple_files=True, key="multi_resumes",
    )
    job_text_rank = st.text_area("Paste the target job description", height=180, key="rank_job")

    if st.button("Rank candidates", type="primary", key="btn_rank_candidates"):
        if not resume_files or not job_text_rank.strip():
            st.error("Please upload at least one resume and a job description.")
        else:
            with st.spinner("Scoring every candidate against the job description..."):
                resumes = [
                    {"name": f.name, "text": _read_upload(f)} for f in resume_files
                ]
                ranked = rank_candidates(resumes, job_text_rank)

            st.success(f"Ranked {len(ranked)} candidate(s).")
            table = pd.DataFrame([{
                "Rank": r["rank"],
                "Candidate": r["name"],
                "Final score": r["final_score"],
                "Matched skills": len(r["matched_skills"]),
                "Missing skills": len(r["missing_skills"]),
            } for r in ranked])
            st.dataframe(table, use_container_width=True, hide_index=True)

            st.download_button(
                "Download ranking as CSV",
                data=table.to_csv(index=False),
                file_name="candidate_ranking.csv",
                mime="text/csv",
            )

            for r in ranked:
                with st.expander(f"#{r['rank']} — {r['name']} (score {r['final_score']})"):
                    st.write(f"**Matched:** {', '.join(r['matched_skills']) or '—'}")
                    st.write(f"**Missing:** {', '.join(r['missing_skills']) or '—'}")
                    st.write(f"**Estimated experience:** {r['candidate_profile']['experience_years']} years")
                    st.write(f"**Education:** {', '.join(r['candidate_profile']['education']) or '—'}")

# ===========================================================================
# TAB 3 — RANK JOBS FOR ONE RESUME (multiple-job matching)
# ===========================================================================
with tab3:
    st.markdown("#### Upload the candidate's resume")
    resume_file_multi = st.file_uploader("Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], key="mj_resume")

    st.markdown("#### Add job descriptions")
    st.caption("Separate each job with a line containing only `---`.")
    jobs_raw = st.text_area(
        "Job descriptions",
        height=220,
        key="mj_jobs",
        placeholder="Job Title: Backend Developer\n...description...\n---\nJob Title: Data Scientist\n...description...",
    )

    if st.button("Find best-fit jobs", type="primary", key="btn_rank_jobs"):
        if not resume_file_multi or not jobs_raw.strip():
            st.error("Please provide a resume and at least one job description.")
        else:
            blocks = [b.strip() for b in jobs_raw.split("\n---\n") if b.strip()]
            jobs = []
            for i, block in enumerate(blocks, start=1):
                first_line = block.splitlines()[0]
                title = first_line.replace("Job Title:", "").strip() if "Job Title:" in first_line else f"Job {i}"
                jobs.append({"title": title, "text": block})

            with st.spinner("Scoring the resume against every job description..."):
                resume_text = _read_upload(resume_file_multi)
                ranked_jobs = rank_jobs_for_resume(resume_text, jobs)

            st.success(f"Compared against {len(ranked_jobs)} job description(s).")
            table = pd.DataFrame([{
                "Rank": r["rank"],
                "Job title": r["title"],
                "Final score": r["final_score"],
            } for r in ranked_jobs])
            st.dataframe(table, use_container_width=True, hide_index=True)

            for r in ranked_jobs:
                with st.expander(f"#{r['rank']} — {r['title']} (score {r['final_score']})"):
                    st.write(f"**Matched:** {', '.join(r['matched_skills']) or '—'}")
                    st.write(f"**Missing:** {', '.join(r['missing_skills']) or '—'}")
