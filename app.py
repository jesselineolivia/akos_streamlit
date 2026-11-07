from pathlib import Path
import sqlite3
from datetime import datetime
import os
import json
import io
import gdown
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

# CNN Model Configuration
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "resnet50_model.keras"

# Google Drive configuration
MODEL_FILE_ID = "14h7yv4LUJuxUF5bwnZ2PIrvoPGQEcnKc"
@st.cache_resource
def ensure_model_downloaded():
    if not MODEL_PATH.exists():
        with st.spinner("Downloading CNN model from Google Drive..."):
            result = gdown.download(
                id=MODEL_FILE_ID,
                output=str(MODEL_PATH),
                quiet=False
            )
        if result is None or not MODEL_PATH.exists():
            raise RuntimeError(
                "Failed to download CNN model from Google Drive."
            )
    if MODEL_PATH.stat().st_size == 0:
        raise RuntimeError("Downloaded CNN model is empty.")
    return str(MODEL_PATH)

# Make the model path accessible to vision_model.py
os.environ["MODEL_PATH"] = ensure_model_downloaded()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
DB_PATH = BASE_DIR / "akos.db"
MODELS_DIR = Path(os.getenv("MODELS_DIR", "D:/KULIAH/Semester 7/M3C/akos_streamlit/models"))

st.set_page_config(
    page_title="A-KOS | Manufacturing Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Styling --------------------------------------
st.markdown("""
<style>
:root { --navy:#13243a; --blue:#2364aa; --teal:#1b998b; --muted:#64748b; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
[data-testid="stSidebar"] { background: #f3f6fa; }
.hero {
  background: linear-gradient(115deg,#13243a 0%,#214e78 65%,#1b998b 100%);
  padding: 1.35rem 1.6rem; border-radius: 14px; color: white; margin-bottom: 1rem;
}
.hero h1 { color: white; margin: 0; font-size: 1.8rem; }
.hero p { color: #e4edf7; margin: .35rem 0 0 0; }
.pillar-card {
  border: 1px solid #dce5ef; border-radius: 12px; padding: 1rem 1.1rem;
  background: white; min-height: 145px;
}
.pillar-kicker { color:#1b998b; font-size:.75rem; font-weight:700; letter-spacing:.08em; }
.pillar-title { color:#13243a; font-weight:750; font-size:1.05rem; margin:.3rem 0; }
.pillar-desc { color:#526174; font-size:.9rem; }
.small-muted { color:#64748b; font-size:.82rem; }
.status-ok { color:#13795b; font-weight:700; }
.status-warn { color:#a16207; font-weight:700; }
.status-bad { color:#b42318; font-weight:700; }
div[data-testid="stMetric"] {
  background:white; border:1px solid #e2e8f0; padding: .75rem 1rem;
  border-radius: 11px;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------- Database -------------------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            category TEXT,
            variant TEXT,
            station TEXT,
            symptom TEXT,
            solution TEXT,
            author TEXT,
            status TEXT DEFAULT 'Pending validation',
            created_at TEXT
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            variant TEXT,
            line_name TEXT,
            station TEXT,
            defect_type TEXT,
            confidence REAL,
            severity TEXT,
            status TEXT,
            route TEXT,
            note TEXT
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id INTEGER,
            action TEXT,
            actor TEXT,
            timestamp TEXT,
            note TEXT
        )""")
        count = conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
        if count == 0:
            seed_knowledge = [
                ("Abnormal vibration on fastening tool", "Maintenance", "HEV", "Assembly 2",
                 "Increased vibration and unstable torque output observed on the fastening tool.",
                 "Inspect socket condition, perform torque calibration check, and escalate to maintenance if deviation persists.",
                 "Senior Foreman - Assembly", "Validated"),
                ("Door panel scratch prevention", "Quality", "ICE", "Trim Line",
                 "Scratches detected on door outer panel surface post-handling transfer.",
                 "Check jig rubber lining and fixture contact pads. Ensure contact surfaces are clear of chips before processing.",
                 "Quality Engineer", "Validated"),
                ("Presence sensor intermittent false reading", "Equipment", "BEV", "Assembly 1",
                 "Presence sensor intermittently fails to detect component seating on fixture.",
                 "Verify optical alignment, connector harness seating, and clean lens surface. Log error frequency in shift log.",
                 "Equipment Engineer", "Pending validation"),
                ("Visual inspection standard for connectors", "SOP", "PHEV", "E-Assembly",
                 "Pre-fastening visual inspection standard for high-voltage and signal connectors.",
                 "Ensure connector housing is crack-free, terminal pins are unbent, and secondary lock engages fully. Hold unit if questionable.",
                 "Senior Foreman - E-Assembly", "Validated"),
            ]
            conn.executemany("""
                INSERT INTO knowledge(title,category,variant,station,symptom,solution,author,status,created_at)
                VALUES(?,?,?,?,?,?,?,?,?)
            """, [(*x, datetime.now().strftime("%Y-%m-%d %H:%M")) for x in seed_knowledge])

        count_i = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
        if count_i == 0:
            seed_inspections = [
                ("2026-11-04 08:12", "HEV", "Assembly Line 2", "Assembly 2", "Scratch", 0.96, "Medium", "Pending review", "Hold → Rework Station", "Right front door outer panel area"),
                ("2026-11-04 08:35", "BEV", "E-Assembly", "E-Assembly", "Connector misalignment", 0.93, "High", "Approved for rework", "E-Assembly → Rework Cell 1", "Requires terminal pin inspection and alignment"),
                ("2026-11-04 09:02", "ICE", "Trim Line", "Trim Line", "Surface contamination", 0.88, "Low", "Resolved", "Cleaning → Re-inspection", "Cleaned with solvent wipe; passed secondary inspection"),
            ]
            conn.executemany("""
                INSERT INTO inspections(timestamp,variant,line_name,station,defect_type,confidence,severity,status,route,note)
                VALUES(?,?,?,?,?,?,?,?,?,?)
            """, seed_inspections)


def load_csv(name):
    p = DATA_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def query_df(sql, params=()):
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def execute(sql, params=()):
    with get_conn() as conn:
        conn.execute(sql, params)
        conn.commit()


init_db()

# ----------------------------- Shared data ----------------------------------
production = load_csv("production_data.csv")
equipment = load_csv("equipment_master.csv")

# ----------------------------- Sidebar --------------------------------------
with st.sidebar:
    st.markdown("## 🏭 A-KOS")
    st.caption("Augmented Kaizen Operating System")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Pillar 1 · Data Foundation", "Pillar 2 · Knowledge Base", "Pillar 3 · Vision Inspection", "Data & Guide"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Environment Status**")
    model_stat = get_model_status()
    if model_stat["available"]:
        st.markdown("🟢 **CNN Model:** Active")
    else:
        st.markdown("🟡 **CNN Model:** Standalone / Demo")
    st.caption("SQLite Backend • Multi-Powertrain")
    st.caption("Prototype version 0.2")


def hero(title, subtitle, eyebrow="AUGMENTED KAIZEN OPERATING SYSTEM"):
    st.markdown(
        f'<div class="hero"><div style="font-size:.72rem;letter-spacing:.12em;font-weight:700;color:#9de5d9">{eyebrow}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )


def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta)


def pill_status(value):
    if value in ("Online", "Validated", "Resolved", "Approved for rework"):
        return f"🟢 {value}"
    if value in ("Warning", "Pending validation", "Pending review"):
        return f"🟡 {value}"
    return f"🔴 {value}"


# ----------------------------- Overview -------------------------------------
if page == "Overview":
    hero("Manufacturing Intelligence Hub", "A unified ecosystem interconnecting production telemetry, frontline know-how, and vision-based early defect detection.")
    open_cases = query_df("SELECT COUNT(*) AS n FROM inspections WHERE status != 'Resolved'").iloc[0, 0]
    total_articles = query_df("SELECT COUNT(*) AS n FROM knowledge").iloc[0, 0]
    
    metric_row([
        ("Production records", f"{len(production):,}" if not production.empty else "0", "Standardized log"),
        ("Registered equipment", f"{len(equipment)}" if not equipment.empty else "0", "Cross-line & variants"),
        ("Knowledge articles", f"{total_articles}", "In operational database"),
        ("Open defect cases", str(open_cases), "Awaiting resolution"),
    ])
    st.markdown("### Three Interconnected Pillars")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="pillar-card"><div class="pillar-kicker">PILLAR 01</div><div class="pillar-title">Unified Data Foundation</div><div class="pillar-desc">Consolidates telemetry across machinery, workstations, and multi-powertrain vehicle variants (ICE/HEV/PHEV/BEV).</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="pillar-card"><div class="pillar-kicker">PILLAR 02</div><div class="pillar-title">Gen-AI Knowledge Base</div><div class="pillar-desc">Captures tacit frontline know-how, structures troubleshooting countermeasures, and validates with senior foremen.</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="pillar-card"><div class="pillar-kicker">PILLAR 03</div><div class="pillar-title">Vision-Based Defect Detection</div><div class="pillar-desc">Detects early visual defects, traces source workstations, and automates closed-loop rework routing with human-in-the-loop.</div></div>', unsafe_allow_html=True)
    
    st.markdown("### Closed-Loop Kaizen Improvement")
    st.info("Closed-loop Kaizen workflow: Process signals → Defect detection → Action recommendation → Human verification → Lessons learned logged back into the Knowledge Base.")
    
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Production output by vehicle variant")
        if not production.empty and "variant" in production.columns:
            val_col = "units" if "units" in production.columns else "output"
            by_variant = production.groupby("variant", as_index=False)[val_col].sum()
            fig = px.bar(by_variant, x="variant", y=val_col, color="variant", title=None,
                         labels={"variant": "Vehicle Variant", val_col: "Produced Units"})
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No production output data available.")
    with right:
        st.markdown("#### Latest inspection cases")
        latest = query_df("SELECT timestamp, variant, station, defect_type, status FROM inspections ORDER BY id DESC LIMIT 5")
        st.dataframe(latest, use_container_width=True, hide_index=True)
    st.caption("All operational figures, inspection cases, and recommendations in this prototype serve for demonstration purposes.")


# ----------------------------- Pillar 1 -------------------------------------
elif page == "Pillar 1 · Data Foundation":
    hero("Pillar 1 · Unified Data Foundation", "Standardized industrial data layer bridging machinery telemetry and workstation signals across multi-powertrain variants.")
    st.markdown("### Data Hub Overview")
    
    online_count = int((equipment["connectivity"] == "Online").sum()) if not equipment.empty else 0
    warn_count = int((equipment["connectivity"] == "Warning").sum()) if not equipment.empty else 0
    var_count = production["variant"].nunique() if not production.empty and "variant" in production.columns else 0
    
    metric_row([
        ("Total equipment", len(equipment), "Master equipment list"),
        ("Online", online_count, "Actively communicating"),
        ("Warning", warn_count, "Requires attention"),
        ("Vehicle variants", var_count, "ICE · HEV · PHEV · BEV"),
    ])
    
    tab1, tab2, tab3 = st.tabs(["Equipment connectivity", "Production data", "Data integration"])
    with tab1:
        if not equipment.empty:
            f1, f2, f3 = st.columns(3)
            variant_filter = f1.multiselect("Vehicle variant", sorted(equipment["variant"].unique()), default=sorted(equipment["variant"].unique()))
            line_filter = f2.multiselect("Production line", sorted(equipment["line"].unique()), default=sorted(equipment["line"].unique()))
            status_filter = f3.multiselect("Connectivity status", sorted(equipment["connectivity"].unique()), default=sorted(equipment["connectivity"].unique()))
            view = equipment[equipment["variant"].isin(variant_filter) & equipment["line"].isin(line_filter) & equipment["connectivity"].isin(status_filter)]
            display_view = view.drop(columns=["protocol"], errors="ignore")
            st.dataframe(display_view, use_container_width=True, hide_index=True)
            st.download_button("⬇ Download equipment master (CSV)", display_view.to_csv(index=False).encode("utf-8"), "equipment_master_filtered.csv", "text/csv")
            
            st.markdown("#### Connectivity status by production line")
            if not view.empty:
                counts = view.groupby(["line", "connectivity"], as_index=False).size()
                fig = px.bar(counts, x="line", y="size", color="connectivity", barmode="stack",
                             labels={"line": "Production Line", "size": "Equipment Count"},
                             color_discrete_map={"Online": "#1b998b", "Warning": "#e09f3e", "Offline": "#b42318"})
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No equipment matches the selected filters.")
        else:
            st.info("Equipment master data is not loaded.")
            
    with tab2:
        if not production.empty:
            p1, p2, p3 = st.columns(3)
            variants = ["All"] + sorted(production["variant"].unique().tolist())
            selected_variant = p1.selectbox("Variant", variants)
            selected_line = p2.selectbox("Production line", ["All"] + sorted(production["line"].unique().tolist()))
            shift_col = "shift" if "shift" in production.columns else None
            shifts = ["All"] + sorted(production[shift_col].unique().tolist()) if shift_col else ["All"]
            selected_shift = p3.selectbox("Shift", shifts)
            
            pv = production.copy()
            if selected_variant != "All":
                pv = pv[pv["variant"] == selected_variant]
            if selected_line != "All":
                pv = pv[pv["line"] == selected_line]
            if selected_shift != "All" and shift_col:
                pv = pv[pv[shift_col] == selected_shift]
                
            st.dataframe(pv, use_container_width=True, hide_index=True)
            st.download_button("⬇ Download production data (CSV)", pv.to_csv(index=False).encode("utf-8"), "production_data_filtered.csv", "text/csv")
            st.caption("Standardized telemetry schema: timestamp, line, station, variant, shift, units, cycle_time_sec, downtime_min, scrap_units.")
        else:
            st.warning("Production data is empty.")
            
    with tab3:
        st.markdown("#### Integration Health")
        integration = pd.DataFrame([
            ["Assembly Line 1", "PLC Gateway", "Online", "2026-11-04 09:10"],
            ["Assembly Line 2", "Edge Gateway", "Online", "2026-11-04 09:09"],
            ["E-Assembly", "Vision Edge Node", "Warning", "2026-11-04 08:52"],
            ["Trim Line", "Legacy Adapter", "Online", "2026-11-04 09:07"],
            ["Final Inspection", "Quality Inspection Node", "Online", "2026-11-04 09:04"],
        ], columns=["Line", "Integration Node", "Status", "Last Sync"])
        st.dataframe(integration, use_container_width=True, hide_index=True)
        st.warning("Industrial note: Actual operational deployment requires PLC tag mapping, OT/IT boundary firewall configurations, and shopfloor authentication controls.")
        
        st.markdown("#### Upload External Production / Equipment Dataset")
        uploaded = st.file_uploader("Upload CSV dataset for ingestion preview", type=["csv"], key="p1_upload")
        if uploaded:
            try:
                incoming = pd.read_csv(uploaded)
                st.success(f"File parsed successfully: {len(incoming)} rows, {len(incoming.columns)} columns.")
                st.dataframe(incoming.head(20), use_container_width=True)
                st.caption("Upload preview mode: Data is displayed in buffer and has not modified core master tables.")
            except Exception as exc:
                st.error(f"Failed to parse CSV file: {exc}")


# ----------------------------- Pillar 2 -------------------------------------
elif page == "Pillar 2 · Knowledge Base":
    hero("Pillar 2 · Gen-AI Knowledge Base", "Capturing tacit frontline know-how and turning troubleshooting experience into structured, searchable organizational intelligence.")
    knowledge = query_df("SELECT * FROM knowledge ORDER BY id DESC")
    validated = int((knowledge["status"] == "Validated").sum()) if len(knowledge) else 0
    pending = int((knowledge["status"] == "Pending validation").sum()) if len(knowledge) else 0
    
    metric_row([
        ("Knowledge articles", len(knowledge), "Total registered"),
        ("Validated", validated, "Approved by reviewer"),
        ("Pending validation", pending, "Awaiting review"),
        ("Coverage categories", knowledge["category"].nunique() if len(knowledge) else 0, "Distinct domains"),
    ])
    
    t_search, t_capture, t_review = st.tabs(["Knowledge assistant", "Capture know-how", "Validation workflow"])
    with t_search:
        st.markdown("#### Search Operational Knowledge")
        search_term = st.text_input("Enter symptom, problem keyword, station, or vehicle variant", placeholder="e.g. abnormal vibration, torque deviation, connector, scratch")
        category = st.selectbox("Filter category", ["All"] + sorted(knowledge["category"].dropna().unique().tolist()) if not knowledge.empty else ["All"])
        
        results = knowledge.copy()
        if category != "All":
            results = results[results["category"] == category]
        if search_term.strip():
            term = search_term.lower()
            cols = ["title", "symptom", "solution", "station", "variant", "category"]
            mask = results[cols].fillna("").astype(str).apply(lambda col: col.str.lower().str.contains(term, regex=False)).any(axis=1)
            results = results[mask]
            
        if len(results):
            for _, row in results.iterrows():
                with st.expander(f"{row['title']}  ·  {row['station']}  ·  {pill_status(row['status'])}", expanded=bool(search_term)):
                    st.write("**Reported Symptom / Condition**")
                    st.write(row["symptom"])
                    st.write("**Recommended Countermeasure**")
                    st.write(row["solution"])
                    st.caption(f"Category: {row['category']} | Variant: {row['variant']} | Contributor: {row['author']} | Recorded: {row['created_at']}")
        else:
            st.info("No matching knowledge articles found. Capture new troubleshooting procedures in the 'Capture know-how' tab.")
            
        st.markdown("#### AI-Assisted Troubleshooting Assistant")
        question = st.text_area("Describe the operational problem or anomaly encountered on the line", placeholder="Detail symptoms, affected station, variant, and operating conditions...", height=90)
        if st.button("Generate simulated recommendation", type="primary", key="ask_knowledge"):
            if not question.strip():
                st.warning("Please provide a problem description first.")
            else:
                candidates = knowledge[knowledge["status"] == "Validated"]
                st.success("Simulated AI troubleshooting recommendation generated. Use as initial guidance subject to standard SOPs.")
                if len(candidates):
                    st.markdown("**Relevant validated articles retrieved as reference:**")
                    for _, r in candidates.head(2).iterrows():
                        st.markdown(f"- **{r['title']}** ({r['station']}): {r['solution']}")
                st.caption("AI Note: Production systems can couple this engine with LLM embeddings (RAG) over validated standard operating sheets.")

    with t_capture:
        st.markdown("#### Capture Frontline Tacit Know-How")
        st.caption("Document problem conditions, verified countermeasures, and process context. New submissions enter the foreman validation queue.")
        with st.form("knowledge_capture_form"):
            title = st.text_input("Knowledge title", placeholder="e.g. Countermeasure for torque instability at fastening cell")
            c1, c2 = st.columns(2)
            cat = c1.selectbox("Category", ["Troubleshooting", "Quality", "Maintenance", "Equipment", "SOP", "Kaizen"])
            var = c2.selectbox("Vehicle variant", ["ICE", "HEV", "PHEV", "BEV", "All variants"])
            c3, c4 = st.columns(2)
            station = c3.text_input("Workstation / area", placeholder="e.g. Assembly 2")
            author = c4.text_input("Contributor name / role", placeholder="e.g. Senior Foreman - Assembly")
            symptom = st.text_area("Symptom or defect condition", height=100)
            solution = st.text_area("Verified inspection steps / corrective countermeasure", height=130)
            submitted = st.form_submit_button("Submit knowledge draft", type="primary")
            if submitted:
                if not title.strip() or not symptom.strip() or not solution.strip():
                    st.error("Title, symptom, and countermeasure fields are required.")
                else:
                    execute("""INSERT INTO knowledge(title,category,variant,station,symptom,solution,author,status,created_at)
                               VALUES(?,?,?,?,?,?,?,?,?)""",
                            (title, cat, var, station, symptom, solution, author or "Contributor", "Pending validation", datetime.now().strftime("%Y-%m-%d %H:%M")))
                    st.success("Knowledge draft saved. Forwarded to senior foreman validation queue.")
                    st.rerun()

    with t_review:
        pending_records = query_df("SELECT * FROM knowledge WHERE status='Pending validation' ORDER BY id ASC")
        if pending_records.empty:
            st.success("No knowledge submissions awaiting validation.")
        else:
            for _, row in pending_records.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['title']}**  ·  `{row['category']}`")
                    st.write(f"**Symptom:** {row['symptom']}")
                    st.write(f"**Proposed Countermeasure:** {row['solution']}")
                    st.caption(f"Contributor: {row['author']} | Station: {row['station']} | Variant: {row['variant']}")
                    a, b = st.columns([1, 1])
                    if a.button("✓ Validate & Approve", key=f"approve_k_{row['id']}", type="primary"):
                        execute("UPDATE knowledge SET status='Validated' WHERE id=?", (int(row["id"]),))
                        st.rerun()
                    if b.button("Return for Revision", key=f"revise_k_{row['id']}"):
                        execute("UPDATE knowledge SET status='Needs revision' WHERE id=?", (int(row["id"]),))
                        st.rerun()


# ----------------------------- Pillar 3 -------------------------------------
elif page == "Pillar 3 · Vision Inspection":
    hero("Pillar 3 · Vision-Based Early Defect Detection", "Early visual defect identification, source workstation traceability, and closed-loop rework routing with human-in-the-loop.")
    cases = query_df("SELECT * FROM inspections ORDER BY id DESC")
    open_cases = int((~cases["status"].eq("Resolved")).sum()) if len(cases) else 0
    approved_rework = int((cases["status"] == "Approved for rework").sum()) if len(cases) else 0
    
    metric_row([
        ("Inspection cases", len(cases), "Recorded in database"),
        ("Open cases", open_cases, "Awaiting resolution"),
        ("Approved rework", approved_rework, "Routing active"),
        ("Avg. confidence", f"{cases['confidence'].mean()*100:.1f}%" if len(cases) else "—", "Model score"),
    ])
    
    t_live, t_queue, t_trace = st.tabs(["Inspection station", "Defect queue & review", "Traceability & learning loop"])
    with t_live:
        st.markdown("#### Inspection Workstation")
        
        # Display model status
        model_status = get_model_status()
        model_path_str = model_status["path"]
        if model_status["available"]:
            st.success(f"🟢 **CNN Model Connected:** `{model_path_str}`")
        else:
            st.warning(f"🟡 **CNN Model Standalone Mode:** Model not found at `{model_path_str}`. Demonstration inference and manual parameters are active.")
            
        up = st.file_uploader("Upload component inspection image (JPEG / PNG)", type=["png", "jpg", "jpeg"], key="vision_upload")
        
        predicted_label = "Scratch"
        predicted_conf = 0.94
        
        if up:
            try:
                img = Image.open(up).convert("RGB")
                img_col, pred_col = st.columns([1.1, 1])
                with img_col:
                    st.image(img, caption="Inspection input preview", use_container_width=True)
                with pred_col:
                    st.markdown("##### Real-Time Classifier Output")
                    if model_status["available"]:
                        try:
                            classifier, class_names = load_classifier()
                            label, conf, probs = predict_image(classifier, class_names, img)
                            predicted_label = label
                            predicted_conf = conf
                            st.metric("Detected Classification", label)
                            st.metric("Model Confidence", f"{conf:.1%}")
                            st.progress(float(conf))
                            if probs:
                                st.write("Class probability distribution:")
                                st.bar_chart(pd.DataFrame({"Probability": probs}, index=list(class_names)))
                            st.success("Prediction computed using connected CNN model.")
                        except Exception as exc:
                            st.error(f"Inference error: {exc}")
                    else:
                        st.info("Uploaded image loaded into inspection buffer. Complete inspection parameters below.")
            except Exception as e:
                st.error(f"Could not load image: {e}")

        st.markdown("##### Inspection Record Logging")
        with st.form("inspection_form"):
            a, b = st.columns(2)
            variant = a.selectbox("Vehicle variant", ["ICE", "HEV", "PHEV", "BEV"])
            line = b.selectbox("Production line", ["Assembly Line 1", "Assembly Line 2", "Trim Line", "E-Assembly"])
            
            c, d = st.columns(2)
            station = c.selectbox("Source workstation", ["Assembly 1", "Assembly 2", "Trim Line", "E-Assembly", "Final Inspection"])
            # Defect options: prioritize loaded model class_names if available
            try:
                _, loaded_names = load_classifier()
                defect_options = list(loaded_names)
            except Exception:
                defect_options = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches", "Scratch", "Surface contamination"]
            default_defect_idx = defect_options.index(predicted_label) if predicted_label in defect_options else 0
            defect = d.selectbox("Defect indication", defect_options, index=default_defect_idx)
            
            e, f = st.columns(2)
            severity = e.selectbox("Defect severity", ["Low", "Medium", "High"])
            confidence = f.slider("Detection confidence", 0.50, 0.999, float(predicted_conf), 0.001)
            note = st.text_input("Inspector observation notes", placeholder="e.g. right front panel surface deviation")
            run = st.form_submit_button("▶ Submit inspection record", type="primary")
            
            if run:
                routes = {
                    "Folding marks": "Hold → Leather Conditioning Station",
                    "Grain off": "Hold → Surface Refinishing",
                    "Growth marks": "Acceptable / Downgrade → Grade B Cell",
                    "Loose grains": "Hold → Re-tanning Inspection",
                    "Non defective": "Pass → Next Station",
                    "Pinhole": "Hold → Sealant & Patch Station",
                    "crazing": "Hold → Surface Treatment Cell",
                    "inclusion": "Quarantine → Metallurgy Lab",
                    "patches": "Hold → Grinding & Buffing",
                    "pitted_surface": "Hold → Surface Rework Cell",
                    "rolled-in_scale": "Cleaning → Pickling Line",
                    "scratches": "Hold → Polishing Station",
                    "Scratch": "Hold → Rework Station",
                    "Surface contamination": "Cleaning → Re-inspection",
                    "Connector misalignment": "E-Assembly → Rework Cell 1",
                    "Missing component": "Hold → Assembly Verification",
                    "Fastening anomaly": "Hold → Torque Verification",
                }
                suggested_route = routes.get(defect, "Hold → Quality Cell")
                execute("""INSERT INTO inspections(timestamp,variant,line_name,station,defect_type,confidence,severity,status,route,note)
                           VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        (datetime.now().strftime("%Y-%m-%d %H:%M"), variant, line, station, defect, float(confidence), severity, "Pending review", suggested_route, note))
                st.success("Inspection logged successfully. Rerouted to Defect Queue for supervisor review.")
                st.rerun()

    with t_queue:
        st.markdown("#### Defect Queue & Human-in-the-Loop Review")
        queue = query_df("SELECT * FROM inspections WHERE status != 'Resolved' ORDER BY id DESC")
        if queue.empty:
            st.success("All inspection cases resolved. No active review items.")
        else:
            for _, row in queue.iterrows():
                with st.container(border=True):
                    head, status_col = st.columns([3, 1])
                    head.markdown(f"**Case #{row['id']} · {row['defect_type']}**")
                    status_col.markdown(pill_status(row["status"]))
                    st.caption(f"{row['timestamp']} | Variant: {row['variant']} | Location: {row['line_name']} / {row['station']} | Confidence: {row['confidence']*100:.1f}% | Severity: {row['severity']}")
                    st.write(f"**Recommended Route:** `{row['route']}`")
                    if row["note"]:
                        st.write(f"**Inspector Note:** {row['note']}")
                    x, y, z = st.columns(3)
                    if x.button("Approve Route", key=f"approve_i_{row['id']}", type="primary", disabled=row["status"] == "Approved for rework"):
                        execute("UPDATE inspections SET status='Approved for rework' WHERE id=?", (int(row["id"]),))
                        execute("INSERT INTO actions(inspection_id,action,actor,timestamp,note) VALUES(?,?,?,?,?)",
                                (int(row["id"]), "Route approved", "Supervisor", datetime.now().strftime("%Y-%m-%d %H:%M"), "Human verification confirmed"))
                        st.rerun()
                    if y.button("Mark Resolved", key=f"resolve_i_{row['id']}", disabled=row["status"] == "Resolved"):
                        execute("UPDATE inspections SET status='Resolved' WHERE id=?", (int(row["id"]),))
                        execute("INSERT INTO actions(inspection_id,action,actor,timestamp,note) VALUES(?,?,?,?,?)",
                                (int(row["id"]), "Case resolved", "Inspector", datetime.now().strftime("%Y-%m-%d %H:%M"), "Confirmed post-rework pass"))
                        st.rerun()
                    if z.button("Reject Indication", key=f"reject_i_{row['id']}", disabled=row["status"] == "Resolved"):
                        execute("UPDATE inspections SET status='False alarm / rejected' WHERE id=?", (int(row["id"]),))
                        execute("INSERT INTO actions(inspection_id,action,actor,timestamp,note) VALUES(?,?,?,?,?)",
                                (int(row["id"]), "Indication rejected", "Reviewer", datetime.now().strftime("%Y-%m-%d %H:%M"), "Evaluated as false positive"))
                        st.rerun()

    with t_trace:
        st.markdown("#### Workstation Traceability & Learning Loop")
        if len(cases):
            selected_id = st.selectbox("Select inspection case", cases["id"].tolist(), format_func=lambda x: f"Case #{x} · " + str(cases.loc[cases["id"] == x, "defect_type"].iloc[0]))
            selected = cases[cases["id"] == selected_id].iloc[0]
            st.markdown("**Origin Trace Path**")
            st.info(f"Vehicle Variant **{selected['variant']}** → Line **{selected['line_name']}** → Source Station **{selected['station']}** → Indication **{selected['defect_type']}** → Route: `{selected['route']}`")
            
            action_log = query_df("SELECT action, actor, timestamp, note FROM actions WHERE inspection_id=? ORDER BY id", (int(selected_id),))
            st.markdown("**Human Decision Audit Log**")
            if action_log.empty:
                st.caption("No human review actions recorded yet.")
            else:
                st.dataframe(action_log, use_container_width=True, hide_index=True)
                
            st.markdown("#### Closed-Loop Kaizen Feedback")
            st.write("When root causes and corrective countermeasures are validated, document the finding directly into the Knowledge Base to prevent recurring shopfloor defects.")
            if st.button("Create knowledge draft from this case", key="create_from_case"):
                execute("""INSERT INTO knowledge(title,category,variant,station,symptom,solution,author,status,created_at)
                           VALUES(?,?,?,?,?,?,?,?,?)""",
                        (f"Kaizen learning from case #{selected_id}: {selected['defect_type']}",
                         "Quality", selected["variant"], selected["station"],
                         f"{selected['defect_type']} anomaly flagged on {selected['line_name']}.",
                         f"Assigned routing: {selected['route']}. Complete primary root-cause and containment verification post-investigation.",
                         "Inspection Workflow", "Pending validation", datetime.now().strftime("%Y-%m-%d %H:%M")))
                st.success("Knowledge draft created. Visible in Pillar 2 validation queue.")
        else:
            st.info("No inspection cases found.")


# ----------------------------- Data & guide ---------------------------------
elif page == "Data & Guide":
    hero("Data & Operational Guide", "Architectural scope, simulation assumptions, and deployment notes.")
    st.markdown("### System Scope & Features")
    st.markdown("""
    - **Pillar 1:** Equipment master connectivity, real-time production signals, multi-powertrain filters (ICE/HEV/PHEV/BEV), and external dataset upload preview.
    - **Pillar 2:** Structured frontline tacit know-how capture, keyword retrieval, simulated AI troubleshooting assistant, and senior foreman approval workflow.
    - **Pillar 3:** CNN vision inference with real Keras model integration (`resnet50_model.keras`), defect queue, human-in-the-loop review, origin traceability, and closed-loop knowledge feedback.
    - **Data Persistence:** Dedicated SQLite database (`akos.db`) persisting knowledge articles, inspection cases, and human action audit logs.
    """)
    
    st.markdown("### Industrial Deployment Prerequisites")
    st.warning("""
    1. **OT/IT Boundaries:** Real line deployment necessitates OPC-UA / MQTT industrial edge gateways, PLC tag polling configurations, and isolated VLAN security.
    2. **Vision Model Calibration:** Ensure the deployed CNN classifier (`resnet50_model.keras`) is calibrated to match factory lighting, working distance, and resolution parameters.
    3. **Routing Authorization:** Rerouting commands generated in Pillar 3 must pass through MES/ANDON system interlocks prior to physical line diversion.
    """)
    
    st.markdown("### Data Schema & Models Directory")
    st.code(f"""Production Data:  data/production_data.csv (timestamp, line, station, variant, shift, units, cycle_time_sec, downtime_min, scrap_units)
Equipment Master: data/equipment_master.csv (equipment_id, equipment_name, line, station, variant, connectivity, last_sync)
Database Backend: akos.db (knowledge, inspections, actions)
CNN Model Path:   {MODELS_DIR} (resnet50_model.keras, class_names.json)
""", language="text")
    
    st.markdown("### Executing Application")
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
    
    if st.button("Reset demo database to initial state"):
        if DB_PATH.exists():
            DB_PATH.unlink()
        init_db()
        st.success("Database reset to factory seeds. Reloading...")
        st.rerun()

st.markdown("---")
st.caption("A-KOS Prototype | Augmented Kaizen Operating System | Industrial Engineering Concept")
