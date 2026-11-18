# ==============================================================
# AI TOOL USAGE AMONG COLLEGE STUDENTS
# Interactive Streamlit Dashboard + Research Analysis
# Dataset: cleaned_ai_student_data (182 valid responses)
# ==============================================================

import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

# --------------------------- PAGE SETUP ---------------------------
st.set_page_config(
    page_title="AI Usage | Research Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------- THEME ---------------------------
BG = "#080D1A"
PANEL = "#111A2E"
PANEL_2 = "#17233B"
TEXT = "#EAF2FF"
MUTED = "#9BAEC9"
CYAN = "#35D6F2"
BLUE = "#668CFF"
PURPLE = "#A78BFA"
PINK = "#F472B6"
GREEN = "#48D6A0"
ORANGE = "#FFB86B"
RED = "#FF718B"
PALETTE = [CYAN, BLUE, PURPLE, PINK, GREEN, ORANGE, RED, "#7DD3FC", "#C4B5FD", "#F9A8D4"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {{
  --bg:{BG}; --panel:{PANEL}; --text:{TEXT}; --muted:{MUTED};
  --cyan:{CYAN}; --purple:{PURPLE};
}}
html, body, [data-testid="stAppViewContainer"] {{
  background: radial-gradient(ellipse at 15% 0%, #14264a 0%, {BG} 48%, #090D19 100%);
  color:var(--text); font-family:'DM Sans',sans-serif;
}}
[data-testid="stHeader"] {{background:rgba(8,13,26,.82);}}
[data-testid="stSidebar"] {{
  background:linear-gradient(180deg,#101B32 0%,#0B1222 100%);
  border-right:1px solid rgba(53,214,242,.18);
}}
[data-testid="stSidebar"] * {{color:{TEXT};}}
h1,h2,h3 {{font-family:'Space Grotesk',sans-serif!important; letter-spacing:-.5px;}}
h1 {{color:{TEXT};}}
p, label, .stMarkdown {{color:{TEXT};}}
div[data-testid="stMetric"] {{
  background:linear-gradient(145deg,rgba(23,35,59,.96),rgba(17,26,46,.94));
  border:1px solid rgba(102,140,255,.25); border-radius:16px;
  padding:17px 18px; box-shadow:0 8px 28px rgba(0,0,0,.18);
  transition:transform .25s ease,border-color .25s ease,box-shadow .25s ease;
}}
div[data-testid="stMetric"]:hover {{
  transform:translateY(-4px); border-color:rgba(53,214,242,.65);
  box-shadow:0 12px 34px rgba(53,214,242,.10);
}}
div[data-testid="stMetric"] label {{color:{MUTED}!important;}}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {{color:{TEXT}!important;font-family:'Space Grotesk';}}
div[data-testid="stPlotlyChart"] {{
div[data-testid="stPlotlyChart"] .modebar {{display:none !important;}}
  background:rgba(17,26,46,.72); border:1px solid rgba(102,140,255,.18);
  border-radius:16px; padding:9px; transition:all .25s ease;
  animation:fadeUp .55s ease both;
}}
div[data-testid="stPlotlyChart"]:hover {{border-color:rgba(53,214,242,.42);}}
.stTabs [data-baseweb="tab-list"] {{gap:8px;}}
.stTabs [data-baseweb="tab"] {{background:{PANEL};border-radius:9px 9px 0 0;color:{MUTED};}}
.stTabs [aria-selected="true"] {{color:{CYAN}!important;border-bottom:2px solid {CYAN};}}
.stButton>button, .stDownloadButton>button {{
  background:linear-gradient(90deg,#236B91,#6251A8); color:white;
  border:0; border-radius:10px; font-weight:600; transition:all .2s ease;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
  transform:translateY(-2px); box-shadow:0 7px 20px rgba(53,214,242,.2); color:white;
}}
div[data-baseweb="select"]>div, div[data-baseweb="input"]>div {{
  background:{PANEL_2}; border-color:rgba(102,140,255,.35); border-radius:9px;
}}
hr {{border-color:rgba(102,140,255,.20);}}
.hero {{
  background:linear-gradient(115deg,rgba(35,107,145,.28),rgba(98,81,168,.25)),
             linear-gradient(135deg,#111D35,#101629);
  border:1px solid rgba(53,214,242,.28); border-radius:20px;
  padding:25px 28px; margin:5px 0 22px 0; position:relative; overflow:hidden;
}}
.hero:after {{
 content:""; position:absolute; width:230px;height:230px;right:-65px;top:-120px;
 border-radius:50%; border:1px solid rgba(53,214,242,.25);
 box-shadow:0 0 0 25px rgba(53,214,242,.035),0 0 0 55px rgba(167,139,250,.035);
}}
.hero-kicker {{color:{CYAN};font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase;}}
.hero-title {{font-family:'Space Grotesk';font-size:31px;font-weight:700;color:{TEXT};margin:7px 0 4px;}}
.hero-sub {{color:{MUTED};font-size:14px;max-width:760px;}}
.section-title {{font-family:'Space Grotesk';font-size:21px;font-weight:700;margin:17px 0 3px;color:{TEXT};}}
.section-sub {{font-size:12px;color:{MUTED};margin-bottom:12px;}}
.insight {{
 background:linear-gradient(120deg,rgba(53,214,242,.09),rgba(167,139,250,.09));
 border-left:3px solid {CYAN}; border-radius:8px;padding:13px 16px;
 color:{TEXT};margin:8px 0 14px 0;font-size:13px;
}}
.pill {{
 display:inline-block;padding:5px 10px;border-radius:30px;
 background:rgba(53,214,242,.12);border:1px solid rgba(53,214,242,.3);
 color:{CYAN};font-size:11px;font-weight:700;letter-spacing:.4px;
}}
.small-muted {{color:{MUTED};font-size:12px;}}
@keyframes fadeUp {{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:translateY(0)}}}}
@media (prefers-reduced-motion: reduce) {{*{{animation:none!important;transition:none!important;}}}}
</style>
""", unsafe_allow_html=True)

# --------------------------- DATA LOADING ---------------------------
@st.cache_data
def load_data(uploaded_bytes=None):
    candidates = [
        "cleaned_ai_student_data (2).csv",
        "cleaned_ai_student_data.csv",
        "data/cleaned_ai_student_data (2).csv",
        "data/cleaned_ai_student_data.csv",
    ]
    if uploaded_bytes is not None:
        data = pd.read_csv(io.BytesIO(uploaded_bytes))
    else:
        for file in candidates:
            if os.path.exists(file):
                data = pd.read_csv(file)
                break
        else:
            return None

    # Normalize likely column names and types
    data.columns = [str(c).strip().lower() for c in data.columns]
    required = ["age","gender","academic_year","program","tools_used","frequency",
                "hours_per_week","overdependence","use_cases","benefits","challenges",
                "satisfaction","usefulness"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        st.error("The dataset is missing required columns: " + ", ".join(missing))
        st.stop()

    for col in ["age","hours_per_week","overdependence","satisfaction","usefulness"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ["gender","academic_year","program","tools_used","frequency",
                "use_cases","benefits","challenges"]:
        data[col] = data[col].fillna("Not stated").astype(str).str.strip()

    # Keep original weekly hours; create capped series for exploratory box plots only.
    q1, q3 = data["hours_per_week"].quantile([.25,.75])
    iqr = q3-q1
    lower, upper = max(0, q1-1.5*iqr), q3+1.5*iqr
    data["hours_per_week_iqr_capped"] = data["hours_per_week"].clip(lower,upper)

    frequency_map = {"Rarely":1,"Sometimes":2,"Very Often":3,"Everyday":4}
    data["frequency_score"] = data["frequency"].map(frequency_map)
    data["age_group"] = pd.cut(data["age"], bins=[0,20,24,100],
                               labels=["17–20","21–24","25+"], include_lowest=True)
    def programme_level(value):
        val = str(value).strip().upper()
        for prefix in ["B.ED","M.ED","PGDE","MACC"]:
            if val.startswith(prefix): return prefix.replace(".","")
        return "Other"
    data["programme_level"] = data["program"].apply(programme_level)
    return data

with st.sidebar:
    st.markdown("## AI RESEARCH HUB")
    st.caption("PRG405 · Introduction to Data Science")
    nav = st.radio("NAVIGATION", ["Dashboard", "Research Analysis"], index=0)

df = load_data()
if df is None:
    st.warning("Place **cleaned_ai_student_data (2).csv** in the same folder as this app.")
    st.stop()

# --------------------------- FILTERS ---------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("### FILTER RESPONSES")
    years = sorted(df["academic_year"].dropna().unique().tolist())
    genders = sorted(df["gender"].dropna().unique().tolist())
    programs = sorted(df["program"].dropna().unique().tolist())

    # Initialize filter state once so the reset button really restores all values.
    for key, options in [("filter_years", years), ("filter_genders", genders), ("filter_programs", programs)]:
        if key not in st.session_state:
            st.session_state[key] = options

    def reset_filters():
        st.session_state["filter_years"] = years
        st.session_state["filter_genders"] = genders
        st.session_state["filter_programs"] = programs

    selected_years = st.multiselect("Academic year", years, key="filter_years")
    selected_genders = st.multiselect("Gender", genders, key="filter_genders")
    selected_programs = st.multiselect("Programme", programs, key="filter_programs")
    st.caption("Filters update the charts and summary indicators.")
    st.button("↺ Reset all filters", use_container_width=True, on_click=reset_filters)

filtered = df[
    df["academic_year"].isin(selected_years) &
    df["gender"].isin(selected_genders) &
    df["program"].isin(selected_programs)
].copy()

# --------------------------- CHART HELPERS ---------------------------
def chart_layout(fig, height=360, legend=True):
    # Horizontal charts need extra space for long category names and value labels.
    is_horizontal = any(
        getattr(trace, "orientation", None) == "h"
        for trace in fig.data
    )
    margins = (
        dict(l=135, r=95, t=62, b=48)
        if is_horizontal else dict(l=48, r=48, t=62, b=48)
    )
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="DM Sans", size=11),
        margin=margins,
        title_font=dict(size=15, color=TEXT, family="Space Grotesk"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED, size=10),
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1
        ),
        showlegend=legend
    )
    fig.update_xaxes(
        gridcolor="rgba(155,174,201,.10)",
        zerolinecolor="rgba(155,174,201,.12)",
        automargin=True,
        tickfont=dict(size=10)
    )
    fig.update_yaxes(
        gridcolor="rgba(155,174,201,.10)",
        zerolinecolor="rgba(155,174,201,.12)",
        automargin=True,
        tickfont=dict(size=10)
    )
    return fig

def multi_counts(data, col):
    vals = data[col].dropna().astype(str).str.split(";").explode().str.strip()
    vals = vals[~vals.isin(["","Not stated","nan","None"])]
    out = vals.value_counts().rename_axis("Response").reset_index(name="Students")
    if len(data):
        out["Percent of respondents"] = out["Students"] / len(data) * 100
    else:
        out["Percent of respondents"] = 0
    return out

def section(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)

def metric_card(label, value, delta=None):
    st.metric(label, value, delta)

def empty_state():
    st.info("No respondents match the selected filters. Please adjust the filters in the sidebar.")

def add_count_labels(fig):
    fig.update_traces(textposition="outside", cliponaxis=False)

# --------------------------- HEADER ---------------------------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">PRG405 · INTRODUCTION TO DATA SCIENCE</div>
  <div class="hero-title">AI Tool Usage Among College Students</div>
  <div class="hero-sub">An interactive exploration of student AI adoption, academic use cases,
  perceived benefits, challenges and satisfaction.</div>
</div>
""", unsafe_allow_html=True)

if filtered.empty:
    empty_state()
    st.stop()

n = len(filtered)
mean_age = filtered["age"].mean()
mean_hours = filtered["hours_per_week"].mean()
mean_sat = filtered["satisfaction"].mean()
mean_use = filtered["usefulness"].mean()

# =========================== DASHBOARD ===========================
if nav == "Dashboard":
    st.markdown('<span class="pill">LIVE FILTERED VIEW</span>', unsafe_allow_html=True)
    section("Research at a glance", "Key indicators update when you change the filters.")
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: metric_card(" Respondents", f"{n:,}")
    with c2: metric_card(" Average age", f"{mean_age:.1f}" if pd.notna(mean_age) else "—", "years")
    with c3: metric_card(" Avg. weekly AI use", f"{mean_hours:.1f}" if pd.notna(mean_hours) else "—", "hours")
    with c4: metric_card(" Satisfaction", f"{mean_sat:.2f}/5" if pd.notna(mean_sat) else "—")
    with c5: metric_card(" Perceived usefulness", f"{mean_use:.2f}/5" if pd.notna(mean_use) else "—")

    # Dynamic findings make the main patterns clear without requiring users to inspect every chart.
    tool_summary = multi_counts(filtered, "tools_used").sort_values("Students", ascending=False)
    purpose_summary = multi_counts(filtered, "use_cases").sort_values("Students", ascending=False)
    benefit_summary = multi_counts(filtered, "benefits").sort_values("Students", ascending=False)
    challenge_summary = multi_counts(filtered, "challenges").sort_values("Students", ascending=False)
    common_frequency = filtered["frequency"].value_counts()
    high_sat = filtered["satisfaction"].dropna()
    high_sat_pct = (high_sat.ge(4).mean() * 100) if len(high_sat) else np.nan

    section("Key findings", "A quick interpretation of the selected respondents.")
    insight_items = []
    if not tool_summary.empty:
        r = tool_summary.iloc[0]
        insight_items.append(f"**Most used AI tool:** {r['Response']} ({int(r['Students'])} selections; {r['Percent of respondents']:.1f}% of respondents).")
    if not common_frequency.empty:
        label = common_frequency.idxmax()
        insight_items.append(f"**Most common usage frequency:** {label} ({int(common_frequency.max())} students; {common_frequency.max()/n*100:.1f}%).")
    if not purpose_summary.empty:
        r = purpose_summary.iloc[0]
        insight_items.append(f"**Main purpose:** {r['Response']} ({int(r['Students'])} selections; {r['Percent of respondents']:.1f}%).")
    if not benefit_summary.empty:
        r = benefit_summary.iloc[0]
        insight_items.append(f"**Most reported benefit:** {r['Response']} ({int(r['Students'])} selections).")
    if not challenge_summary.empty:
        r = challenge_summary.iloc[0]
        insight_items.append(f"**Most reported challenge:** {r['Response']} ({int(r['Students'])} selections).")
    if pd.notna(high_sat_pct):
        insight_items.append(f"**Positive satisfaction ratings:** {high_sat_pct:.1f}% rated satisfaction 4 or 5 out of 5.")
    st.markdown("\n".join([f"- {item}" for item in insight_items]))

    section("Recommendations", "Practical suggestions connected to the benefits and challenges examined in the survey.")
    st.markdown("""
- **Verify AI-generated information:** compare important facts and explanations with reliable academic or official sources.
- **Protect privacy:** avoid entering personal, confidential, or sensitive information into AI tools.
- **Use AI as a learning support:** use it to clarify concepts, generate ideas, and receive guidance while completing and understanding your own work.
- **Practise responsible academic use:** follow lecturers' instructions and the institution's expectations for acknowledging AI assistance.
""")

    st.markdown("---")
    section("01 · Respondent profile", "Understand who participated in the survey.")
    p1,p2,p3 = st.columns([1,1.25,1.4])
    with p1:
        gender_counts = filtered["gender"].value_counts().reset_index()
        gender_counts.columns=["Gender","Students"]
        fig=px.pie(gender_counts,names="Gender",values="Students",hole=.66,
                   color_discrete_sequence=[CYAN,PURPLE,ORANGE,BLUE],title="Gender distribution")
        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            marker=dict(line=dict(color=BG, width=2))
        )
        fig.add_annotation(text=f"<b>{n}</b><br><sup>students</sup>",showarrow=False,
                           font=dict(size=17,color=TEXT))
        st.plotly_chart(chart_layout(fig,330),use_container_width=True)
    with p2:
        year_order=["Year I","Year II","Year III","Year IV"]
        yc=filtered["academic_year"].value_counts().reindex(year_order).dropna().reset_index()
        yc.columns=["Academic year","Students"]
        fig=px.bar(yc,x="Academic year",y="Students",color="Academic year",
                   color_discrete_sequence=[CYAN,BLUE,PURPLE,PINK],title="Respondents by academic year",
                   text="Students")
        fig.update_traces(marker_line_width=0,textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(chart_layout(fig,330,False),use_container_width=True)
    with p3:
        age=filtered["age"].dropna()
        fig=px.histogram(filtered,x="age",nbins=12,color_discrete_sequence=[CYAN],
                         title="Age distribution",marginal="box")
        fig.update_traces(marker_line=dict(color=BG,width=1),opacity=.88)
        st.plotly_chart(chart_layout(fig,330,False),use_container_width=True)

    st.markdown("---")
    section("02 · AI tool adoption", "Multiple selections are counted separately. Percentages are based on respondents and may add to more than 100%.")
    t1,t2 = st.columns([1.35,1])
    with t1:
        tc=multi_counts(filtered,"tools_used").sort_values("Students",ascending=True)
        if not tc.empty:
            tc["Share"] = tc["Students"].astype(str)+" students · "+tc["Percent of respondents"].round(1).astype(str)+"%"
            fig=px.bar(tc,x="Students",y="Response",orientation="h",color="Response",
                       color_discrete_sequence=PALETTE,title="AI tools used by respondents",
                       text="Share",hover_data={"Percent of respondents":":.1f"})
            fig.update_traces(textposition="outside",cliponaxis=False)
            fig.update_layout(showlegend=False,yaxis_title="",xaxis_title="Number of students")
            st.plotly_chart(chart_layout(fig,max(360,35*len(tc)+100),False),use_container_width=True)
    with t2:
        fc=filtered["frequency"].value_counts().reindex(["Rarely","Sometimes","Very Often","Everyday"]).dropna().reset_index()
        fc.columns=["Usage frequency","Students"]
        fig=px.bar(fc,x="Students",y="Usage frequency",orientation="h",color="Usage frequency",
                   color_discrete_sequence=[BLUE,CYAN,PURPLE,PINK],title="How often students use AI",
                   text="Students",category_orders={"Usage frequency":["Everyday","Very Often","Sometimes","Rarely"]})
        fig.update_layout(showlegend=False,xaxis_title="Number of students",yaxis_title="")
        fig.update_traces(textposition="outside",cliponaxis=False)
        st.plotly_chart(chart_layout(fig,360,False),use_container_width=True)
        if not tc.empty:
            st.markdown(f'<div class="insight">Most selected tool: <b>{tc.iloc[-1]["Response"]}</b> '
                        f'({int(tc.iloc[-1]["Students"])} respondents, {tc.iloc[-1]["Percent of respondents"]:.1f}%).</div>',
                        unsafe_allow_html=True)

    st.markdown("---")
    section("03 · Purposes, benefits and challenges", "Explore why students use AI and what they report experiencing.")
    a,b,c=st.columns(3)
    for col, field, title, color in [
        (a,"use_cases","Main purposes",CYAN),
        (b,"benefits","Reported benefits",GREEN),
        (c,"challenges","Reported challenges",PINK)
    ]:
        with col:
            tbl=multi_counts(filtered,field).sort_values("Students",ascending=True)
            if not tbl.empty:
                tbl["Label"] = tbl["Students"].astype(str) + " students"
                fig=px.bar(tbl,x="Students",y="Response",orientation="h",color="Response",
                           color_discrete_sequence=PALETTE,title=title,text="Label",
                           hover_data={"Percent of respondents":":.1f"})
                fig.update_layout(showlegend=False,yaxis_title="",xaxis_title="Number of students")
                fig.update_traces(marker_line_width=0,textposition="outside",cliponaxis=False)
                st.plotly_chart(chart_layout(fig, max(360, 34 * len(tbl) + 120), False), use_container_width=True)

    st.markdown("---")
    section("04 · Satisfaction and perceived usefulness", "Ratings use the original 1–5 survey scale.")
    r1,r2,r3=st.columns(3)
    rating_labels={"satisfaction":"Satisfaction","usefulness":"Perceived usefulness","overdependence":"Overdependence"}
    for container,(col,title) in zip([r1,r2,r3],rating_labels.items()):
        with container:
            vals=filtered[col].dropna()
            counts=vals.value_counts().reindex([1,2,3,4,5],fill_value=0).rename_axis("Rating").reset_index(name="Students")
            fig=px.bar(counts,x="Rating",y="Students",color="Rating",
                       color_discrete_sequence=[BLUE,CYAN,PURPLE,PINK,GREEN],
                       title=title,text="Students")
            fig.update_layout(showlegend=False,xaxis=dict(dtick=1))
            fig.update_traces(textposition="outside")
            st.plotly_chart(chart_layout(fig,300,False),use_container_width=True)

    st.markdown("---")
    section("05 · Usage patterns and group comparisons", "Compare weekly hours and average ratings across academic years. The line connects group averages; it is not a time-series trend.")
    g1,g2=st.columns(2)
    with g1:
        fig=px.box(filtered,x="academic_year",y="hours_per_week_iqr_capped",
                   color="academic_year",points="all",
                   color_discrete_sequence=[CYAN,BLUE,PURPLE,PINK],
                   category_orders={"academic_year":["Year I","Year II","Year III","Year IV"]},
                   title="Weekly AI hours by academic year (IQR-capped view)")
        fig.update_layout(showlegend=False,xaxis_title="",yaxis_title="Hours per week")
        st.plotly_chart(chart_layout(fig,390,False),use_container_width=True)
    with g2:
        group=filtered.groupby("academic_year",observed=True)[["satisfaction","usefulness","overdependence"]].mean().reset_index()
        group=group.melt(id_vars="academic_year",var_name="Measure",value_name="Mean rating")
        group["Measure"]=group["Measure"].map({"satisfaction":"Satisfaction","usefulness":"Usefulness","overdependence":"Overdependence"})
        fig=px.line(group,x="academic_year",y="Mean rating",color="Measure",markers=True,
                    color_discrete_sequence=[CYAN,PURPLE,PINK],
                    category_orders={"academic_year":["Year I","Year II","Year III","Year IV"]},
                    title="Average ratings by academic year")
        fig.update_yaxes(range=[0,5.3],dtick=1)
        st.plotly_chart(chart_layout(fig,390),use_container_width=True)

    with st.expander("View filtered data table"):
        st.dataframe(filtered.drop(columns=["hours_per_week_iqr_capped","frequency_score"],errors="ignore"),
                     use_container_width=True,hide_index=True)
        st.download_button("⬇ Download filtered data as CSV",
                           filtered.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_ai_tool_usage_data.csv",mime="text/csv")

# ========================= RESEARCH ANALYSIS =========================
else:
    st.markdown('<span class="pill">INTERPRETATION & STATISTICAL ANALYSIS</span>', unsafe_allow_html=True)
    section("Research analysis", "Findings are calculated from the selected respondents in the cleaned survey dataset.")
    st.markdown('<div class="insight"><b>Research scope:</b> This project describes AI tool usage among the survey respondents. '
                'The results should not automatically be generalized to all college students.</div>',unsafe_allow_html=True)

    # RQ5: formal hypothesis test for satisfaction, plus an additional usefulness analysis.
    section("RQ5 · Relationships between AI usage, satisfaction and usefulness",
            "Spearman's rank correlation is used because usage frequency is ordinal and ratings use a 1–5 scale.")
    st.markdown("**Research question:** Is there a relationship between AI tool usage and students’ perceived satisfaction or usefulness?")
    st.markdown("**Formal hypothesis (satisfaction)**")
    st.markdown("**H₀:** There is no statistically significant relationship between the frequency of AI tool usage and students’ perceived satisfaction.")
    st.markdown("**H₁:** There is a statistically significant relationship between the frequency of AI tool usage and students’ perceived satisfaction.")
    st.caption("Usage frequency is coded in its natural order: Rarely = 1, Sometimes = 2, Very Often = 3, Everyday = 4. Significance level: α = 0.05. The usefulness analysis is additional and does not introduce a new formal hypothesis.")

    freq_labels = {1: "Rarely", 2: "Sometimes", 3: "Very Often", 4: "Everyday"}

    # Formal hypothesis test: frequency and satisfaction.
    relationship_data = filtered[["frequency_score", "satisfaction"]].copy()
    relationship_data["satisfaction"] = pd.to_numeric(relationship_data["satisfaction"], errors="coerce")
    relationship_data = relationship_data.dropna()
    st.markdown("### Formal hypothesis test · Usage frequency and satisfaction")
    if (len(relationship_data) >= 3 and relationship_data["frequency_score"].nunique() > 1
            and relationship_data["satisfaction"].nunique() > 1):
        rho, p_value = spearmanr(relationship_data["frequency_score"], relationship_data["satisfaction"])
        decision = "Reject H₀" if p_value < 0.05 else "Fail to reject H₀"
        direction = "positive" if rho > 0 else "negative" if rho < 0 else "no monotonic"
        magnitude = abs(rho)
        strength = "very weak" if magnitude < 0.20 else "weak" if magnitude < 0.40 else "moderate" if magnitude < 0.60 else "strong"
        m1, m2, m3, m4 = st.columns(4)
        with m1: metric_card("Spearman's rho (ρ)", f"{rho:.3f}")
        with m2: metric_card("p-value", f"{p_value:.4f}")
        with m3: metric_card("Valid paired responses", f"{len(relationship_data)}")
        with m4: metric_card("Test decision", decision)
        st.markdown(
            f'<div class="insight"><b>Interpretation:</b> The association is {strength} and {direction} '
            f'(ρ = {rho:.3f}). The p-value is {p_value:.4f}, compared with α = 0.05. '
            f'<b>Decision: {decision}.</b> '
            f'{"There is evidence of a statistically significant association in this sample." if p_value < 0.05 else "There is insufficient evidence of a statistically significant relationship in this sample. This does not prove that no relationship exists."} '
            'Correlation does not establish cause and effect.</div>', unsafe_allow_html=True)

        plotdata = relationship_data.copy()
        plotdata["Frequency category"] = plotdata["frequency_score"].map(freq_labels)
        rng_x = np.random.default_rng(12)
        rng_y = np.random.default_rng(21)
        plotdata["Frequency jitter"] = plotdata["frequency_score"] + rng_x.uniform(-0.10, 0.10, len(plotdata))
        plotdata["Rating jitter"] = plotdata["satisfaction"] + rng_y.uniform(-0.08, 0.08, len(plotdata))
        mean_scores = plotdata.groupby("frequency_score")["satisfaction"].mean().reset_index()
        fig = px.scatter(plotdata, x="Frequency jitter", y="Rating jitter", color="Frequency category",
                         color_discrete_map={"Rarely": BLUE, "Sometimes": CYAN, "Very Often": PURPLE, "Everyday": PINK},
                         hover_data={"Frequency category": True, "frequency_score": False,
                                     "Frequency jitter": False, "Rating jitter": False},
                         title="AI usage frequency and student satisfaction")
        fig.add_trace(go.Scatter(x=mean_scores["frequency_score"], y=mean_scores["satisfaction"],
                                 mode="lines+markers", name="Mean satisfaction",
                                 line=dict(color=ORANGE, width=3), marker=dict(size=10)))
        fig.update_xaxes(tickvals=[1,2,3,4], ticktext=["Rarely","Sometimes","Very Often","Everyday"], title="AI usage frequency")
        fig.update_yaxes(range=[0.7,5.3], dtick=1, title="Student satisfaction (1–5)")
        st.plotly_chart(chart_layout(fig, 420), use_container_width=True)
    else:
        st.warning("There are not enough distinct paired satisfaction responses to calculate Spearman's rank correlation for the current filters.")

    # Additional RQ5 analysis: frequency and perceived usefulness. This runs independently.
    st.markdown("### Additional relationship analysis · Usage frequency and perceived usefulness")
    usefulness_data = filtered[["frequency_score", "usefulness"]].copy()
    usefulness_data["usefulness"] = pd.to_numeric(usefulness_data["usefulness"], errors="coerce")
    usefulness_data = usefulness_data.dropna()
    if (len(usefulness_data) >= 3 and usefulness_data["frequency_score"].nunique() > 1
            and usefulness_data["usefulness"].nunique() > 1):
        use_rho, use_p = spearmanr(usefulness_data["frequency_score"], usefulness_data["usefulness"])
        use_result = "Statistically significant" if use_p < 0.05 else "Not statistically significant"
        use_direction = "positive" if use_rho > 0 else "negative" if use_rho < 0 else "no monotonic"
        u1, u2, u3 = st.columns(3)
        with u1: metric_card("Usefulness Spearman's rho", f"{use_rho:.3f}")
        with u2: metric_card("Usefulness p-value", f"{use_p:.4f}")
        with u3: metric_card("Usefulness result", use_result)
        st.markdown(
            f'<div class="insight"><b>Interpretation:</b> The association between usage frequency and perceived usefulness is {use_direction} '
            f'(ρ = {use_rho:.3f}) and {use_result.lower()} at α = 0.05. '
            f'{"There is evidence of a statistically significant association in this sample." if use_p < 0.05 else "There is insufficient evidence of a statistically significant association in this sample."} '
            'This is an additional analysis, not a second formal hypothesis. Correlation does not show that frequent AI use causes greater perceived usefulness.</div>',
            unsafe_allow_html=True)

        use_plot = usefulness_data.copy()
        use_plot["Frequency category"] = use_plot["frequency_score"].map(freq_labels)
        rng_ux = np.random.default_rng(32)
        rng_uy = np.random.default_rng(42)
        use_plot["Frequency jitter"] = use_plot["frequency_score"] + rng_ux.uniform(-0.10, 0.10, len(use_plot))
        use_plot["Usefulness jitter"] = use_plot["usefulness"] + rng_uy.uniform(-0.08, 0.08, len(use_plot))
        use_means = use_plot.groupby("frequency_score")["usefulness"].mean().reset_index()
        use_fig = px.scatter(use_plot, x="Frequency jitter", y="Usefulness jitter", color="Frequency category",
                             color_discrete_map={"Rarely": BLUE, "Sometimes": CYAN, "Very Often": PURPLE, "Everyday": PINK},
                             hover_data={"Frequency category": True, "frequency_score": False,
                                         "Frequency jitter": False, "Usefulness jitter": False},
                             title="AI usage frequency and perceived usefulness")
        use_fig.add_trace(go.Scatter(x=use_means["frequency_score"], y=use_means["usefulness"],
                                     mode="lines+markers", name="Mean usefulness",
                                     line=dict(color=ORANGE, width=3), marker=dict(size=10)))
        use_fig.update_xaxes(tickvals=[1,2,3,4], ticktext=["Rarely","Sometimes","Very Often","Everyday"], title="AI usage frequency")
        use_fig.update_yaxes(range=[0.7,5.3], dtick=1, title="Perceived usefulness (1–5)")
        st.plotly_chart(chart_layout(use_fig, 420), use_container_width=True)
    else:
        st.info("There are not enough distinct paired usefulness responses to calculate Spearman's rank correlation for the current filters.")

    tabs=st.tabs(["Research questions","Descriptive statistics","Group comparisons","Method & limitations"])

    with tabs[0]:
        section("Findings by research question")
        tool_tbl=multi_counts(filtered,"tools_used").sort_values("Students",ascending=False)
        use_tbl=multi_counts(filtered,"use_cases").sort_values("Students",ascending=False)
        ben_tbl=multi_counts(filtered,"benefits").sort_values("Students",ascending=False)
        ch_tbl=multi_counts(filtered,"challenges").sort_values("Students",ascending=False)
        freq_tbl=filtered["frequency"].value_counts()
        if len(tool_tbl):
            st.markdown("**RQ1. What are the most commonly used AI tools?**")
            top=tool_tbl.iloc[0]
            st.write(f"{top['Response']} is the most selected tool in the filtered sample "
                     f"({int(top['Students'])} respondents; {top['Percent of respondents']:.1f}%).")
        st.markdown("**RQ2. How frequently do college students use AI tools?**")
        if len(freq_tbl):
            common=freq_tbl.idxmax()
            st.write(f"The most frequently reported category is **{common}** "
                     f"({int(freq_tbl.max())} respondents; {freq_tbl.max()/n*100:.1f}%).")
        st.markdown("**RQ3. What are the main purposes of AI tool usage?**")
        if len(use_tbl):
            top=use_tbl.iloc[0]
            st.write(f"The most selected purpose is **{top['Response']}** "
                     f"({int(top['Students'])} respondents; {top['Percent of respondents']:.1f}%).")
        st.markdown("**RQ4. What benefits and challenges do students experience?**")
        q1,q2=st.columns(2)
        with q1:
            st.caption("Most selected benefits")
            st.dataframe(ben_tbl.head(5).rename(columns={"Response":"Benefit"}),hide_index=True,use_container_width=True)
        with q2:
            st.caption("Most selected challenges")
            st.dataframe(ch_tbl.head(5).rename(columns={"Response":"Challenge"}),hide_index=True,use_container_width=True)
        if not ben_tbl.empty and not ch_tbl.empty:
            top_benefit, top_challenge = ben_tbl.iloc[0], ch_tbl.iloc[0]
            st.markdown(f"**Interpretation:** The most frequently reported benefit is **{top_benefit['Response']}** ({int(top_benefit['Students'])} selections), while the most frequently reported challenge is **{top_challenge['Response']}** ({int(top_challenge['Students'])} selections). These results show that students can experience both practical benefits and concerns when using AI tools.")
        st.markdown("**RQ5. Is there a relationship between AI tool usage and perceived satisfaction or usefulness?**")
        st.write("The two Spearman analyses above examine whether AI usage frequency is associated with students’ satisfaction and perceived usefulness. Both results are interpreted using α = 0.05; neither test establishes cause and effect.")

    with tabs[1]:
        section("Descriptive statistics","Mean, median, mode, range, variance and standard deviation for numeric variables.")
        numeric_cols=["age","hours_per_week","overdependence","satisfaction","usefulness"]
        names={"age":"Age","hours_per_week":"Weekly AI hours","overdependence":"Overdependence rating",
               "satisfaction":"Satisfaction rating","usefulness":"Usefulness rating"}
        rows=[]
        for col in numeric_cols:
            s=filtered[col].dropna()
            if len(s):
                mode=s.mode()
                rows.append({"Variable":names[col],"N":len(s),"Mean":s.mean(),"Median":s.median(),
                             "Mode":mode.iloc[0] if len(mode) else np.nan,"Minimum":s.min(),
                             "Maximum":s.max(),"Variance":s.var(),"Std. deviation":s.std()})
        stats=pd.DataFrame(rows).round(2)
        st.dataframe(stats,use_container_width=True,hide_index=True)
        st.download_button("⬇ Download descriptive statistics",stats.to_csv(index=False).encode("utf-8"),
                           "descriptive_statistics.csv","text/csv")
        st.markdown("**Distribution of weekly AI usage hours**")
        h1,h2=st.columns(2)
        with h1:
            fig=px.histogram(filtered,x="hours_per_week",nbins=14,marginal="violin",
                             color_discrete_sequence=[CYAN],title="Weekly hours: distribution")
            st.plotly_chart(chart_layout(fig,370,False),use_container_width=True)
        with h2:
            fig=px.box(filtered,y="hours_per_week",points="all",color_discrete_sequence=[PURPLE],
                       title="Weekly hours: spread and potential outliers")
            st.plotly_chart(chart_layout(fig,370,False),use_container_width=True)
        st.caption("The original hours are retained for descriptive statistics. The Dashboard's group box plot uses an IQR-capped view, matching the exploratory notebook approach.")

    with tabs[2]:
        section("Compare average indicators across student groups")
        group_by=st.selectbox("Group respondents by",["academic_year","gender","programme_level","age_group"],format_func=lambda x:{
            "academic_year":"Academic year","gender":"Gender","programme_level":"Programme level","age_group":"Age group"}[x])
        measures=["hours_per_week","satisfaction","usefulness","overdependence"]
        labels={"hours_per_week":"Weekly AI hours","satisfaction":"Satisfaction","usefulness":"Usefulness","overdependence":"Overdependence"}
        grouped=filtered.groupby(group_by,observed=True)[measures].mean().reset_index()
        counts=filtered.groupby(group_by,observed=True).size().rename("Respondents").reset_index()
        grouped=grouped.merge(counts,on=group_by)
        st.dataframe(grouped.rename(columns={group_by:"Group",**labels}).round(2),hide_index=True,use_container_width=True)
        if not grouped.empty:
            highest_hours = grouped.loc[grouped["hours_per_week"].idxmax()]
            highest_sat = grouped.loc[grouped["satisfaction"].idxmax()]
            st.markdown(f"**Group insight:** {highest_hours[group_by]} has the highest average weekly AI usage ({highest_hours['hours_per_week']:.2f} hours), while {highest_sat[group_by]} has the highest average satisfaction ({highest_sat['satisfaction']:.2f}/5). These are descriptive differences only; they do not establish statistically significant group differences.")
        metric=st.selectbox("Visualize measure",measures,index=1,format_func=lambda x:labels[x])
        fig=px.bar(grouped,x=group_by,y=metric,color=group_by,text=metric,
                   color_discrete_sequence=PALETTE,title=f"{labels[metric]} by {group_by.replace('_',' ')}",
                   hover_data={"Respondents":True})
        fig.update_traces(texttemplate="%{text:.2f}",textposition="outside")
        fig.update_layout(showlegend=False,xaxis_title="",yaxis_title=labels[metric])
        if metric!="hours_per_week": fig.update_yaxes(range=[0,5.5])
        st.plotly_chart(chart_layout(fig,400,False),use_container_width=True)
        st.caption("Group averages are descriptive. They do not by themselves establish statistically significant differences.")

    with tabs[3]:
        section("Methodology and interpretation notes")
        st.markdown("**Data quality summary · cleaned dataset**")
        quality = pd.DataFrame({
            "Check": ["Records", "Variables", "Missing cells", "Duplicate rows", "Duplicate student IDs"],
            "Result": [
                len(df), len(df.columns), int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
                int(df["student_id"].duplicated().sum()) if "student_id" in df.columns else "Not available"
            ]
        })
        st.dataframe(quality, hide_index=True, use_container_width=True)
        st.caption("The source survey contained 210 records; the cleaned dataset used here contains 182 records. The reasons for excluding records should be explained using the group's data-cleaning notebook.")
        st.markdown("""
**Dataset and sample**
- The original survey file contains 210 records; the cleaned dataset supplied for analysis contains 182 records.
- Dashboard calculations use the cleaned dataset and the currently selected filters.
- The data are self-reported survey responses from the selected participants.

**Multiple-response questions**
- AI tools, purposes, benefits and challenges allow multiple selections.
- Each selected option is counted once per respondent's selection list; category percentages can therefore add up to more than 100%.

**Statistical approach**
- Frequencies and percentages summarize categorical responses.
- Mean, median, mode, variance and standard deviation summarize numerical and rating variables.
- Spearman's rank correlation examines the association between the ordinal usage-frequency score and both satisfaction and perceived usefulness.
- A simple least-squares linear model, calculated with NumPy, estimates satisfaction from weekly AI usage hours using an 80/20 train-test split.

**Limitations**
- The sample may not represent all college students.
- Self-reported answers may include recall or response bias.
- Correlation indicates association, not cause and effect.
- Filtered results describe only the respondents remaining after the selected filters.
        """)
        st.markdown("**Visualization note:** Matplotlib is used in the analysis workflow/PPT visualizations. Plotly is used in this Streamlit dashboard to make the displayed charts interactive. Both are Python visualization libraries.")
        st.markdown("**Data cleaning note:** weekly usage-hour outliers are not deleted from the dataset. The app preserves the original values; an IQR-capped version is used only for the dashboard's academic-year box plot.")

    st.markdown("---")
    st.download_button("⬇ Download current filtered dataset",
                       filtered.to_csv(index=False).encode("utf-8"),
                       "ai_tool_usage_filtered.csv","text/csv")

st.markdown("---")
st.markdown('<div class="small-muted">PRG405 · AI Tool Usage Among College Students · Built with Streamlit, Pandas, NumPy and Plotly</div>',unsafe_allow_html=True)
