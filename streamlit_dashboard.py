"""
dashboard.py
Run with: streamlit run dashboard.py
Expects in same directory:
  - val_predictions.csv
  - model.pkl
  - processed_train.csv
  - processed_test.csv   (optional, for future week inference)
  - meal_info.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FoodSense — Demand & Waste Intelligence",
    page_icon="🌾",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #0f1923;
    border: 1px solid #1e3048;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label {
    color: #7a9bb5 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8f0f7 !important;
    font-size: 1.9rem !important;
    font-weight: 600;
}
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

/* Main background */
.stApp { background-color: #080f18; color: #d4e4f0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0a1520 !important;
    border-right: 1px solid #1a2e42;
}
[data-testid="stSidebar"] * { color: #a8c4d8 !important; }

/* Section headers */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    color: #5ba3cc;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 28px 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #1a2e42;
}

/* Surplus alert */
.surplus-alert {
    background: linear-gradient(135deg, #1a0a0a, #2a1010);
    border: 1px solid #6b2020;
    border-left: 4px solid #e05252;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #f5c6c6;
    font-size: 0.9rem;
}
.surplus-ok {
    background: linear-gradient(135deg, #0a1a0a, #0f2a15);
    border: 1px solid #1a5c2a;
    border-left: 4px solid #3db860;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #b6f0c6;
    font-size: 0.9rem;
}

/* Redistribution card */
.redist-card {
    background: #0d1e2e;
    border: 1px solid #1a3a52;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 6px 0;
    font-size: 0.88rem;
    color: #c0d8ea;
}
.redist-card strong { color: #5bc4e8; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 8px; }

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    color: #7a9bb5 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #5bc4e8 !important;
    border-bottom-color: #5bc4e8 !important;
}

/* Title area */
.title-block {
    padding: 28px 0 8px 0;
    border-bottom: 1px solid #1a2e42;
    margin-bottom: 24px;
}
.title-block h1 { color: #e8f4fb; font-size: 2.2rem; margin: 0; }
.title-block p  { color: #6a94b0; font-size: 0.92rem; margin: 4px 0 0 0; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
DATA_DIR = Path(".")

@st.cache_data
def load_data():
    val   = pd.read_csv(DATA_DIR / "val_predictions.csv")
    train = pd.read_csv(DATA_DIR / "processed_train.csv")
    meal  = pd.read_csv(DATA_DIR / "meal_info.csv")
    return val, train, meal

@st.cache_resource
def load_model():
    with open(DATA_DIR / "model.pkl", "rb") as f:
        return pickle.load(f)

try:
    val_df, train_df, meal_df = load_data()
    model = load_model()
    data_ok = True
except FileNotFoundError as e:
    st.error(f"Missing file: {e}. Make sure val_predictions.csv, processed_train.csv, meal_info.csv and model.pkl are in the same directory.")
    st.stop()

# Merge meal names into val_df
val_df = val_df.merge(meal_df[["meal_id","category","cuisine"]], on="meal_id", how="left")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 FoodSense")
    st.markdown("---")
    st.markdown("**Filters**")

    all_centers = sorted(val_df["center_id"].unique())
    selected_centers = st.multiselect(
        "Centers", all_centers, default=all_centers[:5],
        help="Filter by serving center"
    )

    all_weeks = sorted(val_df["week"].unique())
    week_range = st.select_slider(
        "Week range",
        options=all_weeks,
        value=(all_weeks[0], all_weeks[-1])
    )

    surplus_threshold = st.slider(
        "Surplus alert threshold (orders)",
        min_value=0, max_value=5000, value=500, step=100,
        help="Flag weeks where total surplus exceeds this value"
    )

    st.markdown("---")
    st.markdown("**Model Info**")
    st.caption(f"Best iteration: 751")
    st.caption(f"Validation RMSLE: 0.4915")
    st.caption(f"Val weeks: 134–145")

# ── Filter data ───────────────────────────────────────────────────────────────
filtered = val_df[
    (val_df["center_id"].isin(selected_centers)) &
    (val_df["week"] >= week_range[0]) &
    (val_df["week"] <= week_range[1])
].copy()

# ensure numeric
filtered["actual"]    = pd.to_numeric(filtered["actual"],    errors="coerce")
filtered["predicted"] = pd.to_numeric(filtered["predicted"], errors="coerce")
filtered["surplus"]   = pd.to_numeric(filtered["surplus"],   errors="coerce").fillna(0)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <h1>FoodSense</h1>
  <p>AI-powered food demand forecasting & waste reduction intelligence</p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
total_predicted = int(filtered["predicted"].sum())
total_actual    = int(filtered["actual"].sum())
total_surplus   = int(filtered["surplus"].sum())
waste_pct       = (total_surplus / total_predicted * 100) if total_predicted > 0 else 0
avg_mae         = float((filtered["predicted"] - filtered["actual"]).abs().mean())
avg_surplus_per_center = total_surplus / max(len(selected_centers) * len(filtered["week"].unique()), 1)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Predicted Orders", f"{total_predicted:,}")
c2.metric("Total Actual Orders",    f"{total_actual:,}")
c3.metric("Estimated Surplus",      f"{total_surplus:,}", delta=f"{waste_pct:.1f}% of predicted", delta_color="inverse")
# c4.metric("Avg MAE / row",          f"{avg_mae:.1f}")
c4.metric("Avg surplus / center / week", f"{avg_surplus_per_center:.0f} orders")
c5.metric("Centers shown",          f"{len(selected_centers)}")

st.markdown("")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Demand Forecast",
    "⚠️  Surplus Alerts",
    "♻️  Redistribution",
    "🔍  Drill Down",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Demand Forecast
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Weekly Demand — Predicted vs Actual</div>', unsafe_allow_html=True)

    weekly = (
        filtered.groupby("week")[["actual","predicted","surplus"]]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0d1a26")
    ax.set_facecolor("#0d1a26")

    ax.fill_between(weekly["week"], weekly["actual"],    alpha=0.15, color="#3db860")
    ax.fill_between(weekly["week"], weekly["predicted"], alpha=0.10, color="#5bc4e8")
    ax.plot(weekly["week"], weekly["actual"],    color="#3db860", lw=2,   label="Actual",    marker="o", markersize=4)
    ax.plot(weekly["week"], weekly["predicted"], color="#5bc4e8", lw=2,   label="Predicted", marker="s", markersize=4)
    ax.fill_between(weekly["week"], weekly["surplus"],   alpha=0.25, color="#e05252", label="Surplus")

    ax.set_xlabel("Week", color="#6a94b0", fontsize=9)
    ax.set_ylabel("Total Orders", color="#6a94b0", fontsize=9)
    ax.tick_params(colors="#6a94b0", labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor("#1a2e42")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(facecolor="#0d1a26", edgecolor="#1a2e42", labelcolor="#a8c4d8", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Per-center breakdown
    st.markdown('<div class="section-header">Per-Center Weekly Totals</div>', unsafe_allow_html=True)
    center_weekly = (
        filtered.groupby(["week","center_id"])[["predicted","surplus"]]
        .sum()
        .reset_index()
    )
    pivot = center_weekly.pivot(index="week", columns="center_id", values="predicted").fillna(0)
    fig2, ax2 = plt.subplots(figsize=(12, 3.5))
    fig2.patch.set_facecolor("#0d1a26")
    ax2.set_facecolor("#0d1a26")
    colors = plt.cm.cool(np.linspace(0.2, 0.9, len(pivot.columns)))
    for i, col in enumerate(pivot.columns):
        ax2.plot(pivot.index, pivot[col], lw=1.5, label=f"C{col}", color=colors[i], alpha=0.85)
    ax2.set_xlabel("Week", color="#6a94b0", fontsize=9)
    ax2.set_ylabel("Predicted Orders", color="#6a94b0", fontsize=9)
    ax2.tick_params(colors="#6a94b0", labelsize=8)
    for spine in ax2.spines.values(): spine.set_edgecolor("#1a2e42")
    ax2.legend(facecolor="#0d1a26", edgecolor="#1a2e42", labelcolor="#a8c4d8",
               fontsize=7, ncol=5, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Surplus Alerts
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Surplus Alert by Week</div>', unsafe_allow_html=True)

    weekly_surplus = (
        filtered.groupby("week")["surplus"].sum().reset_index()
        .rename(columns={"surplus": "total_surplus"})
    )

    for _, row in weekly_surplus.iterrows():
        w, s = int(row["week"]), int(row["total_surplus"])
        if s >= surplus_threshold:
            st.markdown(f"""
            <div class="surplus-alert">
                ⚠️ <strong>Week {w}</strong> — Estimated surplus: <strong>{s:,} orders</strong>
                &nbsp;·&nbsp; Exceeds threshold by {s - surplus_threshold:,}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="surplus-ok">
                ✓ <strong>Week {w}</strong> — Surplus: <strong>{s:,} orders</strong> — Within acceptable range
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Top Surplus Meals</div>', unsafe_allow_html=True)
    meal_surplus = (
        filtered.groupby(["meal_id","category","cuisine"])["surplus"]
        .sum()
        .reset_index()
        .sort_values("surplus", ascending=False)
        .head(10)
    )
    meal_surplus["surplus"] = meal_surplus["surplus"].astype(int)
    st.dataframe(
        meal_surplus.rename(columns={"meal_id":"Meal ID","category":"Category",
                                      "cuisine":"Cuisine","surplus":"Total Surplus"}),
        use_container_width=True, hide_index=True
    )

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Redistribution
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Redistribution Suggestions</div>', unsafe_allow_html=True)
    st.caption("Centers with surplus are matched with centers showing deficit for the same week.")

    # Identify surplus centers and deficit centers per week
    center_week = (
        filtered.groupby(["week","center_id"])[["predicted","actual"]]
        .sum()
        .reset_index()
    )
    center_week["surplus"]  = (center_week["predicted"] - center_week["actual"]).clip(lower=0).astype(int)
    center_week["deficit"]  = (center_week["actual"] - center_week["predicted"]).clip(lower=0).astype(int)

    # NGO/beneficiary list (static, as allowed by problem statement)
    BENEFICIARIES = [
        {"name": "Akshaya Patra Foundation",  "capacity": 5000},
        {"name": "Robin Hood Army",            "capacity": 2000},
        {"name": "No Food Waste NGO",          "capacity": 3000},
        {"name": "Community Kitchen Chennai",  "capacity": 1500},
    ]

    weeks_with_surplus = center_week[center_week["surplus"] > surplus_threshold]["week"].unique()

    if len(weeks_with_surplus) == 0:
        st.info("No weeks with surplus above threshold. Lower the slider in the sidebar to see suggestions.")
    else:
        for week in sorted(weeks_with_surplus):
            wdata = center_week[center_week["week"] == week]
            surplus_centers = wdata[wdata["surplus"] > 0].sort_values("surplus", ascending=False)
            deficit_centers = wdata[wdata["deficit"] > 0].sort_values("deficit", ascending=False)
            total_week_surplus = int(surplus_centers["surplus"].sum())

            st.markdown(f"**Week {int(week)}** — Total redistributable: {total_week_surplus:,} orders")

            col_a, col_b = st.columns(2)

            with col_a:
                st.caption("🔴 Surplus Centers")
                for _, r in surplus_centers.head(3).iterrows():
                    st.markdown(f"""<div class="redist-card">
                        Center <strong>{int(r.center_id)}</strong> —
                        surplus <strong>{int(r.surplus):,}</strong> orders
                    </div>""", unsafe_allow_html=True)

            with col_b:
                st.caption("🟢 Suggested Beneficiaries")
                remaining = total_week_surplus
                for b in BENEFICIARIES:
                    alloc = min(remaining, b["capacity"])
                    if alloc <= 0:
                        break
                    st.markdown(f"""<div class="redist-card">
                        <strong>{b['name']}</strong> —
                        allocate <strong>{alloc:,}</strong> portions
                    </div>""", unsafe_allow_html=True)
                    remaining -= alloc

            st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Drill Down
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Center × Meal Drill Down</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    sel_center = col1.selectbox("Select Center", sorted(filtered["center_id"].unique()))
    sel_meal   = col2.selectbox("Select Meal",   sorted(filtered[filtered["center_id"]==sel_center]["meal_id"].unique()))

    drill = filtered[
        (filtered["center_id"] == sel_center) &
        (filtered["meal_id"]   == sel_meal)
    ].sort_values("week")

    if len(drill) == 0:
        st.warning("No data for this combination in selected week range.")
    else:
        meal_meta = meal_df[meal_df["meal_id"] == sel_meal]
        if len(meal_meta):
            st.caption(f"Category: **{meal_meta.iloc[0]['category']}** · Cuisine: **{meal_meta.iloc[0]['cuisine']}**")

        fig3, ax3 = plt.subplots(figsize=(10, 3.5))
        fig3.patch.set_facecolor("#0d1a26")
        ax3.set_facecolor("#0d1a26")
        ax3.bar(drill["week"], drill["surplus"],   color="#e05252", alpha=0.5, label="Surplus",   width=0.4)
        ax3.plot(drill["week"], drill["actual"],    color="#3db860", lw=2, marker="o", markersize=4, label="Actual")
        ax3.plot(drill["week"], drill["predicted"], color="#5bc4e8", lw=2, marker="s", markersize=4, label="Predicted", linestyle="--")
        ax3.set_xlabel("Week", color="#6a94b0", fontsize=9)
        ax3.set_ylabel("Orders", color="#6a94b0", fontsize=9)
        ax3.tick_params(colors="#6a94b0", labelsize=8)
        for spine in ax3.spines.values(): spine.set_edgecolor("#1a2e42")
        ax3.legend(facecolor="#0d1a26", edgecolor="#1a2e42", labelcolor="#a8c4d8", fontsize=8)
        ax3.set_title(f"Center {sel_center} · Meal {sel_meal}", color="#a8c4d8", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        st.dataframe(
            drill[["week","actual","predicted","surplus"]].rename(
                columns={"week":"Week","actual":"Actual","predicted":"Predicted","surplus":"Surplus"}
            ).astype({"Actual": int, "Predicted": int, "Surplus": int}),
            use_container_width=True, hide_index=True
        )