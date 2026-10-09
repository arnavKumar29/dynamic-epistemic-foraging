"""
app.py — Dynamic Epistemic Foraging: 6-Tab Research Telemetry Console

Run this separately from the simulation:
    streamlit run src/dashboard/app.py

The dashboard reads execution_ledger.jsonl (JSONL format written by
StructuredEventLedger) and auto-refreshes in real time.

Tabs
----
  1. Mission Overview      — KPI metrics, scores, attack Gantt chart
  2. Bayesian Inspector    — Step-by-step DBN forward-filter math
  3. Beta Distribution     — Live trust PDF visualizer + grudge state
  4. D* Lite Planner       — Hazard heatmap, cost comparisons, replan log
  5. Epistemic Events      — Paradox table, SMT results, intent posteriors
  6. Network & Crypto      — MITM timeline, crypto faults, ground truth overlay
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────── #
# Page config                                                              #
# ─────────────────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title  = "DEF — Research Console",
    page_icon   = "🤖",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────── #
# Global CSS                                                               #
# ─────────────────────────────────────────────────────────────────────── #
st.markdown("""
<style>
.stApp { background-color: #0a0c17; }
.block-container { padding-top: 1rem; }
h1, h2, h3 { color: #e0e4f0 !important; }
.metric-card {
    background: #12172a;
    border-radius: 10px;
    padding: 14px 18px;
    border-left: 4px solid #3d7def;
    margin-bottom: 8px;
}
.stDataFrame thead th { background-color: #1a1f35 !important; color: #b0c4de !important; }
.math-box {
    background: #0f1525;
    border: 1px solid #2a3555;
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'Consolas', monospace;
    font-size: 12px;
    color: #a0c8ff;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────── #
# Sidebar                                                                  #
# ─────────────────────────────────────────────────────────────────────── #
with st.sidebar:
    st.markdown("## ⚙️ Dashboard Controls")
    log_path         = st.text_input("Ledger path", "logs/execution_ledger.jsonl")
    refresh_rate     = st.slider("Auto-refresh (s)", 1, 10, 2)
    max_events       = st.slider("Max events loaded", 200, 5000, 1000)
    trust_threshold  = st.slider("Trust reroute threshold", 0.10, 0.90, 0.40, 0.05)
    st.divider()
    st.markdown("### Legend")
    st.markdown("🔴 **ATTACK** · 🟡 **FAULT** · 🟢 **HONEST**")
    st.markdown("**Grudge tiers:**")
    st.markdown("NORMAL → MODERATE → SLOW → **FROZEN**")
    st.divider()
    st.markdown("### Quick start")
    st.code("python -m src.main\nstreamlit run src/dashboard/app.py", language="bash")


# ─────────────────────────────────────────────────────────────────────── #
# Data loading                                                             #
# ─────────────────────────────────────────────────────────────────────── #
@st.cache_data(ttl=1.0)
def load_ledger(path: str, limit: int) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records[-limit:]


records = load_ledger(log_path, max_events)

if not records:
    st.title("🤖 Dynamic Epistemic Foraging — Research Console")
    st.info("⏳ Awaiting simulation data.\n\n```\npython -m src.main\n```")
    st.stop()

df = pd.DataFrame(records)

# ── Event type sub-frames ────────────────────────────────────────────── #
def evt(name: str) -> pd.DataFrame:
    if "event" not in df.columns:
        return pd.DataFrame()
    mask = df["event"] == name
    return df[mask].copy()


trust_df    = evt("TRUST_UPDATE")
dbn_df      = evt("DBN_STEP")
paradox_df  = evt("EPISTEMIC_PARADOX")
replan_df   = evt("REPLAN")
hazard_df   = evt("HAZARD_SNAPSHOT")
attack_df   = evt("ATTACK_WINDOW")
crypto_df   = evt("CRYPTO_FAULT")
sim_end_df  = evt("SIMULATION_END")


# ─────────────────────────────────────────────────────────────────────── #
# Header                                                                   #
# ─────────────────────────────────────────────────────────────────────── #
st.title("🤖 Dynamic Epistemic Foraging — Research Console")
st.caption("Live telemetry · DBN math · Beta trust · D\\* Lite planning · SMT/ToM reasoning · Network/Crypto")

# ─────────────────────────────────────────────────────────────────────── #
# Global metrics row                                                       #
# ─────────────────────────────────────────────────────────────────────── #
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Events",      len(records))
c2.metric("Paradoxes",         len(paradox_df))
c3.metric("Trust Updates",     len(trust_df))
c4.metric("Attack Windows",    len(attack_df))
c5.metric("Replans",           len(replan_df))
c6.metric("Crypto Faults",     len(crypto_df))

st.divider()

# ─────────────────────────────────────────────────────────────────────── #
# Tabs                                                                     #
# ─────────────────────────────────────────────────────────────────────── #
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📡 Mission Overview",
    "🧠 Bayesian Inspector",
    "📊 Beta Distribution",
    "🗺️ D* Lite Planner",
    "⚡ Epistemic Events",
    "🔐 Network & Crypto",
])


# ══════════════════════════════════════════════════════════════════════ #
#  TAB 1 — Mission Overview                                               #
# ══════════════════════════════════════════════════════════════════════ #
with tab1:
    st.subheader("📡 Mission Overview")

    # ── Score leaderboard ──────────────────────────────────────────── #
    if not sim_end_df.empty:
        try:
            end_row = sim_end_df.iloc[-1]
            scores  = end_row.get("scores", {})
            detects = end_row.get("detections", {})
            if scores:
                sb_data = [
                    {"Agent": k, "Resources": v, "Attack Detections": detects.get(k, 0)}
                    for k, v in scores.items()
                ]
                score_df = pd.DataFrame(sb_data).sort_values("Resources", ascending=False)
                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown("#### 🏆 Resource Leaderboard")
                    st.dataframe(score_df, use_container_width=True, height=180)
                with col_r:
                    fig_sc = px.bar(
                        score_df, x="Agent", y="Resources", color="Agent",
                        title="Resources Collected per Agent",
                        template="plotly_dark",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    st.plotly_chart(fig_sc, use_container_width=True)
        except Exception:
            pass

    # ── Trust evolution ──────────────────────────────────────────────── #
    if not trust_df.empty:
        st.markdown("#### 📈 Trust Score Evolution")
        trust_rows = []
        for _, row in trust_df.iterrows():
            trust_rows.append({
                "tick":     row.get("tick", 0),
                "peer_id":  row.get("peer_id", "?"),
                "trust":    row.get("new_trust", 0.5),
                "p_attack": row.get("p_attack", 0.0),
                "verdict":  row.get("verdict", "HONEST"),
            })
        tdf = pd.DataFrame(trust_rows)

        fig_t = px.line(
            tdf, x="tick", y="trust", color="peer_id",
            labels={"trust": "Trust Score", "tick": "Tick"},
            template="plotly_dark",
        )
        fig_t.add_hline(y=trust_threshold, line_dash="dot", line_color="orange",
                        annotation_text=f"Reroute threshold ({trust_threshold})")
        fig_t.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_t, use_container_width=True)

        # Add attack window shading
        if not attack_df.empty:
            for _, aw in attack_df.drop_duplicates(subset=["attacker_id","target_id"]).iterrows():
                t0 = aw.get("tick", 0)
                # Estimate window end from schedule
                fig_t.add_vrect(
                    x0=t0 - 5, x1=t0 + 5,
                    fillcolor="red", opacity=0.15, layer="below", line_width=0,
                )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 🎯 P(Attack) over Time")
            fig_pa = px.line(tdf, x="tick", y="p_attack", color="peer_id",
                             template="plotly_dark")
            fig_pa.add_hline(y=0.5, line_dash="dash", line_color="red",
                             annotation_text="Decision threshold (0.5)")
            st.plotly_chart(fig_pa, use_container_width=True)

        with col_b:
            st.markdown("#### 📊 Current Trust Snapshot")
            latest = tdf.sort_values("tick").groupby("peer_id").last().reset_index()
            colors = ["#e74c3c" if t < trust_threshold else "#27ae60" for t in latest["trust"]]
            fig_bar = go.Figure(go.Bar(
                x=latest["peer_id"], y=latest["trust"], marker_color=colors,
                text=[f"{t:.3f}" for t in latest["trust"]], textposition="auto",
            ))
            fig_bar.update_layout(yaxis_range=[0,1], template="plotly_dark",
                                  title="Latest Trust per Peer")
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── Attack Gantt ─────────────────────────────────────────────────── #
    if not attack_df.empty:
        st.markdown("#### 🗓️ Attack Timeline (Ground Truth)")
        gantt_rows = []
        for _, row in attack_df.iterrows():
            gantt_rows.append({
                "Task":    f"A{row.get('attacker_id','?')}→A{row.get('target_id','?')}",
                "Start":   row.get("tick", 0),
                "Finish":  row.get("tick", 0) + 10,
                "Mode":    row.get("mode", "?"),
            })
        if gantt_rows:
            gdf = pd.DataFrame(gantt_rows)
            fig_g = px.bar(gdf, x="Start", y="Task", color="Mode",
                           template="plotly_dark",
                           title="Ground-Truth Attack Events",
                           color_discrete_map={
                               "position_forge": "#e74c3c",
                               "replay":         "#f39c12",
                               "drop":           "#9b59b6",
                           })
            st.plotly_chart(fig_g, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════ #
#  TAB 2 — Bayesian Inspector                                             #
# ══════════════════════════════════════════════════════════════════════ #
with tab2:
    st.subheader("🧠 Bayesian Network Inspector")
    st.caption(
        "Step-by-step DBN forward filter math. "
        "Select an event to see the exact probability table, CPT lookups, "
        "normalization constant Z, and verdict reasoning."
    )

    if dbn_df.empty:
        st.info("No DBN events yet — start the simulation.")
    else:
        peer_opts = ["All"] + sorted(dbn_df["peer_id"].dropna().unique().tolist())
        sel_peer  = st.selectbox("Filter by peer", peer_opts)
        filt_dbn  = dbn_df if sel_peer == "All" else dbn_df[dbn_df["peer_id"] == sel_peer]

        # ── Step-by-step math for the most recent event ──────────────── #
        latest_dbn = filt_dbn.sort_values("tick").iloc[-1] if not filt_dbn.empty else None
        if latest_dbn is not None:
            trace = latest_dbn.get("math_trace", {})
            if not isinstance(trace, dict):
                try:
                    trace = json.loads(str(trace))
                except Exception:
                    trace = {}

            st.markdown("#### Most Recent DBN Forward-Filter Step")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("P(Attack)", f"{latest_dbn.get('p_attack', 0):.4f}")
            col_b.metric("P(Fault)",  f"{latest_dbn.get('p_fault', 0):.4f}")
            col_c.metric("P(Honest)", f"{latest_dbn.get('p_honest', 0):.4f}")

            st.markdown(f"**Verdict:** `{latest_dbn.get('label', '?')}`")
            if "verdict_reasoning" in trace:
                st.info(f"📝 **Reasoning:** {trace['verdict_reasoning']}")

            # Math walkthrough
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 1️⃣ Previous Posterior → Predicted Prior")
                prev = trace.get("prev_posterior", {})
                pred = trace.get("predicted_prior", {})
                trans = trace.get("transition", {})
                math_text = (
                    f"Previous: P(A)={prev.get('p_A', 0):.5f}  P(F)={prev.get('p_F', 0):.5f}\n\n"
                    f"Transition model:\n"
                    f"  P(A_t=1|A_{{t-1}}=1) = {trans.get('P_A_PERSIST', 0.90)}\n"
                    f"  P(A_t=1|A_{{t-1}}=0) = {trans.get('P_A_EMERGE',  0.05)}\n"
                    f"  P(F_t=1|F_{{t-1}}=1) = {trans.get('P_F_PERSIST', 0.80)}\n"
                    f"  P(F_t=1|F_{{t-1}}=0) = {trans.get('P_F_EMERGE',  0.08)}\n\n"
                    f"Predicted: P(A)={pred.get('p_A', 0):.5f}  P(F)={pred.get('p_F', 0):.5f}"
                )
                st.markdown(f"<div class='math-box'><pre>{math_text}</pre></div>",
                            unsafe_allow_html=True)

            with col2:
                st.markdown("##### 2️⃣ Observations")
                obs = trace.get("observations", {})
                ka_names  = ["Low (<15px)", "Medium (15-80px)", "High (>80px)"]
                var_names = ["Low", "High"]
                loss_names= ["Low", "High"]
                obs_text  = (
                    f"KinematicAnomaly = {obs.get('KinematicAnomaly',0)}"
                    f"  [{ka_names[obs.get('KinematicAnomaly',0)]}]\n"
                    f"Variance         = {obs.get('Variance',0)}"
                    f"  [{var_names[obs.get('Variance',0)]}]\n"
                    f"PacketLoss       = {obs.get('PacketLoss',0)}"
                    f"  [{loss_names[obs.get('PacketLoss',0)]}]"
                )
                st.markdown(f"<div class='math-box'><pre>{obs_text}</pre></div>",
                            unsafe_allow_html=True)

            st.markdown("##### 3️⃣ Joint Distribution Table")
            cpt_detail = trace.get("cpt_detail", {})
            if cpt_detail:
                table_rows = []
                for combo_str, vals in cpt_detail.items():
                    norm_p = trace.get("posterior_table", {}).get(
                        "P(" + combo_str.replace("=","=").replace(",",",F=").replace("A=","A=") + ")",
                        None
                    )
                    table_rows.append({
                        "State":         combo_str,
                        "P(A)×P(F)":    f"{vals.get('prior_A',0):.5f} × {vals.get('prior_F',0):.5f}",
                        "P(K|A,F)":     f"{vals.get('P_K',0):.5f}",
                        "P(V|F,L)":     f"{vals.get('P_V',0):.5f}",
                        "Unnorm. Joint": f"{vals.get('joint_unnorm',0):.8f}",
                    })
                jdf = pd.DataFrame(table_rows)
                st.dataframe(jdf, use_container_width=True)

                z = trace.get("normalization_Z", 1.0)
                post = trace.get("posterior_table", {})
                z_text = (
                    f"Normalization constant Z = {z:.8f}\n\n"
                    f"Posterior (normalized):\n"
                    + "\n".join(f"  {k} = {v:.5f}" for k, v in post.items())
                )
                st.markdown(f"<div class='math-box'><pre>{z_text}</pre></div>",
                            unsafe_allow_html=True)

            st.markdown("##### 4️⃣ Posterior Marginals")
            marg   = trace.get("marginals", {})
            bar_df = pd.DataFrame([
                {"Variable": "P(Attack=1)", "Probability": marg.get("p_A", 0)},
                {"Variable": "P(Fault=1)",  "Probability": marg.get("p_F", 0)},
                {"Variable": "P(Honest)",   "Probability": marg.get("p_H", 0)},
            ])
            fig_marg = px.bar(bar_df, x="Variable", y="Probability",
                              color="Variable", template="plotly_dark",
                              title="Posterior Marginals")
            fig_marg.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig_marg, use_container_width=True)

        # ── P(attack) time series with DBN events ────────────────────── #
        st.markdown("#### P(Attack) / P(Fault) / P(Honest) Over Time")
        dbn_ts = filt_dbn[["tick","peer_id","p_attack","p_fault","p_honest"]].dropna()
        if not dbn_ts.empty:
            fig_dbn = go.Figure()
            for pid in dbn_ts["peer_id"].unique():
                sub = dbn_ts[dbn_ts["peer_id"] == pid].sort_values("tick")
                fig_dbn.add_trace(go.Scatter(
                    x=sub["tick"], y=sub["p_attack"],
                    mode="lines", name=f"{pid} P(Attack)",
                    line=dict(color="#e74c3c"),
                ))
                fig_dbn.add_trace(go.Scatter(
                    x=sub["tick"], y=sub["p_fault"],
                    mode="lines", name=f"{pid} P(Fault)",
                    line=dict(color="#f39c12"),
                ))
                fig_dbn.add_trace(go.Scatter(
                    x=sub["tick"], y=sub["p_honest"],
                    mode="lines", name=f"{pid} P(Honest)",
                    line=dict(color="#27ae60"),
                ))
            fig_dbn.add_hline(y=0.5, line_dash="dash", line_color="red")
            fig_dbn.update_layout(template="plotly_dark", yaxis_range=[0, 1])
            st.plotly_chart(fig_dbn, use_container_width=True)

        # ── Case comparison ──────────────────────────────────────────── #
        st.markdown("#### ⚖️ Attack Wave Comparison (Position Forge vs Drop)")
        atk_events = [(ev["tick"], ev.get("mode","")) for _, ev in attack_df.iterrows()] if not attack_df.empty else []
        forge_ticks = [t for t, m in atk_events if m == "position_forge"]
        drop_ticks  = [t for t, m in atk_events if m == "drop"]
        if forge_ticks and drop_ticks:
            ft = min(forge_ticks)
            dt = min(drop_ticks)
            col_cmp_a, col_cmp_b = st.columns(2)
            with col_cmp_a:
                sub_f = dbn_df[(dbn_df["tick"] >= ft - 30) & (dbn_df["tick"] <= ft + 80)]
                if not sub_f.empty:
                    fig_cf = px.line(sub_f, x="tick", y="p_attack", color="peer_id",
                                     title="Position Forge Attack: P(Attack) response",
                                     template="plotly_dark")
                    fig_cf.add_vrect(x0=ft, x1=ft+140, fillcolor="red", opacity=0.12,
                                     layer="below", line_width=0)
                    st.plotly_chart(fig_cf, use_container_width=True)
            with col_cmp_b:
                sub_d = dbn_df[(dbn_df["tick"] >= dt - 30) & (dbn_df["tick"] <= dt + 120)]
                if not sub_d.empty:
                    fig_cd = px.line(sub_d, x="tick", y="p_attack", color="peer_id",
                                     title="Drop/DoS Attack: P(Attack) response",
                                     template="plotly_dark")
                    fig_cd.add_vrect(x0=dt, x1=dt+100, fillcolor="purple", opacity=0.12,
                                     layer="below", line_width=0)
                    st.plotly_chart(fig_cd, use_container_width=True)
            st.markdown(
                "> **Key insight**: Position forge attacks raise P(Attack) rapidly "
                "due to the KinematicAnomaly=2 observation triggering CPT_K[(1,0)][2]=0.72. "
                "Drop attacks are subtler — variance builds over many ticks, "
                "causing P(Attack) to rise more slowly through the variance channel."
            )


# ══════════════════════════════════════════════════════════════════════ #
#  TAB 3 — Beta Distribution Inspector                                    #
# ══════════════════════════════════════════════════════════════════════ #
with tab3:
    st.subheader("📊 Beta Distribution Trust Inspector")
    st.caption("Jøsang's Beta Reputation System — live PDF per peer.")

    if trust_df.empty:
        st.info("No trust events yet.")
    else:
        # Extract Alpha/Beta series per peer
        ab_rows = []
        for _, row in trust_df.iterrows():
            ab_rows.append({
                "tick":       row.get("tick",  0),
                "peer_id":    row.get("peer_id","?"),
                "alpha":      row.get("alpha",  2.0),
                "beta":       row.get("beta",   1.0),
                "trust":      row.get("new_trust", 0.5),
                "uncertainty":row.get("uncertainty", 0.0),
                "verdict":    row.get("verdict","HONEST"),
                "grudge_tier":row.get("grudge_tier","NORMAL"),
                "min_trust":  row.get("min_trust_ever", 1.0),
                "severity":   row.get("severity", 0.0),
            })
        abdf = pd.DataFrame(ab_rows)

        # ── Beta PDF visualization ───────────────────────────────────── #
        st.markdown("#### Beta Distribution PDF (Epistemic Trust Belief)")
        peers = abdf["peer_id"].dropna().unique().tolist()
        sel   = st.multiselect("Select peers", peers, default=peers[:4])

        x_range = np.linspace(0.001, 0.999, 300)
        fig_pdf = go.Figure()
        for pid in sel:
            sub = abdf[abdf["peer_id"] == pid].sort_values("tick")
            if sub.empty:
                continue
            last = sub.iloc[-1]
            a, b  = max(0.01, last["alpha"]), max(0.01, last["beta"])
            # Beta PDF: x^(a-1) * (1-x)^(b-1) / B(a,b)
            from math import lgamma, exp
            log_beta = lgamma(a) + lgamma(b) - lgamma(a + b)
            pdf = np.array([
                exp((a-1)*math.log(xi) + (b-1)*math.log(1-xi) - log_beta) if 0 < xi < 1 else 0
                for xi in x_range
            ])
            label = (
                f"{pid}  α={a:.2f} β={b:.2f}  "
                f"E[T]={a/(a+b):.3f}  tier={last.get('grudge_tier','?')}"
            )
            fig_pdf.add_trace(go.Scatter(
                x=x_range, y=pdf, mode="lines", name=label, fill="tozeroy"
            ))
        fig_pdf.add_vline(x=trust_threshold, line_dash="dot", line_color="orange",
                          annotation_text="Reroute threshold")
        fig_pdf.add_vline(x=0.5, line_dash="dash", line_color="red",
                          annotation_text="Decision boundary")
        fig_pdf.update_layout(
            template="plotly_dark",
            title="Beta PDF — Epistemic Belief About Each Peer's Trustworthiness",
            xaxis_title="Trust value t ∈ (0,1)",
            yaxis_title="Probability density",
        )
        st.plotly_chart(fig_pdf, use_container_width=True)

        # ── Alpha/Beta parameter evolution ───────────────────────────── #
        st.markdown("#### α / β Parameter Evolution Over Time")
        col_ab_a, col_ab_b = st.columns(2)
        with col_ab_a:
            fig_a = px.line(abdf[abdf["peer_id"].isin(sel)], x="tick", y="alpha",
                            color="peer_id", template="plotly_dark",
                            title="α (evidence of honesty) over time")
            st.plotly_chart(fig_a, use_container_width=True)
        with col_ab_b:
            fig_b = px.line(abdf[abdf["peer_id"].isin(sel)], x="tick", y="beta",
                            color="peer_id", template="plotly_dark",
                            title="β (evidence of deception) over time")
            st.plotly_chart(fig_b, use_container_width=True)

        # ── Grudge tier status ───────────────────────────────────────── #
        st.markdown("#### 🏴 Grudge Memory State")
        tier_order = {"NORMAL": 0, "MODERATE": 1, "SLOW": 2, "FROZEN": 3}
        latest_ab  = abdf.sort_values("tick").groupby("peer_id").last().reset_index()
        tier_color = {"NORMAL": "#27ae60","MODERATE": "#f39c12","SLOW": "#e67e22","FROZEN": "#e74c3c"}
        for _, row in latest_ab.iterrows():
            tier = row.get("grudge_tier","NORMAL")
            col  = tier_color.get(tier, "#aaa")
            st.markdown(
                f"<div style='background:{col}22;border-left:4px solid {col};"
                f"border-radius:6px;padding:8px 12px;margin:4px 0;'>"
                f"<b>{row['peer_id']}</b>  trust={row['trust']:.3f}  "
                f"α={row['alpha']:.2f}  β={row['beta']:.2f}  "
                f"min_ever={row['min_trust']:.3f}  "
                f"<b>Grudge tier: {tier}</b></div>",
                unsafe_allow_html=True
            )

        # ── Asymmetric recovery explanation ─────────────────────────── #
        st.markdown("#### ⚖️ Asymmetric Recovery Visualization")
        st.markdown(
            "The Beta engine applies **fast punishment / slow recovery**. "
            "A FROZEN peer's α is almost never incremented regardless of honest behavior. "
            "This ratchet prevents trust yo-yo effects after attack windows end."
        )
        if not abdf.empty:
            ver_df = abdf.sort_values("tick")
            fig_asym = go.Figure()
            for pid in sel:
                sub = ver_df[ver_df["peer_id"] == pid]
                dn = sub[sub["verdict"] == "ATTACK"]["tick"]
                up = sub[sub["verdict"] == "HONEST"]["tick"]
                fig_asym.add_trace(go.Scatter(
                    x=sub["tick"], y=sub["trust"], mode="lines", name=f"{pid} trust",
                ))
            fig_asym.update_layout(yaxis_range=[0,1], template="plotly_dark",
                                   title="Trust Trajectory — Punishment vs Recovery Rate")
            st.plotly_chart(fig_asym, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════ #
#  TAB 4 — D* Lite Planner                                                #
# ══════════════════════════════════════════════════════════════════════ #
with tab4:
    st.subheader("🗺️ D* Lite Trust-Weighted Planner")
    st.caption(
        "Incremental replanning with trust-modulated hazard costs. "
        "Only changed cells re-expand — O(k) updates, not O(n log n) full A*."
    )

    # ── Hazard field heatmap ─────────────────────────────────────────── #
    if not hazard_df.empty:
        st.markdown("#### Hazard Field Heatmap (Lattest Snapshot)")
        latest_hazard_row = hazard_df.sort_values("tick").iloc[-1]
        hazard_cells = latest_hazard_row.get("hazard", {})
        if not isinstance(hazard_cells, dict):
            try:
                hazard_cells = json.loads(str(hazard_cells))
            except Exception:
                hazard_cells = {}
        if hazard_cells:
            hx, hy, hc = [], [], []
            for cell_str, cost in hazard_cells.items():
                try:
                    cx, cy = cell_str.split(",")
                    hx.append(int(cx)); hy.append(int(cy)); hc.append(float(cost))
                except Exception:
                    pass
            if hx:
                fig_h = go.Figure(go.Scatter(
                    x=hx, y=hy, mode="markers",
                    marker=dict(color=hc, colorscale="Reds", size=6,
                                showscale=True, colorbar=dict(title="Hazard Cost")),
                    text=[f"cost={c:.2f}" for c in hc],
                ))
                fig_h.update_layout(
                    template="plotly_dark",
                    title=(
                        f"Hazard Field at Tick {latest_hazard_row.get('tick',0)} — "
                        f"{len(hazard_cells)} active cells"
                    ),
                    xaxis_title="x (px)", yaxis_title="y (px)",
                    yaxis_autorange="reversed",
                )
                st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("Hazard field is empty (no distrusted peers yet).")
    else:
        st.info("No hazard snapshots logged yet — start simulation.")

    # ── Replan event table ───────────────────────────────────────────── #
    if not replan_df.empty:
        st.markdown("#### Replan Event Log")
        replan_display = replan_df[[
            "tick", "agent_id", "trigger",
            "n_cells_changed", "old_path_cost", "new_path_cost", "cost_delta",
            "path_length", "peers_avoided"
        ]].copy()
        for col in ["old_path_cost","new_path_cost","cost_delta"]:
            if col in replan_display.columns:
                replan_display[col] = replan_display[col].apply(
                    lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
                )
        st.dataframe(replan_display.sort_values("tick", ascending=False).head(100),
                     use_container_width=True, height=300)

        # ── Cost comparison chart ────────────────────────────────────── #
        st.markdown("#### Path Cost: Unconstrained vs Trust-Modulated")
        fig_cost = go.Figure()
        for aid in replan_df["agent_id"].dropna().unique():
            sub_r = replan_df[replan_df["agent_id"] == aid].sort_values("tick")
            if not sub_r.empty:
                fig_cost.add_trace(go.Scatter(
                    x=sub_r["tick"], y=sub_r["old_path_cost"], mode="lines",
                    name=f"{aid} unconstrained (old)", line=dict(dash="dot"),
                ))
                fig_cost.add_trace(go.Scatter(
                    x=sub_r["tick"], y=sub_r["new_path_cost"], mode="lines",
                    name=f"{aid} trust-modulated (new)",
                ))
        fig_cost.update_layout(
            template="plotly_dark",
            title="D* Lite: Path Cost Before and After Trust Replanning",
            xaxis_title="Tick", yaxis_title="Estimated Path Cost",
        )
        fig_cost.add_annotation(
            text=(
                "↑ Gap = detour cost premium paid to avoid distrusted zones.<br>"
                "Larger gap = more distrusted peers in the environment."
            ),
            xref="paper", yref="paper", x=0.01, y=0.95,
            showarrow=False, align="left",
            font=dict(color="#a0c8ff", size=11),
        )
        st.plotly_chart(fig_cost, use_container_width=True)

        # ── Peers avoided summary ────────────────────────────────────── #
        st.markdown("#### Peers Avoided Per Replan")
        avoid_rows = []
        for _, row in replan_df.iterrows():
            pa = row.get("peers_avoided", [])
            if isinstance(pa, list):
                for p in pa:
                    avoid_rows.append({"tick": row.get("tick",0), "avoided": p})
        if avoid_rows:
            avdf = pd.DataFrame(avoid_rows)
            counts = avdf.groupby("avoided").count().reset_index().rename(
                columns={"tick": "times_avoided"})
            fig_av = px.bar(counts, x="avoided", y="times_avoided",
                            title="How Often Each Peer Was Avoided in Replanning",
                            template="plotly_dark", color="avoided")
            st.plotly_chart(fig_av, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════ #
#  TAB 5 — Epistemic Events & SMT                                         #
# ══════════════════════════════════════════════════════════════════════ #
with tab5:
    st.subheader("⚡ Epistemic Events & SMT/ToM Reasoning")
    st.caption(
        "Z3 SMT-backed contradiction proofs + Theory-of-Mind intent inference. "
        "Each paradox is formally proven, not heuristically detected."
    )

    if paradox_df.empty:
        st.info("No paradox events yet.")
    else:
        # ── Paradox summary ──────────────────────────────────────────── #
        c_p1, c_p2, c_p3 = st.columns(3)
        c_p1.metric("Total Paradoxes", len(paradox_df))
        kinematic = len(paradox_df[paradox_df["reason"].str.contains("KINEMATIC", na=False)])
        temporal  = len(paradox_df[paradox_df["reason"].str.contains("TEMPORAL",  na=False)])
        obstacle  = len(paradox_df[paradox_df["reason"].str.contains("OBSTACLE",  na=False)])
        c_p2.metric("Kinematic Bound Violations", kinematic)
        c_p3.metric("Temporal Paradoxes", temporal)

        # ── Reason distribution ──────────────────────────────────────── #
        if "reason" in paradox_df.columns:
            reason_counts = paradox_df["reason"].value_counts().reset_index()
            reason_counts.columns = ["Reason", "Count"]
            col_ra, col_rb = st.columns(2)
            with col_ra:
                fig_r = px.pie(reason_counts, names="Reason", values="Count",
                               title="SMT Verdict Distribution",
                               template="plotly_dark",
                               color_discrete_sequence=px.colors.qualitative.Dark24)
                st.plotly_chart(fig_r, use_container_width=True)
            with col_rb:
                st.markdown("#### Paradox Event Rate Over Time")
                para_rate = paradox_df.groupby(
                    (paradox_df["tick"] // 50) * 50
                ).size().reset_index(name="count")
                para_rate.columns = ["tick_bin", "count"]
                fig_pr = px.bar(para_rate, x="tick_bin", y="count",
                                title="Paradox Events per 50-Tick Window",
                                template="plotly_dark")
                st.plotly_chart(fig_pr, use_container_width=True)

        # ── Full event table ──────────────────────────────────────────── #
        st.markdown("#### 🔬 Paradox Event Detail Table")
        para_cols = ["tick","evaluator","peer_id","delta","reason",
                     "explanation","smt_reason","best_intent","best_intent_p"]
        avail     = [c for c in para_cols if c in paradox_df.columns]
        para_disp = paradox_df[avail].sort_values("tick", ascending=False).head(100)

        def _color_paradox(row):
            r = str(row.get("reason",""))
            if "KINEMATIC" in r:
                return ["background-color: #2d1214"] * len(row)
            elif "TEMPORAL" in r:
                return ["background-color: #12122d"] * len(row)
            return [""] * len(row)

        st.dataframe(para_disp, use_container_width=True, height=350)

        # ── Intent posterior heatmap ──────────────────────────────────── #
        st.markdown("#### Intent Posteriors — What Does Each Peer Want?")
        if "best_intent" in paradox_df.columns and "best_intent_p" in paradox_df.columns:
            pivot_rows = []
            for _, row in paradox_df.dropna(subset=["best_intent"]).iterrows():
                pivot_rows.append({
                    "tick":    row.get("tick",0),
                    "peer":    row.get("peer_id","?"),
                    "goal":    row.get("best_intent","?"),
                    "p_intent": row.get("best_intent_p", 0.0),
                })
            if pivot_rows:
                idf  = pd.DataFrame(pivot_rows)
                fig_intent = px.scatter(idf, x="tick", y="peer", size="p_intent",
                                        color="goal", template="plotly_dark",
                                        title="Inferred Goal Intent Over Time (bubble size = posterior)")
                st.plotly_chart(fig_intent, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════ #
#  TAB 6 — Network & Crypto                                               #
# ══════════════════════════════════════════════════════════════════════ #
with tab6:
    st.subheader("🔐 Network & Crypto — MITM Attack Analysis")

    # ── MITM counters timeline ───────────────────────────────────────── #
    if not attack_df.empty:
        st.markdown("#### MITM Ground-Truth Attack Windows")
        atk_display = attack_df[["tick","attacker_id","target_id","mode"]
                                 ].drop_duplicates().sort_values("tick")
        st.dataframe(atk_display, use_container_width=True, height=200)

        mode_counts = attack_df["mode"].value_counts().reset_index()
        mode_counts.columns = ["Mode", "Events"]
        col_nc_a, col_nc_b = st.columns(2)
        with col_nc_a:
            fig_modes = px.bar(mode_counts, x="Mode", y="Events", color="Mode",
                               template="plotly_dark", title="Attack Mode Distribution")
            st.plotly_chart(fig_modes, use_container_width=True)
        with col_nc_b:
            # Overlay ground truth on trust evolution
            if not trust_df.empty:
                trust_rows_n = []
                for _, row in trust_df.iterrows():
                    trust_rows_n.append({
                        "tick": row.get("tick",0),
                        "peer_id": row.get("peer_id","?"),
                        "trust": row.get("new_trust", 0.5),
                    })
                tdf_n = pd.DataFrame(trust_rows_n)
                fig_ov = px.line(tdf_n, x="tick", y="trust", color="peer_id",
                                 template="plotly_dark",
                                 title="Trust + Attack Overlay")
                # Shade attack windows
                for _, aw in atk_display.iterrows():
                    fig_ov.add_vrect(
                        x0=aw["tick"] - 5, x1=aw["tick"] + 5,
                        fillcolor="red", opacity=0.15, layer="below", line_width=0,
                        annotation_text=f"{aw['mode']}", annotation_position="top left",
                    )
                st.plotly_chart(fig_ov, use_container_width=True)

    # ── Crypto fault log ─────────────────────────────────────────────── #
    st.markdown("#### AES-GCM Crypto Fault Events")
    if not crypto_df.empty:
        st.dataframe(
            crypto_df[["tick","evaluator","peer_id","error"]].sort_values("tick", ascending=False),
            use_container_width=True, height=200,
        )
    else:
        st.info("No crypto fault events — either no insider attack or forge is undetected by tag check.")

    # ── Detection precision explanation ──────────────────────────────── #
    st.markdown("#### How Detection Works (Multi-Layer)")
    st.markdown("""
| Layer | Mechanism | Detects |
|-------|-----------|---------|
| **L1 — Crypto** | AES-256-GCM AEAD tag verification | Forged ciphertext (outsider) |
| **L2 — SMT** | Z3 kinematic bound proof | Position-forge with large offset |
| **L3 — Temporal** | Monotonic tick guard | Replay attacks |
| **L4 — DBN** | 2-slice forward filter | Persistent anomaly pattern |
| **L5 — Trust** | Beta-Binomial grudge engine | Long-term trust erosion |
| **L6 — D\\* Lite** | Hazard-weighted replanning | Avoidance (not detection) |
""")

    # ── MITM attack mode explanations ────────────────────────────────── #
    with st.expander("🔍 Attack Mode Deep Dive"):
        st.markdown("""
**position_forge** (Wave 1 & 4):
- Proxy decrypts using fleet key, adds positional offset (dx, dy), re-encrypts.
- SMT kinematic bound proof catches this within 1-2 ticks if |offset| > v_max × dt + tolerance.
- DBN then sees KA=2 repeatedly → P(Attack) crosses 0.5 → trust punishment.

**replay** (Wave 2 & 5):
- Proxy buffers packets and re-emits them 10-15 ticks later.
- Monotonic tick guard at the receiver rejects any packet where tick ≤ last_seen_tick.
- Shows as TEMPORAL_PARADOX_NON_MONOTONIC in the epistemic events log.

**drop** (Wave 3):
- Proxy silently discards all packets from attacker.
- Hardest to detect — indistinguishable from packet loss in one observation.
- DBN tracks variance over rolling window; paradox frequency triggers PacketLoss=1.
- P(Attack) rises slowly but eventually crosses threshold.
        """)


# ─────────────────────────────────────────────────────────────────────── #
# Auto-refresh                                                             #
# ─────────────────────────────────────────────────────────────────────── #
time.sleep(0.1)
st.rerun()
