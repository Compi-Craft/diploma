import os
import time
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

API_URL = os.getenv("API_URL", "http://timescale_api:5000")

st.set_page_config(
    page_title="LPA Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def api_get(path: str, params: Optional[dict] = None) -> Any:
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_URL}")
        return None
    except Exception as e:
        st.error(f"❌ API Error: {e}")
        return None


def api_put(path: str, json: Optional[dict] = None) -> Any:
    try:
        r = requests.put(f"{API_URL}{path}", json=json, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ API Error: {e}")
        return None


def api_post(
    path: str,
    json: Optional[dict] = None,
    files: Optional[dict] = None,
    data: Optional[dict] = None,
) -> Any:
    try:
        r = requests.post(
            f"{API_URL}{path}", json=json, files=files, data=data, timeout=15
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ API Error: {e}")
        return None


# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.title("🤖 LPA Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📈 Metrics", "🗂️ Model Registry", "📤 Upload Model", "⚙️ Settings"],
    label_visibility="collapsed",
)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: METRICS
# ═════════════════════════════════════════════════════════════════════════════
if page == "📈 Metrics":
    st.title("📈 Real-time Metrics")

    st.sidebar.markdown("### Chart options")
    refresh_interval = st.sidebar.slider("Auto-refresh (sec)", 5, 60, 15)
    limit = st.sidebar.slider("History points", 20, 300, 100)

    st_autorefresh(interval=refresh_interval * 1000, key="metrics_refresh")

    RESOURCES: dict[str, tuple[str, str, str]] = {
        "cpu": ("CPU Usage (cores)", "#2196F3", "#FF5722"),
        "ram": ("RAM Usage (MB)", "#4CAF50", "#FF9800"),
        "rps": ("Requests Per Second", "#9C27B0", "#F44336"),
    }

    for resource, (label, actual_color, pred_color) in RESOURCES.items():
        data = api_get("/metrics/history", {"resource": resource, "limit": limit})

        st.subheader(label)
        if data:
            df = pd.DataFrame(data)
            df["ts"] = pd.to_datetime(df["ts"])
            df["target_ts"] = pd.to_datetime(df["target_ts"])
            df = df.sort_values("ts")

            fig = go.Figure()

            # Actual measured values
            fig.add_trace(
                go.Scatter(
                    x=df["ts"],
                    y=df["input_value"],
                    name="Actual",
                    line=dict(color=actual_color, width=2),
                    mode="lines+markers",
                    marker=dict(size=4),
                )
            )

            # Predicted values (plotted at measurement time for alignment)
            fig.add_trace(
                go.Scatter(
                    x=df["ts"],
                    y=df["predicted_value"],
                    name="Predicted (next interval)",
                    line=dict(color=pred_color, width=2, dash="dot"),
                    mode="lines+markers",
                    marker=dict(size=4, symbol="diamond"),
                )
            )

            # Retrospective actual values (filled in when target_ts is reached)
            has_actual = df["actual_value"].notna()
            if has_actual.any():
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[has_actual, "target_ts"],
                        y=df.loc[has_actual, "actual_value"],
                        name="Actual @ Target Time",
                        line=dict(color=actual_color, width=1, dash="dash"),
                        marker=dict(size=3),
                        opacity=0.6,
                    )
                )

            fig.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                xaxis_title="Time",
                yaxis_title=label,
                hovermode="x unified",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No data available for **{resource}** yet.")

        st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL REGISTRY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗂️ Model Registry":
    st.title("🗂️ Model Registry")

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh"):
            st.rerun()

    models = api_get("/models")

    if models:
        df = pd.DataFrame(models)

        # Format columns for display
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        df["mse"] = df["mse"].apply(lambda x: f"{x:.6f}" if x is not None else "—")
        df["mae"] = df["mae"].apply(lambda x: f"{x:.6f}" if x is not None else "—")
        df["status"] = df["is_active"].apply(lambda x: "✅ Active" if x else "⬜ Inactive")

        display_cols = ["version", "status", "mse", "mae", "created_at"]
        st.dataframe(
            df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "version": st.column_config.TextColumn("Version", width="medium"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "mse": st.column_config.TextColumn("MSE"),
                "mae": st.column_config.TextColumn("MAE"),
                "created_at": st.column_config.TextColumn("Created At"),
            },
        )

        st.divider()

        # Active model info
        active_models = [m for m in models if m["is_active"]]
        if active_models:
            active = active_models[0]
            st.subheader("🟢 Currently Active Model")
            col1, col2, col3 = st.columns(3)
            col1.metric("Version", active["version"])
            col2.metric("MSE", f"{active['mse']:.6f}" if active.get("mse") is not None else "—")
            col3.metric("MAE", f"{active['mae']:.6f}" if active.get("mae") is not None else "—")
            with st.expander("File paths"):
                st.code(f"Model:  {active['model_path']}\nScaler: {active['scaler_path']}")

        # Activate a different model
        st.divider()
        st.subheader("🔄 Activate Model")

        all_versions = [m["version"] for m in models]
        inactive_versions = [m["version"] for m in models if not m["is_active"]]

        if inactive_versions:
            selected = st.selectbox(
                "Select version to activate",
                inactive_versions,
                help="Only inactive models are listed here",
            )
            if st.button(f"✅ Activate **{selected}**", type="primary"):
                with st.spinner(f"Activating {selected}..."):
                    result = api_put(f"/models/{selected}/activate")
                if result:
                    st.success(result.get("message", f"Model {selected} activated."))
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("ℹ️ There is only one model registered, or all models are active.")

    elif models is not None:
        st.info("No models registered yet. Upload one using **Upload Model**.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD MODEL
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📤 Upload Model":
    st.title("📤 Upload New Model")
    st.markdown(
        "Upload a trained `.h5` Keras model and its `.pkl` scikit-learn scaler. "
        "Leave *Version* empty to auto-generate one."
    )

    with st.form("upload_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        version = col1.text_input("Version", placeholder="auto-generated if empty")
        mse = col2.number_input("MSE", value=0.0, min_value=0.0, format="%.6f")
        mae = col3.number_input("MAE", value=0.0, min_value=0.0, format="%.6f")

        model_file = st.file_uploader(
            "Model file (.h5)",
            type=["h5"],
            help="Keras model exported with model.save()",
        )
        scaler_file = st.file_uploader(
            "Scaler file (.pkl)",
            type=["pkl"],
            help="scikit-learn scaler exported with joblib.dump()",
        )

        submitted = st.form_submit_button("📤 Upload", type="primary", use_container_width=True)

    if submitted:
        if not model_file or not scaler_file:
            st.error("Both model (.h5) and scaler (.pkl) files are required.")
        else:
            with st.spinner("Uploading model..."):
                files = {
                    "model_file": (
                        model_file.name,
                        model_file.getvalue(),
                        "application/octet-stream",
                    ),
                    "scaler_file": (
                        scaler_file.name,
                        scaler_file.getvalue(),
                        "application/octet-stream",
                    ),
                }
                form_data: dict[str, str] = {"mse": str(mse), "mae": str(mae)}
                stripped_version = version.strip()
                if stripped_version:
                    form_data["version"] = stripped_version

                result = api_post("/models/upload", files=files, data=form_data)

            if result:
                st.success(f"✅ Model **{result['version']}** uploaded successfully!")
                st.json(result)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.title("⚙️ System Settings")

    settings_data = api_get("/settings")

    if settings_data:
        with st.form("settings_form"):
            st.subheader("Collector")
            is_active = st.checkbox(
                "Collector Active",
                value=settings_data.get("is_collector_active", True),
                help="Enable or disable the metric collection loop",
            )

            st.subheader("Prometheus")
            prom_url = st.text_input(
                "Prometheus URL",
                value=settings_data.get("prometheus_url", ""),
                help="Full URL to Prometheus query API, e.g. http://host:9090/api/v1/query",
            )

            st.subheader("PromQL Queries")
            cpu_query = st.text_area(
                "CPU Query",
                value=settings_data.get("cpu_query", ""),
                height=90,
            )
            ram_query = st.text_area(
                "RAM Query",
                value=settings_data.get("ram_query", ""),
                height=90,
            )
            rps_query = st.text_area(
                "RPS Query",
                value=settings_data.get("rps_query", ""),
                height=90,
            )

            saved = st.form_submit_button("💾 Save Settings", type="primary", use_container_width=True)

        if saved:
            payload = {
                "is_collector_active": is_active,
                "prometheus_url": prom_url,
                "cpu_query": cpu_query,
                "ram_query": ram_query,
                "rps_query": rps_query,
            }
            result = api_put("/settings", json=payload)
            if result:
                st.success("✅ Settings saved successfully!")
                st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption(f"API: `{API_URL}`")
