import datetime
import html
import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from config import API_URL, PREDICTOR_URL
from shared.schemas import (
    GenericResponse,
    LogRead,
    LogServiceRead,
    MetricHistoryRangeRead,
    MetricHistoryRead,
    MetricRead,
    ModelRead,
    RetrainCommand,
    SettingsRead,
    SettingsUpdate,
)
from shared.utils import sync_http_request
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="LPA Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.title("🤖 LPA Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📈 Metrics", "🗂️ Model Registry", "📤 Upload Model", "⚙️ Settings", "📝 Logs"],
    label_visibility="collapsed",
)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: METRICS
# ═════════════════════════════════════════════════════════════════════════════
if page == "📈 Metrics":
    st.title("📈 Real-time Metrics")
    st.sidebar.markdown("### Chart options")
    auto_refresh_enabled = st.sidebar.toggle("🔄 Auto-refresh", value=True)

    refresh_interval = st.sidebar.slider(
        "Refresh interval (sec)",
        5,
        60,
        15,
        disabled=not auto_refresh_enabled,
    )

    limit = st.sidebar.slider("History points", 20, 300, 100)

    if auto_refresh_enabled:
        st_autorefresh(interval=refresh_interval * 1000, key="metrics_refresh")

    data = sync_http_request(
        "GET",
        f"{API_URL}/metrics/history",
        payload=MetricHistoryRead(limit=limit),
        response_model=list[MetricRead],
    )

    if data:
        raw_dicts = [item.model_dump() for item in data]
        df = pd.DataFrame(raw_dicts)
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts")

        # ── CPU Chart ──
        st.subheader("CPU Utilization [0, 1]")
        fig_cpu = go.Figure()
        fig_cpu.add_trace(
            go.Scatter(
                x=df["ts"],
                y=df["input_cpu"],
                name="Input CPU (Current)",
                line=dict(color="#2196F3", width=3),
                mode="lines+markers",
                marker=dict(size=4),
            )
        )
        fig_cpu.add_trace(
            go.Scatter(
                x=df["ts"],
                y=df["predicted_cpu"],
                name="Predicted CPU (+60s)",
                line=dict(color="#FF5722", width=2, dash="dot"),
                mode="lines+markers",
                marker=dict(size=5, symbol="diamond"),
            )
        )
        df_actual = df.dropna(subset=["actual_cpu"])
        fig_cpu.add_trace(
            go.Scatter(
                x=df_actual["ts"],
                y=df_actual["actual_cpu"],
                name="Actual CPU (Outcome)",
                line=dict(color="#FFD700", width=2, dash="dash"),
                mode="lines+markers",
                marker=dict(size=4, symbol="circle-open"),
            )
        )
        fig_cpu.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Time",
            yaxis_title="CPU utilization [0, 1]",
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cpu, use_container_width=True)
        st.divider()

        # ── RAM Chart ──
        st.subheader("RAM Usage (MB)")
        fig_ram = go.Figure()
        fig_ram.add_trace(
            go.Scatter(
                x=df["ts"],
                y=df["input_ram"],
                name="RAM (Current)",
                line=dict(color="#4CAF50", width=3),
                mode="lines+markers",
                marker=dict(size=4),
            )
        )
        fig_ram.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Time",
            yaxis_title="RAM (MB)",
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ram, use_container_width=True)
        st.divider()

        # ── RPS Chart ──
        st.subheader("Requests Per Second")
        fig_rps = go.Figure()
        fig_rps.add_trace(
            go.Scatter(
                x=df["ts"],
                y=df["input_rps"],
                name="RPS (Current)",
                line=dict(color="#9C27B0", width=3),
                mode="lines+markers",
                marker=dict(size=4),
            )
        )
        fig_rps.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Time",
            yaxis_title="RPS",
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_rps, use_container_width=True)
    else:
        st.info("No metric data available yet.")

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

    models = sync_http_request(
        method="GET", url=f"{API_URL}/models", response_model=list[ModelRead]
    )

    if models:
        raw_dicts = [model.model_dump() for model in models]
        df = pd.DataFrame(raw_dicts)

        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        df["mse"] = df["mse"].apply(lambda x: f"{x:.6f}" if x is not None else "—")
        df["mae"] = df["mae"].apply(lambda x: f"{x:.6f}" if x is not None else "—")
        df["status"] = df["is_active"].apply(
            lambda x: "✅ Active" if x else "⬜ Inactive"
        )

        display_cols = [
            "version",
            "status",
            "mse",
            "mae",
            "window_size",
            "forecast_horizon",
            "created_at",
        ]
        st.dataframe(
            df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "version": st.column_config.TextColumn("Version", width="medium"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "mse": st.column_config.TextColumn("MSE"),
                "mae": st.column_config.TextColumn("MAE"),
                "window_size": st.column_config.NumberColumn("Window"),
                "forecast_horizon": st.column_config.NumberColumn("Horizon"),
                "created_at": st.column_config.TextColumn("Created At"),
            },
        )

        st.divider()

        active_models = [m for m in models if m.is_active]
        if active_models:
            active = active_models[0]
            st.subheader("🟢 Currently Active Model")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Version", active.version)
            col2.metric("MSE", f"{active.mse:.6f}" if active.mse is not None else "—")
            col3.metric("MAE", f"{active.mae:.6f}" if active.mae is not None else "—")
            col4.metric("Window", active.window_size)
            col5.metric("Horizon", active.forecast_horizon)
            with st.expander("File paths"):
                st.code(
                    f"Model:    {active.model_path}\nScaler X: {active.scaler_x_path}"
                )

        st.divider()
        st.subheader("🔄 Activate / Reload Model")

        all_versions = [m.version for m in models]

        if all_versions:
            active_index = 0
            for i, m in enumerate(models):
                if m.is_active:
                    active_index = i
                    break

            selected = st.selectbox(
                "Select version to activate or force-reload",
                all_versions,
                index=active_index,
                help="Selecting the currently active model forces the Predictor to reload its files.",
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button(f"✅ Activate / Reload **{selected}**", type="primary"):
                    with st.spinner(f"Sending reload signal for {selected}..."):
                        result = sync_http_request(
                            method="PUT",
                            url=f"{API_URL}/models/{selected}/activate",
                            response_model=GenericResponse,
                        )
                    if result:
                        st.success(result.message)
                        time.sleep(1)
                        st.rerun()
            with col2:
                if st.button("📊 Evaluate Performance", use_container_width=True):
                    with st.spinner(f"Calculating real MSE/MAE for {selected}..."):
                        eval_result = sync_http_request(
                            method="POST",
                            url=f"{API_URL}/models/{selected}/evaluate",
                            response_model=ModelRead,
                        )
                    if eval_result:
                        st.success(
                            f"Metrics updated! "
                            f"MSE: {eval_result.mse:.4f} | MAE: {eval_result.mae:.4f}"
                        )
                        time.sleep(2)
                        st.rerun()
        else:
            st.info("No models registered yet.")

        st.divider()
        st.subheader("🛠️ Fine-Tune a Model")
        st.markdown("Run background fine-tuning on historical data.")

        col_ft1, col_ft2 = st.columns(2)
        with col_ft1:
            tune_version = st.selectbox(
                "Base Model Version",
                all_versions,
                help="Weights from this model are used as the starting point.",
            )

        with col_ft2:
            epochs = st.number_input(
                "Epochs", min_value=1, max_value=100, value=5, step=1
            )
            batch_size = st.number_input(
                "Batch Size", min_value=1, max_value=256, value=16, step=1
            )

        st.write("📅 **Select Data Range**")

        if "ft_defaults_set" not in st.session_state:
            now = datetime.datetime.now()
            yesterday = now - datetime.timedelta(days=1)

            st.session_state.ft_start_date = yesterday.date()
            st.session_state.ft_start_time = yesterday.time()
            st.session_state.ft_end_date = now.date()
            st.session_state.ft_end_time = now.time()
            st.session_state.ft_defaults_set = True

        col_d1, col_d2, col_d3, col_d4 = st.columns(4)

        with col_d1:
            start_date = st.date_input(
                "Start Date",
                value=st.session_state.ft_start_date,
                key="input_start_date",
            )
        with col_d2:
            start_time = st.time_input(
                "Start Time",
                value=st.session_state.ft_start_time,
                key="input_start_time",
            )
        with col_d3:
            end_date = st.date_input(
                "End Date", value=st.session_state.ft_end_date, key="input_end_date"
            )
        with col_d4:
            end_time = st.time_input(
                "End Time", value=st.session_state.ft_end_time, key="input_end_time"
            )

        # ── Preview ──────────────────────────────────────────────────────────
        start_dt_preview = datetime.datetime.combine(start_date, start_time)
        end_dt_preview = datetime.datetime.combine(end_date, end_time)

        preview_data = sync_http_request(
            method="GET",
            url=f"{API_URL}/metrics/history/range",
            payload=MetricHistoryRangeRead(
                start_time=start_dt_preview,
                end_time=end_dt_preview,
            ),
            response_model=list[MetricRead],
        )

        if preview_data:
            pv_df = pd.DataFrame([r.model_dump() for r in preview_data])
            pv_df["ts"] = pd.to_datetime(pv_df["ts"])
            pv_df = pv_df.sort_values("ts")

            st.caption(f"📊 Preview: **{len(pv_df)}** points in selected range")

            fig_cpu_ft = go.Figure()
            fig_cpu_ft.add_trace(
                go.Scatter(
                    x=pv_df["ts"],
                    y=pv_df["input_cpu"],
                    name="CPU util [0–1]",
                    line=dict(color="#2196F3", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(33,150,243,0.1)",
                )
            )
            fig_cpu_ft.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(title="CPU util", range=[0, 1]),
                xaxis=dict(showticklabels=False),
                hovermode="x unified",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cpu_ft, use_container_width=True)

            fig_rps_ft = go.Figure()
            fig_rps_ft.add_trace(
                go.Scatter(
                    x=pv_df["ts"],
                    y=pv_df["input_rps"],
                    name="RPS",
                    line=dict(color="#9C27B0", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(156,39,176,0.1)",
                )
            )
            fig_rps_ft.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=5, b=0),
                yaxis=dict(title="RPS"),
                hovermode="x unified",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_rps_ft, use_container_width=True)
        elif preview_data is not None:
            st.info("No closed predictions in the selected range.")

        # ─────────────────────────────────────────────────────────────────────
        if st.button("🚀 Start Fine-Tuning", type="secondary"):
            start_dt = datetime.datetime.combine(start_date, start_time)
            end_dt = datetime.datetime.combine(end_date, end_time)

            payload = {
                "target_version": tune_version,
                "start_time": start_dt.isoformat(),
                "end_time": end_dt.isoformat(),
                "epochs": epochs,
                "batch_size": batch_size,
            }

            with st.spinner(f"Initiating fine-tuning for {tune_version}..."):
                try:
                    response = sync_http_request(
                        method="POST",
                        url=f"{PREDICTOR_URL}/retrain",
                        payload=RetrainCommand.model_validate(payload),
                        response_model=GenericResponse,
                    )
                    if response:
                        st.success(
                            "Fine-tuning started in the background. "
                            "The new model will appear in the registry when done (refresh in a few minutes)."
                        )
                    else:
                        st.error("Failed to start fine-tuning.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach the predictor service: {e}")

    elif models is not None:
        st.info("No models registered yet. Upload one using **Upload Model**.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD MODEL
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📤 Upload Model":
    st.title("📤 Upload New Model")
    st.markdown(
        "Upload a trained `.keras` GRU model and its Scaler X `.joblib` file. "
        "Leave *Version* empty to auto-generate one."
    )

    with st.form("upload_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        version = col1.text_input("Version", placeholder="auto-generated if empty")
        mse = col2.number_input("MSE", value=0.0, min_value=0.0, format="%.6f")
        mae = col3.number_input("MAE", value=0.0, min_value=0.0, format="%.6f")

        col_ws, col_fh = st.columns(2)
        with col_ws:
            window_size = st.number_input(
                "Window Size (steps)",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                help="Number of historical time steps the model uses as input.",
            )
        with col_fh:
            forecast_horizon = st.number_input(
                "Forecast Horizon (steps)",
                min_value=1,
                max_value=60,
                value=12,
                step=1,
                help="How many steps ahead the model predicts (12 steps × 5s = 60s).",
            )

        model_file = st.file_uploader(
            "Model file (.keras)",
            type=["keras"],
            help="Keras model exported with model.save()",
        )

        scaler_X_file = st.file_uploader(
            "Scaler X (Features) (.joblib)",
            type=["joblib", "pkl"],
            help="StandardScaler for input features",
        )

        submitted = st.form_submit_button(
            "📤 Upload", type="primary", use_container_width=True
        )

    if submitted:
        if not model_file or not scaler_X_file:
            st.error("Both model file and Scaler X are required.")
        else:
            with st.spinner("Uploading model and scaler..."):
                files = {
                    "model_file": (
                        model_file.name,
                        model_file.getvalue(),
                        "application/octet-stream",
                    ),
                    "scaler_x_file": (
                        scaler_X_file.name,
                        scaler_X_file.getvalue(),
                        "application/octet-stream",
                    ),
                }
                form_data: dict[str, str] = {
                    "mse": str(mse),
                    "mae": str(mae),
                    "window_size": str(int(window_size)),
                    "forecast_horizon": str(int(forecast_horizon)),
                }
                stripped_version = version.strip()
                if stripped_version:
                    form_data["version"] = stripped_version

                result = sync_http_request(
                    url=f"{API_URL}/models/upload",
                    method="POST",
                    data=form_data,
                    files=files,
                    response_model=ModelRead,
                )
            if result:
                st.success(f"✅ Model **{result.version}** uploaded successfully!")
                st.json(result.model_dump())

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.title("⚙️ System Settings")

    settings_data = sync_http_request(
        method="GET", url=f"{API_URL}/settings", response_model=SettingsRead
    )

    if settings_data:
        with st.form("settings_form"):
            st.subheader("Collector")
            is_active = st.checkbox(
                "Collector Active",
                value=settings_data.is_collector_active,
                help="Enable or disable the metric collection loop",
            )

            st.subheader("Prometheus")
            prom_url = st.text_input(
                "Prometheus URL",
                value=settings_data.prometheus_url,
                help="Full URL to Prometheus query API, e.g. http://host:9090/api/v1/query",
            )

            st.subheader("PromQL Queries")
            cpu_query = st.text_area(
                "CPU Query",
                value=settings_data.cpu_query,
                height=90,
            )
            ram_query = st.text_area(
                "RAM Query",
                value=settings_data.ram_query,
                height=90,
            )
            rps_query = st.text_area(
                "RPS Query",
                value=settings_data.rps_query,
                height=90,
            )

            st.subheader("OOD Blend")
            ood_blend_threshold = st.number_input(
                "OOD Blend Threshold [0, 1]",
                min_value=0.0,
                max_value=1.0,
                value=float(settings_data.ood_blend_threshold),
                step=0.01,
                format="%.3f",
                help=(
                    "Below this CPU utilization the model is outside its training distribution. "
                    "The exposed prediction is linearly blended toward actual_cpu to avoid "
                    "idle over-provisioning."
                ),
            )

            saved = st.form_submit_button(
                "💾 Save Settings", type="primary", use_container_width=True
            )

        if saved:
            payload = {
                "is_collector_active": is_active,
                "prometheus_url": prom_url,
                "cpu_query": cpu_query,
                "ram_query": ram_query,
                "rps_query": rps_query,
                "ood_blend_threshold": ood_blend_threshold,
            }
            result = sync_http_request(
                url=f"{API_URL}/settings",
                method="PUT",
                payload=SettingsUpdate.model_validate(payload),
                response_model=SettingsRead,
            )
            if result:
                st.success("Settings saved.")
                st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: LOGS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📝 Logs":
    st.subheader("📋 System Logs & Events")

    if "log_limit" not in st.session_state:
        st.session_state.log_limit = 100
    if "log_service" not in st.session_state:
        st.session_state.log_service = "All"

    col1, col2, col3, col4 = st.columns([1.5, 2, 1.5, 1.5])

    with col1:
        log_limit = st.number_input(
            "Limit", min_value=10, max_value=5000, step=50, key="log_limit"
        )

    with col3:
        st.write("")
        st.write("")
        auto_refresh = st.toggle(
            "🔄 Auto-Refresh (5s)", value=True, key="log_auto_refresh"
        )

    with col4:
        st.write("")
        st.write("")
        if st.button("🔄 Manual Refresh", use_container_width=True):
            st.rerun()

    if auto_refresh:
        st_autorefresh(interval=5000, limit=None, key="logs_autorefresh")

    db_services = sync_http_request(
        method="GET", url=f"{API_URL}/logs/services", response_model=list[str]
    )
    if not db_services:
        db_services = []

    unique_services = ["All"] + sorted(db_services)

    if st.session_state.log_service not in unique_services:
        st.session_state.log_service = "All"

    current_index = unique_services.index(st.session_state.log_service)

    with col2:
        selected_service = st.selectbox(
            "Service Filter", unique_services, index=current_index
        )
        st.session_state.log_service = selected_service

    selected_service = selected_service if selected_service != "All" else None
    logs = sync_http_request(
        method="GET",
        url=f"{API_URL}/logs",
        response_model=list[LogRead],
        payload=LogServiceRead(service=selected_service, limit=log_limit),
    )
    if logs:
        raw_dicts = [log.model_dump() for log in logs]
        df_logs = pd.DataFrame(raw_dicts)
        df_logs["ts"] = pd.to_datetime(df_logs["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        st.divider()

        if not df_logs.empty:
            log_html = """
            <div style='
                background-color: #1e1e1e;
                padding: 15px;
                border-radius: 8px;
                font-family: "Courier New", Courier, monospace;
                font-size: 14px;
                height: 600px;
                overflow-y: auto;
                color: #cccccc;
                line-height: 1.5;
                box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
            '>
            """

            for _, row in df_logs.iterrows():
                ts = row["ts"]
                lvl = row["level"]
                svc = row["service"]
                msg = row["message"]

                if lvl == "ERROR":
                    color = "#ff5252"
                    lvl_pad = lvl
                elif lvl == "WARNING":
                    color = "#ffb142"
                    lvl_pad = lvl
                else:
                    color = "#33d9b2"
                    lvl_pad = lvl + " "

                log_html += "<div style='margin-bottom: 4px; border-bottom: 1px solid #333; padding-bottom: 2px;'>"
                log_html += f"<span style='color: #7f8c8d;'>[{ts}]</span> "
                log_html += f"<span style='color: {color}; font-weight: bold;'>[{lvl_pad}]</span> "
                log_html += f"<span style='color: #34ace0;'>[{svc}]</span> "
                log_html += f"<span style='color: #f1f2f6;'>{html.escape(msg)}</span>"
                log_html += "</div>"

            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)
        else:
            st.info(f"No logs for service '{selected_service}'.")
    else:
        st.info("No system logs yet.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption(f"API: `{API_URL}`")
