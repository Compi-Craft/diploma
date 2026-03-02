import os
import time
from typing import Any, Optional
import datetime
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

API_URL = os.getenv("API_URL", "http://timescale_api:5000")
PREDICTOR_URL = os.getenv("PREDICTOR_URL", "http://lstm-predictor:6000")
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
    ["📈 Metrics", "🗂️ Model Registry", "📤 Upload Model", "⚙️ Settings", "📝 Logs"],
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
        st.subheader("🔄 Activate / Reload Model")

        # Беремо абсолютно всі версії, незалежно від їхнього статусу
        all_versions = [m["version"] for m in models]

        if all_versions:
            # Знаходимо індекс поточної активної моделі, щоб вона була вибрана за замовчуванням
            active_index = 0
            for i, m in enumerate(models):
                if m["is_active"]:
                    active_index = i
                    break

            selected = st.selectbox(
                "Select version to activate or force-reload",
                all_versions,
                index=active_index,
                help="Select any model to activate it. Selecting the currently active model will force the Predictor to reload its files."
            )
            
            if st.button(f"✅ Activate / Reload **{selected}**", type="primary"):
                with st.spinner(f"Sending reload signal for {selected}..."):
                    # Звертаємося до нашого API
                    result = api_put(f"/models/{selected}/activate")
                    
                if result:
                    st.success(result.get("message", f"Model {selected} activated."))
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("ℹ️ No models registered yet.")

        st.divider()
        st.subheader("🛠️ Fine-Tune a Model")
        st.markdown("Запусти фоновий процес донавчання моделі на історичних даних.")

        col_ft1, col_ft2 = st.columns(2)
        with col_ft1:
            tune_version = st.selectbox(
                "Base Model Version",
                all_versions, # Тут можна вибирати будь-яку модель, навіть активну
                help="Обери модель, ваги якої будуть використані як базові."
            )

        with col_ft2:
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=5, step=1)
            batch_size = st.number_input("Batch Size", min_value=1, max_value=256, value=16, step=1)

        st.write("📅 **Select Data Range**")
        
        # Дефолтні значення: від вчорашнього дня до зараз
        now = datetime.datetime.now()
        yesterday = now - datetime.timedelta(days=1)

        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            start_date = st.date_input("Start Date", value=yesterday)
        with col_d2:
            start_time = st.time_input("Start Time", value=yesterday.time())
        with col_d3:
            end_date = st.date_input("End Date", value=now)
        with col_d4:
            end_time = st.time_input("End Time", value=now.time())

        if st.button("🚀 Start Fine-Tuning", type="secondary"):
            # Збираємо дату та час у ISO-формат (наприклад, 2026-03-01T10:00:00Z)
            start_dt = datetime.datetime.combine(start_date, start_time).isoformat() + "Z"
            end_dt = datetime.datetime.combine(end_date, end_time).isoformat() + "Z"

            payload = {
                "target_version": tune_version,
                "start_time": start_dt,
                "end_time": end_dt,
                "epochs": epochs,
                "batch_size": batch_size
            }

            with st.spinner(f"Initiating fine-tuning for {tune_version}..."):
                try:
                    # УВАГА: Streamlit має стукати безпосередньо у контейнер Предиктора.
                    # Перевір, чи правильний тут URL для твоєї Docker-мережі.
                    response = requests.post(f"{PREDICTOR_URL}/retrain", json=payload)
                    
                    if response.status_code == 200:
                        st.success("✅ Процес донавчання запущено у фоні! Нова модель з'явиться в таблиці після завершення (натисни Refresh за кілька хвилин).")
                    else:
                        st.error(f"❌ Помилка: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Не вдалося з'єднатися з сервісом Предиктора: {e}")

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
elif page == "📝 Logs":
    st.subheader("📋 System Logs & Events")

    if "log_limit" not in st.session_state:
        st.session_state.log_limit = 100
    if "log_service" not in st.session_state:
        st.session_state.log_service = "All"

    # ==========================================
    # ПАНЕЛЬ КЕРУВАННЯ
    # ==========================================
    col1, col2, col3, col4 = st.columns([1.5, 2, 1.5, 1.5])
    
    with col1:
        log_limit = st.number_input("Limit", min_value=10, max_value=5000, step=50, key="log_limit")
        
    with col3:
        st.write("") 
        st.write("")
        auto_refresh = st.toggle("🔄 Auto-Refresh (5s)", value=True, key="log_auto_refresh")
        
    with col4:
        st.write("") 
        st.write("")
        if st.button("🔄 Manual Refresh", use_container_width=True):
            st.rerun()

    if auto_refresh:
        st_autorefresh(interval=5000, limit=None, key="logs_autorefresh")

    # ==========================================
    # ОТРИМАННЯ СЕРВІСІВ ДЛЯ ФІЛЬТРУ
    # ==========================================
    # Робимо запит до нового ендпоінту
    db_services = api_get("/logs/services")
    if not db_services:
        db_services = []
        
    unique_services = ["All"] + sorted(db_services)
    
    if st.session_state.log_service not in unique_services:
        st.session_state.log_service = "All"
        
    current_index = unique_services.index(st.session_state.log_service)
    
    with col2:
        selected_service = st.selectbox("Service Filter", unique_services, index=current_index)
        st.session_state.log_service = selected_service

    # ==========================================
    # ОТРИМАННЯ ТА ФІЛЬТРАЦІЯ ЛОГІВ
    # ==========================================
    # Тепер, якщо вибрано конкретний сервіс, ми можемо передати його прямо в API запит
    # (Це ще більше оптимізує роботу, бо база не буде тягнути зайві логи)
    api_url = f"/logs?limit={log_limit}"
    if selected_service != "All":
        api_url += f"&service={selected_service}"
        
    logs = api_get(api_url)

    if logs:
        df_logs = pd.DataFrame(logs)
        df_logs["ts"] = pd.to_datetime(df_logs["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        
        st.divider()

        # ==========================================
        # ВІЗУАЛІЗАЦІЯ В СТИЛІ GRAFANA
        # ==========================================
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
                    
                log_html += f"<div style='margin-bottom: 4px; border-bottom: 1px solid #333; padding-bottom: 2px;'>"
                log_html += f"<span style='color: #7f8c8d;'>[{ts}]</span> "
                log_html += f"<span style='color: {color}; font-weight: bold;'>[{lvl_pad}]</span> "
                log_html += f"<span style='color: #34ace0;'>[{svc}]</span> "
                log_html += f"<span style='color: #f1f2f6;'>{msg}</span>"
                log_html += "</div>"
                
            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)
        else:
            st.info(f"ℹ️ Немає логів для сервісу '{selected_service}'.")
    else:
        st.info("ℹ️ Системних логів поки немає.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption(f"API: `{API_URL}`")
