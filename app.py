"""
Industrial Safety Monitoring System — Main Streamlit Application.

Real-time PPE compliance monitoring powered by computer vision.
Detects workers, identifies safety gear, tracks violations,
and displays live analytics.

Run with:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import time
import os
import sys

# Page configuration — MUST be the first Streamlit command
st.set_page_config(
    page_title="Industrial Safety Monitoring System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from models.person_detector.person_detector import PersonDetector
from models.ppe_detector.ppe_detector import PPEDetector
from models.tracking.centroid_tracker import CentroidTracker
from inference.predictor import SafetyPredictor
from inference.webcam_inference import WebcamProcessor
from inference.video_inference import VideoProcessor
from database.database import DatabaseManager
from alerts.alert_manager import AlertManager
from analytics.dashboard import (
    render_realtime_metrics, render_violation_chart,
    render_compliance_timeline, render_worker_stats_table,
    render_violation_heatmap, render_active_alerts,
    render_violation_log
)
from analytics.statistics import SafetyStatistics
from analytics.reports import ReportGenerator
from utils.logger import setup_logger
from utils.helpers import ensure_directories

# Initialize logger
logger = setup_logger()

# ─────────────────────────────────────────────────────────────
# Custom CSS for premium dark-themed UI
# ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Metric card styling */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 8px;
        padding: 8px 16px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
    }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #1a1a2e 100%);
    }

    /* Header gradient */
    .header-gradient {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2em;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: #8892b0;
        font-size: 1em;
        margin-top: 0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #667eea; border-radius: 3px; }

    /* Divider */
    hr { border: 1px solid rgba(102, 126, 234, 0.2); }
</style>
"""


@st.cache_resource
def init_system():
    """Initialize all system components (cached — runs once)."""
    ensure_directories()
    config = Config.get_instance()

    db_manager = DatabaseManager(
        mongo_uri=config.database.mongo_uri or None,
        db_name=config.database.db_name
    )
    person_detector = PersonDetector(
        confidence_threshold=config.detection.confidence_threshold
    )
    ppe_detector = PPEDetector(
        required_ppe=config.ppe.required_items
    )
    tracker = CentroidTracker(
        max_disappeared=config.tracker.max_disappeared,
        max_distance=config.tracker.max_distance
    )
    alert_manager = AlertManager(config)
    predictor = SafetyPredictor(
        person_detector=person_detector,
        ppe_detector=ppe_detector,
        tracker=tracker,
        db_manager=db_manager,
        alert_manager=alert_manager,
        config=config
    )
    statistics = SafetyStatistics(db_manager)

    logger.info(
        f"System initialized — Detection backend: "
        f"{person_detector.backend}"
    )

    return {
        'config': config,
        'db_manager': db_manager,
        'predictor': predictor,
        'alert_manager': alert_manager,
        'statistics': statistics,
        'person_detector': person_detector,
        'ppe_detector': ppe_detector,
        'tracker': tracker,
    }


def render_sidebar(config):
    """Render sidebar controls."""
    with st.sidebar:
        st.markdown("### 🎥 Video Source")
        source_type = st.radio(
            "Select source",
            ["Webcam", "Upload Video", "Demo Mode"],
            key="source_type",
            label_visibility="collapsed"
        )

        if source_type == "Webcam":
            camera_id = st.number_input("Camera ID", 0, 10, 0)
            st.session_state.video_source = int(camera_id)
        elif source_type == "Upload Video":
            uploaded = st.file_uploader(
                "Upload video",
                type=['mp4', 'avi', 'mov', 'mkv']
            )
            if uploaded:
                os.makedirs('data/raw', exist_ok=True)
                video_path = os.path.join('data', 'raw', uploaded.name)
                with open(video_path, 'wb') as f:
                    f.write(uploaded.read())
                st.session_state.video_source = video_path
                st.success(f"Video saved: {uploaded.name}")
        else:
            st.session_state.video_source = 'demo'

        st.markdown("---")
        st.markdown("### ⚙️ PPE Requirements")
        ppe_items = [
            'helmet', 'vest', 'gloves', 'shoes',
            'mask', 'goggles', 'ear_protection'
        ]
        required = []
        for item in ppe_items:
            default = item in ['helmet', 'vest']
            if st.checkbox(
                item.replace('_', ' ').title(),
                value=default,
                key=f"ppe_{item}"
            ):
                required.append(item)
        st.session_state.required_ppe = required

        st.markdown("---")
        st.markdown("### 🔔 Alert Settings")
        st.session_state.sound_enabled = st.checkbox(
            "🔊 Sound Alerts", True
        )
        st.session_state.email_enabled = st.checkbox(
            "📧 Email Alerts", False
        )
        st.session_state.sms_enabled = st.checkbox(
            "📱 SMS Alerts", False
        )

        st.markdown("---")
        st.markdown("### 📊 Detection Settings")
        st.session_state.confidence = st.slider(
            "Confidence Threshold", 0.1, 1.0, 0.5, 0.05
        )
        st.session_state.cooldown = st.slider(
            "Alert Cooldown (sec)", 5, 120, 30, 5
        )

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; color:#555; font-size:0.75em'>"
            "Industrial Safety Monitor v1.0<br>"
            "Powered by SSD MobileNet V2"
            "</div>",
            unsafe_allow_html=True
        )

        return source_type


def create_demo_frame(frame_num: int = 0) -> tuple:
    """Create a demo visualization frame when no camera is available.

    Returns:
        Tuple of (frame, demo_stats).
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Background gradient
    for y in range(480):
        val = int(20 + y * 0.03)
        frame[y, :] = (val, val, val + 5)

    # Title
    cv2.putText(
        frame, 'DEMO MODE', (200, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (102, 126, 234), 2
    )
    cv2.putText(
        frame, 'No Camera Connected', (170, 85),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (136, 146, 176), 1
    )

    # Simulated worker boxes
    workers = [
        (80, 120, 200, 400, True, "Worker #0"),
        (260, 100, 380, 420, False, "Worker #1"),
        (450, 130, 570, 410, True, "Worker #2"),
    ]

    for (x1, y1, x2, y2, compliant, label) in workers:
        color = (118, 230, 0) if compliant else (68, 23, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

        if compliant:
            cv2.putText(
                frame, '[+] Helmet', (x2 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (118, 230, 0), 1
            )
            cv2.putText(
                frame, '[+] Vest', (x2 + 5, y1 + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (118, 230, 0), 1
            )
        else:
            cv2.putText(
                frame, '[X] Helmet', (x2 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (68, 23, 255), 1
            )
            cv2.putText(
                frame, '[+] Vest', (x2 + 5, y1 + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (118, 230, 0), 1
            )

    # Timestamp
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(
        frame, ts, (10, 470),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 120), 1
    )

    demo_stats = {
        'total_workers': 3,
        'compliant_workers': 2,
        'total_violations': 1,
        'fps': 24.5,
        'compliance_rate': 66.7,
        'frame_number': frame_num,
    }

    return frame, demo_stats


def main():
    """Main application entry point."""
    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown(
        '<p class="header-gradient">🏭 Industrial Safety Monitor</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">'
        'Real-time PPE compliance monitoring powered by computer vision'
        '</p>',
        unsafe_allow_html=True
    )

    # Initialize system
    system = init_system()

    # Initialize session state
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False
        st.session_state.processor = None
        st.session_state.video_source = 0
        st.session_state.required_ppe = ['helmet', 'vest']

    # Render sidebar
    source_type = render_sidebar(system['config'])

    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        start = st.button("▶️ Start Monitoring", type="primary")
    with col2:
        stop = st.button("⏹️ Stop Monitoring")

    # Handle start
    if start and not st.session_state.monitoring:
        system['ppe_detector'].set_required_ppe(
            st.session_state.get('required_ppe', ['helmet', 'vest'])
        )
        system['person_detector'].confidence_threshold = \
            st.session_state.get('confidence', 0.5)

        try:
            if source_type == "Webcam":
                processor = WebcamProcessor(
                    system['predictor'],
                    source=st.session_state.video_source
                )
                processor.start()
                st.session_state.processor = processor
                st.session_state.monitoring = True
            elif (source_type == "Upload Video" and
                  isinstance(st.session_state.get('video_source'), str) and
                  st.session_state.video_source != 'demo'):
                processor = VideoProcessor(
                    system['predictor'],
                    st.session_state.video_source
                )
                processor.start()
                st.session_state.processor = processor
                st.session_state.monitoring = True
            else:
                st.session_state.monitoring = True
                st.session_state.processor = None
        except Exception as e:
            st.error(f"Failed to start: {e}")
            logger.error(f"Start failed: {e}")

    # Handle stop
    if stop and st.session_state.monitoring:
        st.session_state.monitoring = False
        if st.session_state.processor:
            try:
                st.session_state.processor.stop()
            except Exception:
                pass
            st.session_state.processor = None
        st.rerun()

    # ─────────────────────────────────────────────────────────
    # Main content with tabs
    # ─────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📹 Live Monitor", "📊 Analytics", "📋 Violation Log"
    ])

    with tab1:
        if st.session_state.monitoring and st.session_state.processor:
            # Live video feed with real processor
            video_col, info_col = st.columns([2, 1])
            with video_col:
                frame_placeholder = st.empty()
            with info_col:
                status_placeholder = st.empty()
                alerts_placeholder = st.empty()
            metrics_placeholder = st.empty()

            while st.session_state.monitoring:
                frame = st.session_state.processor.get_frame()
                results = st.session_state.processor.get_results()

                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(
                        frame_rgb, channels="RGB",
                        use_container_width=True
                    )

                if results:
                    with metrics_placeholder.container():
                        render_realtime_metrics(
                            results.get('stats', {})
                        )
                    with status_placeholder.container():
                        st.markdown("### 📍 Active Workers")
                        workers = results.get('workers', {})
                        for wid, wdata in workers.items():
                            status = "✅" if wdata.get('compliant') else "❌"
                            missing = wdata.get('missing_ppe', [])
                            missing_str = (
                                f" (missing: {', '.join(missing)})"
                                if missing else ""
                            )
                            st.markdown(
                                f"{status} **Worker #{wid}**{missing_str}"
                            )
                    with alerts_placeholder.container():
                        render_active_alerts(
                            system['alert_manager'], limit=5
                        )

                # Check if video processing is complete
                if (hasattr(st.session_state.processor, 'is_complete') and
                        st.session_state.processor.is_complete):
                    st.success("✅ Video processing complete!")
                    break

                time.sleep(0.033)

        elif st.session_state.monitoring:
            # Demo mode
            st.info(
                "🎬 **Demo Mode** — Showing simulated safety monitoring. "
                "Connect a webcam or upload a video for real detection."
            )
            frame, demo_stats = create_demo_frame()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, channels="RGB", use_container_width=True)
            render_realtime_metrics(demo_stats)

        else:
            # Idle state
            st.markdown(
                """
                <div style="
                    text-align: center;
                    padding: 80px 20px;
                    color: #8892b0;
                ">
                    <p style="font-size: 3em; margin-bottom: 10px;">📹</p>
                    <p style="font-size: 1.2em;">
                        Click <strong>Start Monitoring</strong> to begin
                    </p>
                    <p style="font-size: 0.9em;">
                        Select a video source from the sidebar,
                        configure PPE requirements, and start monitoring.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab2:
        st.markdown("### 📊 Safety Analytics Dashboard")
        if system['db_manager']:
            col1, col2 = st.columns(2)
            with col1:
                render_violation_chart(system['db_manager'])
            with col2:
                render_compliance_timeline(system['db_manager'])

            st.markdown("---")
            col3, col4 = st.columns(2)
            with col3:
                render_worker_stats_table(system['db_manager'])
            with col4:
                render_violation_heatmap(system['db_manager'])

    with tab3:
        st.markdown("### 📋 Violation History")
        if system['db_manager']:
            render_violation_log(system['db_manager'])

            st.markdown("---")
            st.markdown("### 📥 Export Reports")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Export CSV Report"):
                    report_gen = ReportGenerator(system['db_manager'])
                    path = report_gen.generate_csv_report()
                    st.success(f"CSV report saved to `{path}`")
            with col2:
                if st.button("📥 Export JSON Report"):
                    report_gen = ReportGenerator(system['db_manager'])
                    path = report_gen.generate_json_report()
                    st.success(f"JSON report saved to `{path}`")


if __name__ == '__main__':
    main()
