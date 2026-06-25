"""
Dashboard visualization components for Streamlit.

Provides functions for rendering real-time metrics, charts,
tables, and alert feeds using Plotly and Streamlit widgets.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger('safety_monitor')

# Professional color palette
CHART_COLORS = [
    '#667eea', '#764ba2', '#f093fb', '#f5576c',
    '#4facfe', '#00f2fe', '#43e97b', '#fa709a',
    '#fee140', '#ff9a9e', '#a18cd1', '#fbc2eb',
]


def render_realtime_metrics(stats: dict):
    """Render real-time monitoring metrics as metric cards.

    Args:
        stats: Dict with total_workers, compliance_rate,
               total_violations, fps.
    """
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "👷 Workers Detected",
            stats.get('total_workers', 0)
        )
    with col2:
        rate = stats.get('compliance_rate', 0)
        st.metric("✅ Compliance Rate", f"{rate:.1f}%")
    with col3:
        st.metric(
            "⚠️ Violations",
            stats.get('total_violations', 0)
        )
    with col4:
        st.metric("📹 FPS", f"{stats.get('fps', 0):.1f}")


def render_violation_chart(db_manager):
    """Render violations by type as a horizontal bar chart.

    Args:
        db_manager: DatabaseManager instance.
    """
    st.subheader("📊 Violations by PPE Type")
    try:
        data = db_manager.get_violations_by_type()
        if not data:
            st.info("No violations recorded yet.")
            return

        df = pd.DataFrame(
            list(data.items()),
            columns=['PPE Type', 'Count']
        )
        df['PPE Type'] = df['PPE Type'].str.replace('_', ' ').str.title()

        fig = go.Figure(go.Bar(
            x=df['Count'],
            y=df['PPE Type'],
            orientation='h',
            marker=dict(
                color=CHART_COLORS[:len(df)],
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ),
            text=df['Count'],
            textposition='auto',
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading violation chart: {e}")


def render_compliance_timeline(db_manager, hours: int = 24):
    """Render compliance rate over time as a line chart.

    Args:
        db_manager: DatabaseManager instance.
        hours: Number of hours to display.
    """
    st.subheader("📈 Compliance Rate Over Time")
    try:
        data = db_manager.get_compliance_history(hours)
        if not data:
            st.info("No compliance data recorded yet.")
            return

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['compliance_rate'],
            mode='lines+markers',
            name='Compliance Rate',
            line=dict(color='#43e97b', width=2),
            marker=dict(size=4),
            fill='tozeroy',
            fillcolor='rgba(67, 233, 123, 0.1)',
        ))
        fig.add_hline(
            y=80, line_dash="dash",
            line_color="rgba(255, 87, 108, 0.5)",
            annotation_text="Target (80%)"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(
                range=[0, 105],
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                title='Compliance %'
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                title='Time'
            ),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading compliance timeline: {e}")


def render_worker_stats_table(db_manager):
    """Render worker-wise statistics table.

    Args:
        db_manager: DatabaseManager instance.
    """
    st.subheader("👷 Worker Statistics")
    try:
        stats = db_manager.get_worker_stats()
        if not stats:
            st.info("No worker data available yet.")
            return

        df = pd.DataFrame(stats)
        display_cols = {
            'worker_id': 'Worker ID',
            'total_violations': 'Total Violations',
            'total_frames_tracked': 'Frames Tracked',
            'violation_count': 'Violation Events',
            'first_seen': 'First Seen',
            'last_seen': 'Last Seen',
        }
        available = [c for c in display_cols if c in df.columns]
        df_display = df[available].rename(
            columns={k: v for k, v in display_cols.items() if k in available}
        )

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
        )
    except Exception as e:
        st.error(f"Error loading worker stats: {e}")


def render_violation_heatmap(db_manager, hours: int = 24):
    """Render violation frequency heatmap by hour and type.

    Args:
        db_manager: DatabaseManager instance.
        hours: Number of hours to display.
    """
    st.subheader("🔥 Violation Heatmap")
    try:
        trends = db_manager.get_violation_trends(hours)
        by_type = db_manager.get_violations_by_type()

        if not trends or not by_type:
            st.info("Not enough data for heatmap yet.")
            return

        # Create a simple time-series bar chart instead
        df = pd.DataFrame(
            list(trends.items()),
            columns=['Hour', 'Violations']
        )

        fig = go.Figure(go.Bar(
            x=df['Hour'],
            y=df['Violations'],
            marker=dict(
                color=df['Violations'],
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(title='Count'),
            ),
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(
                showgrid=False, title='Hour',
                tickangle=-45
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                title='Violations'
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading heatmap: {e}")


def render_active_alerts(alert_manager, limit: int = 10):
    """Render scrolling list of recent alerts.

    Args:
        alert_manager: AlertManager instance.
        limit: Max alerts to display.
    """
    st.subheader("🚨 Recent Alerts")
    alerts = alert_manager.get_recent_alerts(limit)

    if not alerts:
        st.info("No alerts triggered yet.")
        return

    for alert in alerts:
        missing = ", ".join(alert.get('missing_ppe', []))
        ts = alert.get('timestamp', '')
        if isinstance(ts, str) and len(ts) > 19:
            ts = ts[:19]

        st.markdown(
            f"""<div style="
                background: linear-gradient(135deg, #ff6b6b22, #ff6b6b11);
                border-left: 4px solid #ff6b6b;
                padding: 8px 12px;
                border-radius: 5px;
                margin: 4px 0;
                font-size: 0.85em;
            ">
                <strong>⚠️ Worker #{alert.get('worker_id', '?')}</strong>
                — Missing: {missing}<br>
                <span style="color: #888;">{ts}</span>
            </div>""",
            unsafe_allow_html=True
        )


def render_violation_log(db_manager, limit: int = 50):
    """Render searchable violation history table.

    Args:
        db_manager: DatabaseManager instance.
        limit: Max violations to display.
    """
    try:
        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            search_worker = st.text_input(
                "Filter by Worker ID", "",
                key="violation_search_worker"
            )
        with col2:
            filter_type = st.selectbox(
                "Filter by Type",
                ["All", "helmet", "vest", "mask", "goggles",
                 "gloves", "shoes", "ear_protection"],
                key="violation_filter_type"
            )

        violations = db_manager.get_recent_violations(limit)
        if not violations:
            st.info("No violations recorded yet.")
            return

        df = pd.DataFrame(violations)

        # Apply filters
        if search_worker:
            try:
                wid = int(search_worker)
                df = df[df['worker_id'] == wid]
            except ValueError:
                pass

        if filter_type != "All":
            df = df[df['violation_type'] == filter_type]

        # Display columns
        display_cols = [
            'id', 'worker_id', 'violation_type',
            'timestamp', 'frame_number', 'confidence'
        ]
        available = [c for c in display_cols if c in df.columns]

        if len(df) > 0:
            st.dataframe(
                df[available].rename(columns={
                    'id': 'ID',
                    'worker_id': 'Worker',
                    'violation_type': 'Violation Type',
                    'timestamp': 'Time',
                    'frame_number': 'Frame',
                    'confidence': 'Confidence',
                }),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"Showing {len(df)} of {limit} most recent violations")
        else:
            st.info("No violations match the current filters.")

    except Exception as e:
        st.error(f"Error loading violation log: {e}")


def render_session_summary(db_manager, session_id: int):
    """Render summary statistics for a session.

    Args:
        db_manager: DatabaseManager instance.
        session_id: Session ID to summarize.
    """
    stats = db_manager.get_session_stats(session_id)
    if not stats:
        st.warning(f"No data for session {session_id}")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Frames", stats.get('total_frames', 0))
    with col2:
        st.metric("Total Detections", stats.get('total_detections', 0))
    with col3:
        st.metric("Total Violations", stats.get('total_violations', 0))
