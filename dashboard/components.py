"""
Reusable UI components, editorial styling, and clean metric cards for the dashboard.
"""

from typing import Any, Dict, Optional
import streamlit as st


def inject_custom_css() -> None:
    """Inject clean, modern editorial styling with a light warm white background and forest green accents."""
    custom_css = """
    <style>
    /* Warm white background and crisp charcoal typography */
    .stApp {
        background-color: #fafafa;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Streamlit header */
    header[data-testid="stHeader"] {
        background-color: #fafafa;
    }

    /* Clean editorial card */
    .custom-metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .custom-metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .custom-metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .custom-metric-subtitle {
        font-size: 0.8rem;
        color: #475569;
        margin-top: 4px;
    }

    /* Risk Badges (Text + subtle tint for accessibility) */
    .risk-badge-high {
        display: inline-block;
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .risk-badge-medium {
        display: inline-block;
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .risk-badge-low {
        display: inline-block;
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.95rem;
    }

    /* Compact status badges */
    .status-ok {
        display: inline-block;
        background-color: #ecfdf5;
        color: #065f46;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #a7f3d0;
    }
    .status-missing {
        display: inline-block;
        background-color: #fef2f2;
        color: #991b1b;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #fecaca;
    }

    /* Editorial scope banner */
    .editorial-callout {
        background-color: #f8fafc;
        border-left: 3px solid #15803d;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 20px;
        font-size: 0.9rem;
        color: #334155;
    }

    /* Sidebar clean styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    /* Primary button: restrained forest green */
    div.stButton > button[kind="primary"],
    div.stButton > button:focus {
        background-color: #15803d;
        color: #ffffff;
        border: 1px solid #166534;
        border-radius: 6px;
        font-weight: 600;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #166534;
        border-color: #14532d;
        color: #ffffff;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_metric_card(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    badge: Optional[str] = None,
) -> None:
    """Render a clean white metric card with subtle borders."""
    badge_html = f"<span class='status-ok' style='float: right;'>{badge}</span>" if badge else ""
    sub_html = f"<div class='custom-metric-subtitle'>{subtitle}</div>" if subtitle else ""
    html = f"""
    <div class="custom-metric-card">
        <div class="custom-metric-title">{title} {badge_html}</div>
        <div class="custom-metric-value">{value}</div>
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_status_badge(is_available: bool, text_ok: str = "Ready", text_missing: str = "Missing") -> str:
    """Return HTML string for an artifact status badge."""
    if is_available:
        return f"<span class='status-ok'>✓ {text_ok}</span>"
    return f"<span class='status-missing'>✗ {text_missing}</span>"


def render_empty_state(
    artifact_name: str, remediation_command: str, explanation: str
) -> None:
    """Render a clear, helpful empty state message when an artifact is missing."""
    st.info(
        f"**Artifact Not Found**: {artifact_name}\n\n"
        f"{explanation}\n\n"
        f"**To generate this artifact:**\n"
        f"```bash\n{remediation_command}\n```"
    )
