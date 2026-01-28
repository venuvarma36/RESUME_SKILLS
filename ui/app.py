"""
Streamlit UI for Resume Skill Recognition & Matching System
Production-ready web interface for resume-JD matching.
"""

import os
import tempfile
import json
import hashlib
from pathlib import Path
from typing import List
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_js_eval import streamlit_js_eval

from matching_engine import ResumeJDMatcher
from utils import config, LoggerManager, get_logger


# Initialize logging
LoggerManager.setup_logging(
    level=config.get('logging.level', 'INFO'),
    log_to_file=config.get('logging.log_to_file', True),
    log_to_console=config.get('logging.log_to_console', True),
    log_dir=str(config.get_path('logs_dir'))
)

logger = get_logger(__name__)


# Page configuration
st.set_page_config(
    page_title=config.get('ui.page_title', 'Resume Skill Recognition'),
    page_icon=config.get('ui.page_icon', '📄'),
    layout=config.get('ui.layout', 'wide')
)


def inject_theme():
    """Apply clean, minimalistic, futuristic theme."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
            
            :root {
                --bg-dark: #0a0e1a;
                --bg-panel: rgba(255,255,255,0.03);
                --bg-panel-hover: rgba(255,255,255,0.06);
                --border: rgba(255,255,255,0.1);
                --text: #e8f0ff;
                --text-muted: #8b9dc3;
                --accent: #00d4ff;
                --accent-2: #7b5cff;
                --success: #00ff88;
                --error: #ff4466;
            }
            
            * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
            
            /* Main layout */
            [data-testid="stAppViewContainer"] {
                background: radial-gradient(ellipse at top left, #0f1628 0%, #0a0e1a 50%, #050810 100%);
            }
            
            .block-container {
                padding: 2rem 2rem 3rem 2rem;
                max-width: 1400px;
            }
            
            /* Sidebar */
            section[data-testid="stSidebar"] > div {
                background: linear-gradient(180deg, rgba(15,22,40,0.95) 0%, rgba(10,14,26,0.98) 100%);
                border-right: 1px solid var(--border);
                backdrop-filter: blur(20px);
            }
            
            section[data-testid="stSidebar"] .block-container {
                padding: 1.5rem 1rem;
            }
            
            /* Sidebar brand */
            .sidebar-brand {
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 20px;
                font-weight: 800;
                margin-bottom: 2rem;
                padding: 0.5rem;
                background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            /* Sidebar navigation - navbar style */
            .nav-tabs {
                display: flex;
                flex-direction: column;
                gap: 6px;
                margin-bottom: 2rem;
            }
            
            .nav-tab {
                border-radius: 10px;
                border: 1px solid transparent;
                padding: 12px 14px;
                background: var(--bg-panel);
                color: var(--text-muted);
                font-weight: 600;
                font-size: 14px;
                transition: all 0.2s ease;
                cursor: pointer;
                text-align: left;
                position: relative;
                overflow: hidden;
            }
            
            .nav-tab:hover {
                border-color: var(--border);
                background: var(--bg-panel-hover);
                color: var(--text);
                transform: translateX(2px);
            }
            
            .nav-tab.active {
                border: 1px solid var(--accent);
                background: linear-gradient(135deg, rgba(0,212,255,0.1) 0%, rgba(123,92,255,0.1) 100%);
                color: var(--text);
                box-shadow: 0 4px 20px rgba(0,212,255,0.2);
            }
            
            .nav-tab.active::before {
                content: '';
                position: absolute;
                left: 0;
                top: 0;
                bottom: 0;
                width: 3px;
                background: linear-gradient(180deg, var(--accent) 0%, var(--accent-2) 100%);
            }
            
            /* Processing flow */
            .process-flow {
                display: flex;
                flex-direction: column;
                gap: 12px;
                padding: 1.5rem;
                background: var(--bg-panel);
                border: 1px solid var(--border);
                border-radius: 16px;
                margin-bottom: 1.5rem;
            }
            
            .process-step {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 10px;
                border-radius: 10px;
                background: rgba(255,255,255,0.02);
                transition: all 0.3s ease;
            }
            
            .process-step.active {
                background: linear-gradient(135deg, rgba(0,212,255,0.15) 0%, rgba(123,92,255,0.15) 100%);
                border: 1px solid var(--accent);
            }
            
            .process-step.completed {
                background: rgba(0,255,136,0.1);
                border: 1px solid rgba(0,255,136,0.3);
            }
            
            .step-icon {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                background: var(--bg-panel);
                border: 2px solid var(--border);
            }
            
            .process-step.active .step-icon {
                background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
                border-color: var(--accent);
                animation: pulse 1.5s ease-in-out infinite;
            }
            
            .process-step.completed .step-icon {
                background: var(--success);
                border-color: var(--success);
                color: #000;
            }
            
            @keyframes pulse {
                0%, 100% { box-shadow: 0 0 0 0 rgba(0,212,255,0.7); }
                50% { box-shadow: 0 0 0 8px rgba(0,212,255,0); }
            }
            
            .step-content {
                flex: 1;
            }
            
            .step-title {
                font-weight: 600;
                color: var(--text);
                font-size: 14px;
            }
            
            .step-desc {
                font-size: 12px;
                color: var(--text-muted);
                margin-top: 2px;
            }
            
            /* Loading overlay */
            .loading-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(5, 8, 16, 0.95);
                backdrop-filter: blur(10px);
                z-index: 9999;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .loading-content {
                text-align: center;
                max-width: 600px;
                padding: 2rem;
            }
            
            .loading-spinner {
                width: 80px;
                height: 80px;
                margin: 0 auto 2rem;
                border: 4px solid var(--border);
                border-top: 4px solid var(--accent);
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            /* Tab loading animation */
            .tab-loading {
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 4rem 2rem;
                flex-direction: column;
                gap: 1.5rem;
            }
            
            .tab-loading-spinner {
                width: 50px;
                height: 50px;
                border: 3px solid var(--border);
                border-top: 3px solid var(--accent);
                border-radius: 50%;
                animation: spin 0.8s linear infinite;
            }
            
            .tab-loading-text {
                color: var(--text-muted);
                font-size: 14px;
            }
            
            /* Auth loading animation */
            .auth-loading {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(5, 8, 16, 0.98);
                backdrop-filter: blur(15px);
                z-index: 10000;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .auth-loading-content {
                text-align: center;
                padding: 3rem;
                background: var(--bg-panel);
                border: 1px solid var(--border);
                border-radius: 16px;
                min-width: 300px;
            }
            
            .auth-loading-spinner {
                width: 60px;
                height: 60px;
                margin: 0 auto 1.5rem;
                border: 3px solid var(--border);
                border-top: 3px solid var(--accent);
                border-radius: 50%;
                animation: spin 0.8s linear infinite;
            }
            
            .auth-loading-text {
                color: var(--text);
                font-size: 16px;
                font-weight: 500;
                margin-bottom: 0.5rem;
            }
            
            .auth-loading-subtext {
                color: var(--text-muted);
                font-size: 13px;
            }
            
            /* Panels */
            .futuristic-panel {
                background: var(--bg-panel);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            }
            
            .hero-panel {
                background: linear-gradient(135deg, rgba(0,212,255,0.1) 0%, rgba(123,92,255,0.1) 100%);
                border: 1px solid var(--accent);
                border-radius: 20px;
                padding: 2.5rem;
                box-shadow: 0 12px 40px rgba(0,212,255,0.2);
            }
            
            .auth-panel {
                background: var(--bg-panel);
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 2rem;
                box-shadow: 0 12px 40px rgba(0,0,0,0.4);
            }
            
            /* Typography */
            .hero-title {
                font-size: 36px;
                font-weight: 800;
                background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 0.5rem;
            }
            
            .hero-subtitle {
                font-size: 18px;
                color: var(--text-muted);
                line-height: 1.6;
                margin-bottom: 1.5rem;
            }
            
            .section-title {
                font-size: 24px;
                font-weight: 700;
                color: var(--text);
                margin-bottom: 0.5rem;
            }
            
            .pill {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 6px 14px;
                border-radius: 20px;
                background: rgba(0,212,255,0.15);
                border: 1px solid rgba(0,212,255,0.3);
                color: var(--accent);
                font-size: 13px;
                font-weight: 600;
                margin-bottom: 1rem;
            }
            
            /* Buttons */
            div.stButton > button {
                width: 100%;
                background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
                color: #000;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 12px;
                font-weight: 700;
                font-size: 15px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 20px rgba(0,212,255,0.3);
            }
            
            div.stButton > button:hover {
                box-shadow: 0 8px 30px rgba(0,212,255,0.5);
                transform: translateY(-2px);
            }
            
            /* Sidebar nav buttons */
            .nav-tabs div.stButton > button {
                background: var(--bg-panel);
                color: var(--text-muted);
                text-align: left;
                border: 1px solid transparent;
                box-shadow: none;
                font-weight: 600;
                font-size: 14px;
            }
            
            .nav-tabs div.stButton > button:hover {
                border-color: var(--border);
                background: var(--bg-panel-hover);
                color: var(--text);
                transform: translateX(2px);
                box-shadow: none;
            }
            
            .logout-btn > button {
                background: rgba(255,68,102,0.15) !important;
                color: var(--error) !important;
                border: 1px solid rgba(255,68,102,0.3) !important;
                box-shadow: none !important;
            }
            
            .logout-btn > button:hover {
                background: rgba(255,68,102,0.25) !important;
                box-shadow: 0 4px 20px rgba(255,68,102,0.3) !important;
            }
            
            /* Form elements */
            .stTextInput > div > div > input,
            .stTextArea > div > div > textarea,
            .stSelectbox > div > div > select {
                background: rgba(255,255,255,0.05) !important;
                border: 1px solid var(--border) !important;
                border-radius: 10px !important;
                color: var(--text) !important;
                font-size: 14px !important;
            }
            
            .stTextInput > div > div > input:focus,
            .stTextArea > div > div > textarea:focus {
                border-color: var(--accent) !important;
                box-shadow: 0 0 0 1px var(--accent) !important;
            }
            
            /* Auth mode toggle */
            .auth-toggle [role="radiogroup"] {
                display: flex;
                gap: 8px;
                margin-bottom: 1.5rem;
            }
            
            .auth-toggle [role="radiogroup"] > label {
                flex: 1;
                text-align: center;
                padding: 12px;
                border-radius: 10px;
                border: 1px solid var(--border);
                background: var(--bg-panel);
                color: var(--text-muted);
                font-weight: 600;
                transition: all 0.2s ease;
            }
            
            .auth-toggle [role="radiogroup"] > label:hover {
                border-color: var(--accent);
                background: var(--bg-panel-hover);
            }
            
            .auth-toggle [aria-checked="true"] {
                background: linear-gradient(135deg, rgba(0,212,255,0.2) 0%, rgba(123,92,255,0.2) 100%);
                border-color: var(--accent);
                color: var(--text);
            }
            
            /* Skills chips */
            .skill-chip {
                background: rgba(0,255,136,0.1);
                border: 1px solid rgba(0,255,136,0.3);
                color: var(--success);
                padding: 6px 12px;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 600;
                display: inline-flex;
                margin: 3px;
            }
            
            /* Hide Streamlit elements */
            #MainMenu { visibility: hidden; }
            footer { visibility: hidden; }
            header { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True
    )


def initialize_session_state():
    """Initialize session state variables."""
    if 'matcher' not in st.session_state:
        with st.spinner('Initializing system... This may take a moment.'):
            st.session_state.matcher = ResumeJDMatcher()
    
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    if 'processed_resumes' not in st.session_state:
        st.session_state.processed_resumes = []
    
    if 'uploaded_file_objects' not in st.session_state:
        st.session_state.uploaded_file_objects = {}

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if 'auth_accounts' not in st.session_state:
        st.session_state.auth_accounts = []

    if 'active_nav' not in st.session_state:
        st.session_state.active_nav = "Upload & Job Description"

    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = "Login"

    if 'processing_step' not in st.session_state:
        st.session_state.processing_step = None

    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

    if 'uploaded_files_data' not in st.session_state:
        st.session_state.uploaded_files_data = None

    if 'jd_text_input' not in st.session_state:
        st.session_state.jd_text_input = ""

    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False

    if 'tab_loading' not in st.session_state:
        st.session_state.tab_loading = False

    if 'logout_triggered' not in st.session_state:
        st.session_state.logout_triggered = False

    if 'auth_loading' not in st.session_state:
        st.session_state.auth_loading = False

    if 'auth_loading_message' not in st.session_state:
        st.session_state.auth_loading_message = ""

    if 'pending_auth' not in st.session_state:
        st.session_state.pending_auth = None

    if 'session_restored' not in st.session_state:
        st.session_state.session_restored = False


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def load_accounts() -> list:
    """Load accounts from browser localStorage; fallback to session cache."""
    # Return cached if already loaded
    if st.session_state.get('auth_accounts'):
        return st.session_state.auth_accounts
    
    # Best-effort read; fallback to session if component not available
    try:
        data = streamlit_js_eval(
            js_expressions="JSON.parse(localStorage.getItem('skills_auth') || '[]')",
            key="load_accounts_static",
            default=[],
        )
        if isinstance(data, list):
            st.session_state.auth_accounts = data
            return data
    except Exception:
        pass
    return st.session_state.get('auth_accounts', [])


def persist_session(user: str, password_hash: str) -> None:
    """Persist the current logged-in session to localStorage."""
    try:
        import time
        timestamp = str(int(time.time() * 1000))
        
        streamlit_js_eval(
            js_expressions=f"localStorage.setItem('skills_auth_session', JSON.stringify({json.dumps({'u': user, 'p': password_hash})}))",
            key=f"save_session_{timestamp}",
        )
        # Set logged_in flag to true
        streamlit_js_eval(
            js_expressions="localStorage.setItem('logged_in', 'true')",
            key=f"set_logged_in_{timestamp}",
        )
    except Exception:
        pass


def clear_persisted_session() -> None:
    """Remove the persisted session from localStorage."""
    try:
        import time
        timestamp = str(int(time.time() * 1000))
        
        streamlit_js_eval(
            js_expressions="localStorage.removeItem('skills_auth_session')",
            key=f"clear_session_{timestamp}",
        )
        # Set logged_in flag to false
        streamlit_js_eval(
            js_expressions="localStorage.setItem('logged_in', 'false')",
            key=f"set_logged_out_{timestamp}",
        )
    except Exception:
        pass


def restore_session_if_available() -> None:
    """Restore login state from localStorage if valid credentials exist."""
    # Don't restore if user just logged out, already authenticated, or already tried
    if (st.session_state.get('authenticated', False) or 
        st.session_state.get('logout_triggered', False) or
        st.session_state.get('session_restored', False)):
        return

    # Mark as attempted to prevent infinite loop
    st.session_state.session_restored = True

    # Ensure accounts are loaded for validation
    accounts = st.session_state.get('auth_accounts', []) or load_accounts()

    try:
        # Check logged_in flag first (static key)
        logged_in = streamlit_js_eval(
            js_expressions="localStorage.getItem('logged_in')",
            key="check_logged_in_static",
            default="false",
        )
        
        if logged_in == "true":
            data = streamlit_js_eval(
                js_expressions="JSON.parse(localStorage.getItem('skills_auth_session') || 'null')",
                key="load_session_static",
                default=None,
            )
            if isinstance(data, dict):
                u = data.get('u')
                p = data.get('p')
                match = next((acct for acct in accounts if acct.get('u') == u and acct.get('p') == p), None)
                if match:
                    st.session_state.authenticated = True
                    st.session_state.username = u
                    st.session_state.auth_accounts = accounts
    except Exception:
        pass


def save_accounts(accounts: list) -> None:
    """Persist accounts to localStorage and cache in session."""
    try:
        import time
        timestamp = str(int(time.time() * 1000))
        
        streamlit_js_eval(
            js_expressions=f"localStorage.setItem('skills_auth', JSON.stringify({json.dumps(accounts)}))",
            key=f"save_accounts_{timestamp}",
        )
    except Exception:
        pass
    st.session_state.auth_accounts = accounts


def render_sidebar():
    """Render sidebar with navigation."""
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">🚀 Skills Console</div>', unsafe_allow_html=True)
        
        nav_options = [
            ("📤 Upload & Job Description", "Upload & Job Description"),
            ("🔍 Extraction Preview", "Extraction Preview"),
            ("⚡ Skill Extraction", "Skill Extraction"),
            ("📊 Match Score & Explainability", "Match Score & Explainability"),
            ("🎯 Career Path & Suggestions", "Career Path & Suggestions"),
        ]
        
        # Disable navigation during processing
        is_disabled = st.session_state.get('is_processing', False)
        
        st.markdown('<div class="nav-tabs">', unsafe_allow_html=True)
        for label, value in nav_options:
            active_class = "active" if st.session_state.active_nav == value else ""
            if st.button(label, key=f"nav_{value}", use_container_width=True, disabled=is_disabled):
                # If switching tabs with results available, show loading
                if st.session_state.results is not None and st.session_state.active_nav != value:
                    st.session_state.tab_loading = True
                st.session_state.active_nav = value
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
        if st.button("🚪 Logout", key="sidebar_logout", disabled=is_disabled):
            # Show logout animation
            st.session_state.auth_loading = True
            st.session_state.auth_loading_message = "Logging out..."
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)


def render_processing_flow():
    """Render processing flow visualization."""
    steps = [
        {"icon": "📄", "title": "Extracting Text", "desc": "Reading resume content", "key": "extract"},
        {"icon": "⚡", "title": "Skills Extraction", "desc": "Identifying skills using AI", "key": "skills"},
        {"icon": "🔄", "title": "Comparing with JD", "desc": "Matching against job description", "key": "compare"},
        {"icon": "✅", "title": "Generating Results", "desc": "Calculating scores & rankings", "key": "result"},
    ]
    
    st.markdown('<div class="process-flow">', unsafe_allow_html=True)
    
    for i, step in enumerate(steps):
        if st.session_state.processing_complete:
            step_class = "completed"
            icon = "✓"
        elif st.session_state.processing_step == step["key"]:
            step_class = "active"
            icon = step["icon"]
        elif st.session_state.processing_step and steps.index(next(s for s in steps if s["key"] == st.session_state.processing_step)) > i:
            step_class = "completed"
            icon = "✓"
        else:
            step_class = ""
            icon = step["icon"]
        
        st.markdown(
            f'''
            <div class="process-step {step_class}">
                <div class="step-icon">{icon}</div>
                <div class="step-content">
                    <div class="step-title">{step["title"]}</div>
                    <div class="step-desc">{step["desc"]}</div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_loading_overlay():
    """Render full-screen blocking loading overlay."""
    st.markdown(
        '''
        <div class="loading-overlay">
            <div class="loading-content">
                <div class="loading-spinner"></div>
                <div class="section-title">Processing Resumes</div>
                <p style="color: var(--text-muted); margin-top: 1rem;">Analyzing candidates using advanced AI. This may take a moment...</p>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )


def render_auth_loading(message: str):
    """Render auth loading animation overlay."""
    st.markdown(
        f'''
        <div class="auth-loading">
            <div class="auth-loading-content">
                <div class="auth-loading-spinner"></div>
                <div class="auth-loading-text">{message}</div>
                <div class="auth-loading-subtext">Please wait...</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )


def render_auth_gate():
    """Render login/register page before showing dashboard."""
    # Reset logout flag and session restored flag when showing auth page
    st.session_state.logout_triggered = False
    st.session_state.session_restored = False
    
    accounts = st.session_state.get('auth_accounts', []) or load_accounts()

    # Center the login form
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        # Main heading
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 3rem;">
                <h1 style="font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;">
                    Resume Intelligence Console
                </h1>
                <p style="color: var(--text-muted); font-size: 1rem;">AI-powered resume analysis platform</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # st.markdown('<div class="auth-panel">', unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">Welcome Back</div>', unsafe_allow_html=True)
        st.markdown('<p style="color: var(--text-muted); margin-bottom: 1.5rem;">Sign in to access your dashboard</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="auth-toggle">', unsafe_allow_html=True)
        auth_mode = st.radio(
            "Mode",
            options=["Login", "Register"],
            horizontal=True,
            index=0 if st.session_state.auth_mode == "Login" else 1,
            label_visibility="collapsed",
            key="auth_mode_radio"
        )
        st.session_state.auth_mode = auth_mode
        # st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.auth_mode == "Login":
            username = st.text_input("Username", key="login_user", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
            if st.button("Login", type="primary"):
                if not username or not password:
                    st.error("Enter username and password")
                else:
                    hashed = hash_password(password)
                    match = next((acct for acct in accounts if acct.get('u') == username and acct.get('p') == hashed), None)
                    if match:
                        st.session_state.auth_loading = True
                        st.session_state.auth_loading_message = "Logging in..."
                        st.session_state.pending_auth = {'username': username, 'hashed': hashed}
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()
                    else:
                        st.error("Invalid credentials")
        else:
            new_user = st.text_input("Username", key="reg_user", placeholder="Choose a username")
            new_pass = st.text_input("Password", type="password", key="reg_pass", placeholder="Choose a password")
            confirm_pass = st.text_input("Confirm Password", type="password", key="reg_confirm", placeholder="Re-enter password")
            if st.button("Create Account", type="primary"):
                if not new_user or not new_pass:
                    st.error("Username and password required")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match")
                elif any(acct.get('u') == new_user for acct in accounts):
                    st.error("Username already exists")
                else:
                    hashed = hash_password(new_pass)
                    updated = accounts + [{"u": new_user, "p": hashed}]
                    save_accounts(updated)
                    # Show registration loading animation
                    st.session_state.auth_loading = True
                    st.session_state.auth_loading_message = "Creating account..."
                    st.session_state.pending_auth = {'username': new_user, 'hashed': hashed}
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main application function."""
    inject_theme()
    initialize_session_state()
    restore_session_if_available()

    # Handle auth loading states (login/logout animations)
    if st.session_state.get('auth_loading', False):
        message = st.session_state.get('auth_loading_message', 'Processing...')
        render_auth_loading(message)
        
        import time
        time.sleep(1.5)  # Show animation for 1.5 seconds
        
        # Check if this is a logout or login/register
        if message == "Logging out...":
            # Clear localStorage
            clear_persisted_session()
            
            # Clear all session state
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.logout_triggered = True
            st.session_state.results = None
            st.session_state.processed_resumes = []
            st.session_state.uploaded_files_data = None
            st.session_state.jd_text_input = ""
            st.session_state.active_nav = "Upload & Job Description"
            st.session_state.auth_loading = False
            st.session_state.pending_auth = None
        else:
            # Handle login/register
            if st.session_state.pending_auth:
                username = st.session_state.pending_auth['username']
                hashed = st.session_state.pending_auth['hashed']
                st.session_state.authenticated = True
                st.session_state.username = username
                persist_session(username, hashed)
                st.session_state.auth_loading = False
                st.session_state.pending_auth = None
        
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
        return

    if not st.session_state.get('authenticated', False):
        render_auth_gate()
        return

    render_sidebar()

    # Show loading overlay if processing
    if st.session_state.is_processing:
        render_loading_overlay()

    # Header
    st.markdown(
        """
        <div class="futuristic-panel">
            <div class="pill">🤖 AI-Powered Matching</div>
            <div class="section-title">Resume Skill Recognition & Matching</div>
            <p style="color: var(--text-muted);">Upload resumes, paste job descriptions, and discover the best candidates using advanced AI.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show results banner if available
    if st.session_state.results is not None and not st.session_state.is_processing:
        st.success(f"✅ Results ready! {len(st.session_state.results)} resume(s) analyzed. Navigate through tabs to explore.")

    nav_choice = st.session_state.active_nav

    # If processing flag is set, do the actual work
    if st.session_state.is_processing and st.session_state.uploaded_files_data and st.session_state.jd_text_input:
        do_actual_processing(st.session_state.uploaded_files_data, st.session_state.jd_text_input)

    # Handle tab loading animation
    if st.session_state.tab_loading:
        st.markdown(
            '''
            <div class="tab-loading">
                <div class="tab-loading-spinner"></div>
                <div class="tab-loading-text">Loading content...</div>
            </div>
            ''',
            unsafe_allow_html=True
        )
        import time
        time.sleep(2)
        st.session_state.tab_loading = False
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
        return

    if nav_choice == "Upload & Job Description":
        uploaded_files, jd_text, match_button = render_upload_tab()
        if match_button:
            process_matching(uploaded_files, jd_text)
    elif nav_choice == "Extraction Preview":
        if st.session_state.results is None:
            st.info("⬅️ Run matching in Upload to see extraction details.")
        else:
            render_extraction_preview(st.session_state.results)
    elif nav_choice == "Skill Extraction":
        if st.session_state.results is None:
            st.info("⬅️ Run matching in Upload to see extracted skills.")
        else:
            render_skill_extraction(st.session_state.results)
    elif nav_choice == "Match Score & Explainability":
        if st.session_state.results is None:
            st.info("⬅️ Run matching in Upload to see scores and explainability.")
        else:
            render_match_and_explainability(st.session_state.results)
    elif nav_choice == "Career Path & Suggestions":
        if st.session_state.results is None:
            display_welcome_message()
        else:
            render_career_and_downloads(st.session_state.results)


def process_matching(uploaded_files, jd_text):
    """
    Process resume matching.
    
    Args:
        uploaded_files: List of uploaded files
        jd_text: Job description text
    """
    try:
        # Set processing flag
        st.session_state.is_processing = True
        st.session_state.processing_step = "extract"
        st.session_state.processing_complete = False
        
        # Force rerun to show overlay
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
        
    except Exception as e:
        st.error(f"❌ Error starting process: {str(e)}")
        logger.error("Error starting resume matching: %s", str(e), exc_info=True)
        st.session_state.is_processing = False
        st.session_state.processing_step = None
        st.session_state.processing_complete = False


def do_actual_processing(uploaded_files, jd_text):
    """
    Perform the actual resume matching process.
    
    Args:
        uploaded_files: List of uploaded files
        jd_text: Job description text
    """
    try:
        # Save uploaded files temporarily
        temp_dir = tempfile.mkdtemp()
        resume_paths = []
        st.session_state.uploaded_file_objects = {}
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            resume_paths.append(file_path)
            st.session_state.uploaded_file_objects[file_path] = uploaded_file.getvalue()
        
        logger.info("Starting resume matching for %d files", len(resume_paths))
        
        # Perform matching
        results_df = st.session_state.matcher.match_resumes_to_jd(
            resume_paths, jd_text
        )
        
        # Store results
        st.session_state.results = results_df
        
        logger.info("Resume matching completed successfully")
        
        # Clear processing state
        st.session_state.is_processing = False
        st.session_state.processing_step = None
        st.session_state.processing_complete = True
        
        # Navigate to results
        st.session_state.active_nav = "Match Score & Explainability"
        
        # Rerun to show results
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
        
    except Exception as e:
        st.session_state.is_processing = False
        st.session_state.processing_step = None
        st.session_state.processing_complete = False
        raise e


def render_upload_tab():
    """Tab 1: Upload resumes and provide job description."""
    st.subheader("Upload & Job Description")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Choose resume files (PDF/DOCX)",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX. Scanned PDFs OK (OCR)."
        )
        
        # Store uploaded files in session state
        if uploaded_files:
            st.session_state.uploaded_files_data = uploaded_files
            st.success(f"✓ {len(uploaded_files)} file(s) uploaded")
            with st.expander("View uploaded files"):
                for file in uploaded_files:
                    st.text(f"• {file.name}")
        elif st.session_state.uploaded_files_data:
            # Show previously uploaded files
            st.success(f"✓ {len(st.session_state.uploaded_files_data)} file(s) uploaded")
            with st.expander("View uploaded files"):
                for file in st.session_state.uploaded_files_data:
                    st.text(f"• {file.name}")

    with col2:
        jd_text = st.text_area(
            "Paste job description",
            height=220,
            placeholder="Enter the full job description with required skills...",
            value=st.session_state.jd_text_input
        )
        
        # Store JD text in session state
        if jd_text:
            st.session_state.jd_text_input = jd_text
        
        st.caption("Tip: Include must-have and nice-to-have skills for better matching.")

    # Use stored values if current widgets are empty
    files_to_use = uploaded_files if uploaded_files else st.session_state.uploaded_files_data
    jd_to_use = jd_text if jd_text else st.session_state.jd_text_input

    match_button = st.button(
        "🚀 Match Resumes",
        type="primary",
        width="stretch",
        disabled=(not files_to_use or not jd_to_use)
    )

    return files_to_use, jd_to_use, match_button


def render_extraction_preview(results_df: pd.DataFrame):
    """Tab 2: Extraction preview and basic stats."""
    st.subheader("Extraction Preview")

    if results_df.empty:
        st.warning("No results to display")
        return

    with st.expander("View extraction details", expanded=True):
        for _, row in results_df.iterrows():
            resume_name = Path(row['resume_file']).name
            cols = st.columns([3, 1, 1, 1])
            cols[0].markdown(f"**{resume_name}**")
            cols[1].markdown(f"Method: `{row.get('extraction_method', 'unknown')}`")
            ocr_used = 'Yes' if 'ocr' in str(row.get('extraction_method', '')).lower() else 'No'
            cols[2].markdown(f"OCR used: **{ocr_used}**")
            cols[3].markdown(f"Success: **{row.get('extraction_success', False)}**")
            pages = row.get('pages', []) if isinstance(row.get('pages', []), list) else []
            if pages:
                with st.expander("Page-level evidence", expanded=False):
                    for page in pages[:5]:
                        st.markdown(f"Page {page.get('page_number', '?')}")
                        st.caption(page.get('text', '')[:400])
                        blk_count = len(page.get('blocks', []))
                        st.caption(f"Blocks captured: {blk_count}")
        st.caption("Text preview not stored in results; add capture in backend if needed.")


def render_skill_extraction(results_df: pd.DataFrame):
    """Tab 3: Skill extraction grouped and separated matched vs missing."""
    st.subheader("Skill Extraction")

    if results_df.empty:
        st.warning("No results to display")
        return

    for idx, row in results_df.iterrows():
        resume_name = Path(row['resume_file']).name
        with st.expander(f"📄 Rank #{row['rank']}: {resume_name} - {row['match_percentage']}", expanded=(idx == 0)):
            all_skills = [s.strip() for s in row.get('all_extracted_skills', '').split(',') if s.strip()]
            matched_skills = [s.strip() for s in row.get('matched_skills', '').split(',') if s.strip()]
            missing_skills = [s.strip() for s in row.get('missing_skills', '').split(',') if s.strip()]
            skill_evidence = row.get('skill_evidence', {}) if isinstance(row.get('skill_evidence', {}), dict) else {}

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**All Extracted Skills**")
                st.caption(f"Total: {len(all_skills)}")
                _render_skill_tags(all_skills, bg="#d1ecf1", fg="#0c5460")
            with col2:
                st.markdown("**Matched Skills**")
                st.caption(f"Count: {len(matched_skills)}")
                _render_skill_tags(matched_skills, bg="#d4edda", fg="#155724")

            st.markdown("**Missing Skills**")
            st.caption(f"Count: {len(missing_skills)}")
            _render_skill_tags(missing_skills, bg="#f8d7da", fg="#721c24")

            if skill_evidence:
                with st.expander("Skill Evidence (page & bbox)", expanded=False):
                    for skill, entries in skill_evidence.items():
                        st.markdown(f"**{skill}**")
                        for ev in entries:
                            page = ev.get('page', '?')
                            bbox = ev.get('bbox')
                            method = ev.get('method', 'text')
                            st.caption(f"Page {page} | Method: {method} | BBox: {bbox if bbox else 'n/a'}")


def render_match_and_explainability(results_df: pd.DataFrame):
    """Tab 4: Scores, similarity fusion, charts, and explainability placeholder."""
    if results_df.empty:
        st.warning("No results to display")
        return

    st.subheader("Match Score & Explainability")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Candidates", len(results_df))
    col2.metric("Best Match", results_df.iloc[0]['match_percentage'])
    col3.metric("Average Score", f"{results_df['overall_score'].mean() * 100:.1f}%")
    col4.metric("Qualified (≥50%)", (results_df['overall_score'] >= 0.5).sum())
    col5.metric("Fusion Similarity", f"{results_df.get('quad_score', pd.Series([0])).mean() * 100:.1f}%")

    top_row = results_df.iloc[0]
    st.markdown("### Overall Match")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=top_row['overall_score'] * 100,
            title={'text': "Overall Match"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"}
            }
        ))
        st.plotly_chart(fig_gauge, width="stretch")

    with col_b:
        categories = {
            'Technical': top_row.get('technical_skills_score', 0),
            'Tools': top_row.get('tools_score', 0),
            'Frameworks': top_row.get('frameworks_score', 0),
            'Soft': top_row.get('soft_skills_score', 0)
        }
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=list(categories.values()),
            theta=list(categories.keys()),
            fill='toself'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
        st.plotly_chart(fig_radar, width="stretch")

    st.markdown("### Similarity Fusion Breakdown")
    fusion_cols = st.columns(7)
    fusion_cols[0].metric("Semantic", f"{top_row.get('quad_semantic', 0.0) * 100:.1f}%")
    fusion_cols[1].metric("Jaccard", f"{top_row.get('quad_jaccard', 0.0) * 100:.1f}%")
    fusion_cols[2].metric("Fuzzy", f"{top_row.get('quad_fuzzy', 0.0) * 100:.1f}%")
    fusion_cols[3].metric("Graph", f"{top_row.get('quad_graph', 0.0) * 100:.1f}%")
    fusion_cols[4].metric("Fused", f"{top_row.get('quad_score', 0.0) * 100:.1f}%")
    fusion_cols[5].metric("Context", f"{top_row.get('context_match', 0.0) * 100:.1f}%")
    fusion_cols[6].metric("Domain", f"{top_row.get('domain_relevance', 0.0) * 100:.1f}%")

    fusion_rows = [
        ["Semantic (SBERT)", top_row.get('quad_semantic', 0.0)],
        ["Jaccard (skills)", top_row.get('quad_jaccard', 0.0)],
        ["Fuzzy (phrasing)", top_row.get('quad_fuzzy', 0.0)],
        ["Graph (coverage)", top_row.get('quad_graph', 0.0)],
        ["Context (classifier)", top_row.get('context_match', 0.0)],
        ["Domain relevance", top_row.get('domain_relevance', 0.0)],
        ["Fused Score", top_row.get('quad_score', 0.0)]
    ]
    fusion_df = pd.DataFrame(fusion_rows, columns=["Signal", "Score"])
    fusion_df['Score'] = fusion_df['Score'].apply(lambda x: f"{float(x) * 100:.1f}%")
    st.table(fusion_df)

    shap_vals = top_row.get('shap_values')
    if shap_vals:
        st.markdown("### SHAP Feature Contributions")
        shap_items = [(k, v) for k, v in shap_vals.items()]
        shap_df = pd.DataFrame(shap_items, columns=["Feature", "SHAP value"])
        shap_df['SHAP value'] = shap_df['SHAP value'].apply(lambda x: f"{x:.3f}")
        st.table(shap_df)

    st.markdown("### Category Scores by Candidate")
    cat_df = results_df[
        ['resume_file', 'technical_skills_score', 'tools_score', 'frameworks_score', 'soft_skills_score']
    ].copy()
    cat_df['resume_name'] = cat_df['resume_file'].apply(lambda x: Path(x).name)
    cat_melt = cat_df.melt(id_vars=['resume_name'], var_name='Category', value_name='Score')
    fig_cat_bar = px.bar(
        cat_melt,
        x='Category',
        y='Score',
        color='resume_name',
        barmode='group',
        title='Category Scores (All Candidates)',
        labels={'Score': 'Score', 'Category': 'Category', 'resume_name': 'Resume'}
    )
    fig_cat_bar.update_layout(xaxis_title='Category', yaxis_title='Score (0-1)')
    st.plotly_chart(fig_cat_bar, width="stretch")

    st.markdown("### Category Heatmap (Top 10)")
    heat_df = cat_df.head(10).set_index('resume_name')
    fig_heat = px.imshow(
        heat_df[[
            'technical_skills_score', 'tools_score', 'frameworks_score', 'soft_skills_score'
        ]],
        labels={'x': 'Category', 'y': 'Resume', 'color': 'Score'},
        title='Category Coverage Heatmap',
        color_continuous_scale='Blues',
        aspect='auto'
    )
    st.plotly_chart(fig_heat, width="stretch")

    st.markdown("### Explainability")
    st.info("SHAP/attribution placeholder: integrate SHAP plots when available.")

    st.markdown("### Rankings & Distribution")
    col1, col2 = st.columns(2)
    with col1:
        top_10 = results_df.head(10).copy()
        top_10['resume_name'] = top_10['resume_file'].apply(lambda x: Path(x).name)
        fig_bar = px.bar(
            top_10,
            x='overall_score',
            y='resume_name',
            orientation='h',
            title='Top 10 Candidates by Match Score',
            labels={'overall_score': 'Match Score', 'resume_name': 'Resume'},
            color='overall_score',
            color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar, width="stretch")
    with col2:
        fig_hist = px.histogram(
            results_df,
            x='overall_score',
            nbins=20,
            title='Score Distribution',
            labels={'overall_score': 'Match Score', 'count': 'Number of Candidates'},
            color_discrete_sequence=['#636EFA']
        )
        st.plotly_chart(fig_hist, width="stretch")


def render_career_and_downloads(results_df: pd.DataFrame):
    """Tab 5: Career suggestions, missing skills, ranked table, downloads."""
    st.subheader("Career / Improvement Suggestions")

    if results_df.empty:
        st.warning("No results to display")
        return

    # Use top candidate missing skills as quick guidance
    top_row = results_df.iloc[0]
    missing_skills = [s.strip() for s in top_row.get('missing_skills', '').split(',') if s.strip()]
    if missing_skills:
        st.markdown("**Priority Gaps:**")
        _render_skill_tags(missing_skills[:10], bg="#ffeeba", fg="#7c5a00")
        st.markdown("**Quick Upskilling Suggestions:**")
        st.write("Focus next on: " + ", ".join(missing_skills[:3]))
    else:
        st.info("No missing skills detected for the top candidate.")

    st.markdown("### Ranked Candidates (Top 10)")
    key_columns = [
        'rank', 'resume_file', 'match_percentage', 'overall_score',
        'matched_skills_count', 'missing_skills_count',
        'technical_skills_score', 'tools_score', 'frameworks_score', 'soft_skills_score'
    ]
    display_df = results_df[[col for col in key_columns if col in results_df.columns]].copy()
    if 'resume_file' in display_df.columns:
        display_df['resume_file'] = display_df['resume_file'].apply(lambda x: Path(x).name)
    score_columns = [col for col in display_df.columns if 'score' in col.lower()]
    for col in score_columns:
        if col != 'match_percentage':
            display_df[col] = display_df[col].apply(lambda x: f"{x * 100:.1f}%")
    st.dataframe(display_df, width="stretch", height=320)

    st.markdown("### Download Resumes")
    selected_resumes = st.multiselect(
        "Choose resumes to download",
        options=results_df['resume_file'].tolist(),
        format_func=lambda x: f"Rank #{results_df[results_df['resume_file'] == x]['rank'].iloc[0]}: {Path(x).name} (Score: {results_df[results_df['resume_file'] == x]['match_percentage'].iloc[0]})",
        help="Select one or more resumes to download",
        key="main_resume_download_selector"
    )

    if selected_resumes:
        import zipfile
        import io

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for resume_path in selected_resumes:
                if resume_path in st.session_state.uploaded_file_objects:
                    file_name = Path(resume_path).name
                    rank = results_df[results_df['resume_file'] == resume_path]['rank'].iloc[0]
                    score = results_df[results_df['resume_file'] == resume_path]['overall_score'].iloc[0]
                    name_parts = file_name.rsplit('.', 1)
                    new_name = f"Rank{rank}_Score{score*100:.0f}_{name_parts[0]}.{name_parts[1]}"
                    zip_file.writestr(new_name, st.session_state.uploaded_file_objects[resume_path])
        zip_buffer.seek(0)

        st.download_button(
            label=f"⬇️ Download {len(selected_resumes)} Resume(s) as ZIP",
            data=zip_buffer,
            file_name="selected_resumes.zip",
            mime="application/zip",
            type="primary",
            width="stretch",
            key="main_zip_download_button"
        )
    else:
        st.info("Select candidates above to download their resumes.")

    qualified_df = results_df[results_df['overall_score'] >= 0.5]
    if len(qualified_df) > 0:
        import zipfile
        import io
        qualified_zip_buffer = io.BytesIO()
        with zipfile.ZipFile(qualified_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for _, row in qualified_df.iterrows():
                resume_path = row['resume_file']
                if resume_path in st.session_state.uploaded_file_objects:
                    file_name = Path(resume_path).name
                    rank = row['rank']
                    score = row['overall_score']
                    name_parts = file_name.rsplit('.', 1)
                    new_name = f"Rank{rank}_Score{score*100:.0f}_{name_parts[0]}.{name_parts[1]}"
                    zip_file.writestr(new_name, st.session_state.uploaded_file_objects[resume_path])
        qualified_zip_buffer.seek(0)
        st.download_button(
            label=f"✅ Download {len(qualified_df)} Qualified Resume(s) (≥50%)",
            data=qualified_zip_buffer,
            file_name="qualified_resumes.zip",
            mime="application/zip",
            width="stretch",
            key="download_qualified_resumes_button"
        )
    else:
        st.warning("No qualified candidates (score ≥50%) found")


def display_resume_details(resume_file: str, results_df: pd.DataFrame, jd_text: str):
    """
    Display detailed analysis for a single resume.
    
    Args:
        resume_file: Resume file path
        results_df: Results DataFrame
        jd_text: Job description text
    """
    # Get resume data
    filtered_df = results_df[results_df['resume_file'] == resume_file]
    
    if filtered_df.empty:
        st.error(f"Could not find data for resume: {Path(resume_file).name}")
        st.info("Please select a different resume from the dropdown.")
        return

    resume_data = filtered_df.iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Match Scores")
        
        # Create gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=resume_data['overall_score'] * 100,
            title={'text': "Overall Match"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 50], 'color': "lightyellow"},
                    {'range': [50, 70], 'color': "lightgreen"},
                    {'range': [70, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        st.plotly_chart(fig_gauge, width="stretch")
    
    with col2:
        st.subheader("📈 Category Breakdown")
        
        categories = {
            'Technical Skills': resume_data.get('technical_skills_score', 0),
            'Tools': resume_data.get('tools_score', 0),
            'Frameworks': resume_data.get('frameworks_score', 0),
            'Soft Skills': resume_data.get('soft_skills_score', 0)
        }
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=list(categories.values()),
            theta=list(categories.keys()),
            fill='toself',
            name='Match Score'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=False
        )
        
        st.plotly_chart(fig_radar, width="stretch")
    
    st.markdown("---")
    
    # Skills comparison for selected resume
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📚 All Skills")
        all_skills = resume_data.get('all_extracted_skills', '')
        if all_skills:
            skills_list = [s.strip() for s in all_skills.split(',') if s.strip()]
            if skills_list:
                st.caption(f"Total: {len(skills_list)}")
                skills_html = ' '.join([f'<span style="background-color: #d1ecf1; color: #0c5460; padding: 4px 8px; margin: 1px; border-radius: 12px; display: inline-block; font-size: 12px;">{skill}</span>' 
                                       for skill in skills_list])
                st.markdown(skills_html, unsafe_allow_html=True)
        else:
            st.info("No skills extracted")
    
    with col2:
        st.subheader("✅ Matched")
        matched_skills = resume_data.get('matched_skills', '')
        if matched_skills:
            skills_list = [s.strip() for s in matched_skills.split(',') if s.strip()]
            if skills_list:
                st.caption(f"Matched: {len(skills_list)}")
                skills_html = ' '.join([f'<span style="background-color: #d4edda; color: #155724; padding: 4px 8px; margin: 1px; border-radius: 12px; display: inline-block; font-size: 12px;">{skill}</span>' 
                                       for skill in skills_list])
                st.markdown(skills_html, unsafe_allow_html=True)
        else:
            st.info("No matches")
    
    with col3:
        st.subheader("❌ Missing")
    with col3:
        st.subheader("❌ Missing")
        missing_skills = resume_data.get('missing_skills', '')
        if missing_skills:
            skills_list = [s.strip() for s in missing_skills.split(',') if s.strip()]
            if skills_list:
                st.caption(f"Missing: {len(skills_list)}")
                skills_html = ' '.join([f'<span style="background-color: #f8d7da; color: #721c24; padding: 4px 8px; margin: 1px; border-radius: 12px; display: inline-block; font-size: 12px;">{skill}</span>' 
                                       for skill in skills_list])
                st.markdown(skills_html, unsafe_allow_html=True)
        else:
            st.info("None missing")


def display_welcome_message():
    """Display welcome message when no results are available."""
    st.info("""
    👋 **Welcome to the Resume Skill Recognition System!**
    
    **Getting Started:**
    1. Upload one or more resume files (PDF or DOCX) using the sidebar
    2. Paste the job description in the text area
    3. Click "Match Resumes" to find the best candidates
    
    **Features:**
    - 🤖 AI-powered skill extraction using NER and rule-based methods
    - 🎯 Advanced matching algorithms with weighted scoring
    - 📊 Comprehensive analytics and visualizations
    - 💾 Export results to CSV or JSON
    - 🔍 Detailed individual candidate analysis
    """)
    
    # System information
    with st.expander("ℹ️ System Information"):
        st.markdown(f"""
        **Configuration:**
        - NER Model: {config.get('skill_extraction.ner_model_name', 'N/A')}
        - Embedding Model: {config.get('feature_engineering.embedding_model', 'N/A')}
        - Similarity Metric: {config.get('matching.similarity_metric', 'cosine')}
        
        **Category Weights:**
        - Technical Skills: {config.get('matching.weights.technical_skills', 0.5) * 100}%
        - Tools: {config.get('matching.weights.tools', 0.3) * 100}%
        - Frameworks: {config.get('matching.weights.frameworks', 0.15) * 100}%
        - Soft Skills: {config.get('matching.weights.soft_skills', 0.05) * 100}%
        """)


def _render_skill_tags(skills: list, bg: str = "#d1ecf1", fg: str = "#0c5460"):
    """Utility to render skills as inline chips."""
    if not skills:
        st.info("No items to display")
        return
    skills_html = ' '.join([
        f'<span style="background-color: {bg}; color: {fg}; padding: 5px 10px; margin: 2px; border-radius: 15px; display: inline-block; font-size: 13px;">{skill}</span>'
        for skill in skills
    ])
    st.markdown(skills_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
