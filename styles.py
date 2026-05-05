# styles.py
import streamlit as st

def apply_custom_css():
    st.markdown("""
    <style>
    @keyframes fadeSlideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(46, 204, 113, 0); }
        100% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0); }
    }
    .animate-card {
        animation: fadeSlideUp 0.6s ease-out;
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        background: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .animate-card:hover {
        transform: translateY(-5px);
    }
    .result-highlight {
        animation: pulse 1.5s infinite;
        border-radius: 12px;
        background: #f0fff4;
        padding: 0.8rem;
    }
    .center {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)