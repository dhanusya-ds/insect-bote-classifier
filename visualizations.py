# visualizations.py
import numpy as np
import plotly.graph_objects as go

def create_radar_chart(probabilities, class_names):
    """Radar chart showing probabilities for all classes (sorted descending)."""
    sorted_indices = np.argsort(probabilities)[::-1]
    display_names = [class_names[i].replace('_', ' ').title() for i in sorted_indices]
    display_probs = probabilities[sorted_indices]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=display_probs,
        theta=display_names,
        fill='toself',
        name='Probability',
        line_color='#2ecc71',
        fillcolor='rgba(46, 204, 113, 0.3)',
        marker=dict(size=6, color='#27ae60')
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.0%',
                tickfont=dict(size=8)
            ),
            angularaxis=dict(
                tickfont=dict(size=9),
                tickangle=45
            )
        ),
        title=dict(
            text="Probability Distribution Across All Insect Types",
            font=dict(size=14),
            x=0.5
        ),
        height=500,
        margin=dict(l=80, r=80, t=60, b=80),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,0.3)'
    )
    return fig

def create_gauge_chart(confidence, insect_class):
    """Gauge chart for confidence level."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        title={'text': f"Confidence for {insect_class.replace('_', ' ').title()}", 'font': {'size': 14}},
        delta={'reference': 80, 'increasing': {'color': "#2ecc71"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#2ecc71"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#f8d7da'},
                {'range': [50, 75], 'color': '#fff3cd'},
                {'range': [75, 100], 'color': '#d4edda'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig