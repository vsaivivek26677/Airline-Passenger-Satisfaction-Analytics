import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from functools import lru_cache
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

TEMPLATE = 'plotly_white'
COLOR_SATISFIED = '#2E86AB'
COLOR_DISSATISFIED = '#E63946'
ACCENT = '#1a1a2e'
COLOR_SEQUENCE = px.colors.qualitative.Set2
SATISFACTION_MAP = {'satisfied': COLOR_SATISFIED, 'neutral or dissatisfied': COLOR_DISSATISFIED}

df = pd.read_csv('train.csv')
df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')])

service_cols = [
    'Inflight wifi service',
    'Departure/Arrival time convenient',
    'Ease of Online booking',
    'Gate location',
    'Food and drink',
    'Online boarding',
    'Seat comfort',
    'Inflight entertainment',
    'On-board service',
    'Leg room service',
    'Baggage handling',
    'Checkin service',
    'Inflight service',
    'Cleanliness',
]

age_bins = [0, 18, 30, 45, 60, 200]
age_labels = ['0-17', '18-29', '30-44', '45-59', '60+']
df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

distance_bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 100000]
distance_labels = ['0-499', '500-999', '1000-1499', '1500-1999', '2000-2499', '2500-2999', '3000+']
df['distance_bracket'] = pd.cut(df['Flight Distance'], bins=distance_bins, labels=distance_labels, right=False)


def style_fig(fig, height=420):
    fig.update_layout(
        template=TEMPLATE,
        height=height,
        margin=dict(l=50, r=30, t=60, b=40),
        font=dict(family='Segoe UI, Arial', size=12, color=ACCENT),
        title_font=dict(size=16, color=ACCENT),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def fig_age_distribution():
    fig = px.histogram(
        df, x='Age', color='satisfaction', nbins=40, barmode='overlay', opacity=0.7,
        color_discrete_map=SATISFACTION_MAP, title='Age Distribution by Satisfaction',
    )
    return style_fig(fig)


def fig_class_donut():
    counts = df['Class'].value_counts().reset_index()
    counts.columns = ['Class', 'count']
    fig = px.pie(
        counts, names='Class', values='count', hole=0.55, color_discrete_sequence=COLOR_SEQUENCE,
        title='Passenger Share by Travel Class',
    )
    fig.update_traces(textinfo='percent+label')
    return style_fig(fig)


def fig_customer_type_bar():
    counts = df['Customer Type'].value_counts().reset_index()
    counts.columns = ['Customer Type', 'count']
    fig = px.bar(
        counts, x='Customer Type', y='count', color='Customer Type',
        color_discrete_sequence=COLOR_SEQUENCE, title='Passenger Count by Customer Type', text='count',
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False)
    return style_fig(fig)


def fig_distance_box():
    fig = px.box(
        df, y='Flight Distance', points='outliers', color_discrete_sequence=[ACCENT],
        title='Flight Distance Distribution',
    )
    return style_fig(fig)


def fig_arrival_delay_hist():
    fig = px.histogram(
        df, x='Arrival Delay in Minutes', nbins=60, color_discrete_sequence=[COLOR_DISSATISFIED],
        title='Arrival Delay Distribution (Log Scale)',
    )
    fig.update_yaxes(type='log', title='Passenger Count (log scale)')
    return style_fig(fig)


def fig_satisfaction_by_class_stacked():
    counts = df.groupby(['Class', 'satisfaction']).size().reset_index(name='count')
    counts['pct'] = counts.groupby('Class')['count'].transform(lambda s: 100 * s / s.sum())
    fig = px.bar(
        counts, x='Class', y='pct', color='satisfaction', barmode='stack',
        color_discrete_map=SATISFACTION_MAP, title='Satisfaction Share by Travel Class',
        text=counts['pct'].round(1).astype(str) + '%',
    )
    fig.update_yaxes(title='Percent of Passengers')
    return style_fig(fig)


def fig_service_avg_ranked():
    avgs = df[service_cols].mean().sort_values(ascending=True).reset_index()
    avgs.columns = ['service', 'avg_rating']
    fig = px.bar(
        avgs, x='avg_rating', y='service', orientation='h', color='avg_rating',
        color_continuous_scale='Blues', title='Average Rating by Service',
    )
    fig.update_layout(coloraxis_showscale=False, yaxis_title='')
    return style_fig(fig, height=520)


def fig_age_violin_travel_type():
    fig = px.violin(
        df, x='Type of Travel', y='Age', color='Type of Travel', box=True, points=False,
        color_discrete_sequence=COLOR_SEQUENCE, title='Age Distribution by Travel Purpose',
    )
    fig.update_layout(showlegend=False)
    return style_fig(fig)


def fig_distance_box_by_satisfaction():
    fig = px.box(
        df, x='satisfaction', y='Flight Distance', color='satisfaction',
        color_discrete_map=SATISFACTION_MAP, title='Flight Distance by Satisfaction',
    )
    fig.update_layout(showlegend=False)
    return style_fig(fig)


def fig_time_convenient_by_delay_bucket():
    bins = [-1, 0, 15, 60, 1000000]
    labels = ['On time', '1-15 min', '15-60 min', '60+ min']
    temp = df.copy()
    temp['delay_bucket'] = pd.cut(temp['Departure Delay in Minutes'], bins=bins, labels=labels)
    avgs = temp.groupby('delay_bucket', observed=True)['Departure/Arrival time convenient'].mean().reset_index()
    fig = px.bar(
        avgs, x='delay_bucket', y='Departure/Arrival time convenient', color='delay_bucket',
        color_discrete_sequence=COLOR_SEQUENCE, title='Time Convenience Rating by Departure Delay Bucket',
    )
    fig.update_layout(showlegend=False, xaxis_title='Departure Delay Bucket')
    return style_fig(fig)


def fig_service_corr_heatmap():
    corr = df[service_cols].corr()
    fig = px.imshow(
        corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
        title='Correlation Between Service Ratings',
    )
    return style_fig(fig, height=580)


def fig_delay_scatter_trend():
    sample = df.dropna(subset=['Arrival Delay in Minutes']).sample(n=3000, random_state=42)
    fig = px.scatter(
        sample, x='Departure Delay in Minutes', y='Arrival Delay in Minutes',
        trendline='ols', trendline_color_override=COLOR_DISSATISFIED,
        color_discrete_sequence=[ACCENT], opacity=0.35,
        title='Departure Delay vs Arrival Delay',
    )
    return style_fig(fig)


def fig_satisfaction_heatmap_age_distance():
    temp = df.copy()
    temp['satisfied_flag'] = (temp['satisfaction'] == 'satisfied').astype(int)
    pivot = temp.pivot_table(
        index='age_group', columns='distance_bracket', values='satisfied_flag', aggfunc='mean', observed=True,
    ) * 100
    fig = px.imshow(
        pivot, text_auto='.0f', color_continuous_scale='Blues', aspect='auto',
        title='Satisfaction Rate (%) by Age Group and Flight Distance',
    )
    fig.update_layout(xaxis_title='Flight Distance Bracket', yaxis_title='Age Group')
    return style_fig(fig, height=480)


def fig_service_clustermap():
    corr = df[service_cols].corr()
    dist = 1 - corr
    condensed = squareform(dist.values, checks=False)
    z = linkage(condensed, method='average')
    dendro = dendrogram(z, labels=corr.columns.tolist(), no_plot=True)
    order = dendro['ivl']
    ordered_corr = corr.loc[order, order]
    fig = px.imshow(
        ordered_corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
        title='Clustered Service Rating Correlation',
    )
    return style_fig(fig, height=580)


def fig_age_delay_density():
    sample = df.dropna(subset=['Arrival Delay in Minutes'])
    fig = px.density_heatmap(
        sample, x='Age', y='Arrival Delay in Minutes', nbinsx=40, nbinsy=40,
        color_continuous_scale='Blues', title='Passenger Density: Age vs Arrival Delay',
    )
    fig.update_yaxes(range=[0, 200])
    return style_fig(fig)


def fig_service_radar_class():
    biz = df[df['Class'] == 'Business'][service_cols].mean()
    eco = df[df['Class'] == 'Eco'][service_cols].mean()
    categories = service_cols + [service_cols[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(biz) + [biz.iloc[0]], theta=categories, fill='toself', name='Business', line_color=COLOR_SATISFIED,
    ))
    fig.add_trace(go.Scatterpolar(
        r=list(eco) + [eco.iloc[0]], theta=categories, fill='toself', name='Eco', line_color=COLOR_DISSATISFIED,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])), title='Service Ratings: Business vs Eco',
    )
    return style_fig(fig, height=540)


def fig_wifi_likert_diverging():
    counts = df.groupby(['Customer Type', 'Inflight wifi service']).size().reset_index(name='count')
    counts['pct'] = counts.groupby('Customer Type')['count'].transform(lambda s: 100 * s / s.sum())
    pivot = counts.pivot(index='Customer Type', columns='Inflight wifi service', values='pct').fillna(0)
    for rating in range(1, 6):
        if rating not in pivot.columns:
            pivot[rating] = 0.0
    colors = {1: '#B23A48', 2: '#E8998D', 3: '#D8D8D8', 4: '#8FBFE0', 5: '#2E86AB'}
    fig = go.Figure()
    fig.add_trace(go.Bar(y=pivot.index, x=pivot[1], base=-(pivot[1] + pivot[2]), orientation='h', name='Rating 1', marker_color=colors[1]))
    fig.add_trace(go.Bar(y=pivot.index, x=pivot[2], base=-pivot[2], orientation='h', name='Rating 2', marker_color=colors[2]))
    fig.add_trace(go.Bar(y=pivot.index, x=pivot[3], base=-pivot[3] / 2, orientation='h', name='Rating 3', marker_color=colors[3]))
    fig.add_trace(go.Bar(y=pivot.index, x=pivot[4], base=0, orientation='h', name='Rating 4', marker_color=colors[4]))
    fig.add_trace(go.Bar(y=pivot.index, x=pivot[5], base=pivot[4], orientation='h', name='Rating 5', marker_color=colors[5]))
    fig.update_layout(
        barmode='overlay', title='Inflight Wifi Rating Breakdown by Customer Type', xaxis_title='Percent of Passengers',
    )
    return style_fig(fig)


def fig_seat_comfort_moving_avg():
    temp = df.sort_values('Flight Distance')[['Flight Distance', 'Seat comfort']].copy()
    temp['moving_avg'] = temp['Seat comfort'].rolling(window=500, min_periods=100).mean()
    fig = px.line(
        temp, x='Flight Distance', y='moving_avg', color_discrete_sequence=[ACCENT],
        title='Seat Comfort Moving Average vs Flight Distance',
    )
    fig.update_yaxes(title='Seat Comfort (500-flight moving average)')
    return style_fig(fig)


def fig_legroom_strip_longhaul():
    long_haul = df[df['Flight Distance'] > 2500]
    fig = px.strip(
        long_haul, x='Leg room service', y='Flight Distance', color='Class',
        color_discrete_sequence=COLOR_SEQUENCE, title='Leg Room Rating for Long-Haul Flights (over 2500 miles)',
    )
    return style_fig(fig)


def fig_pairplot():
    sample = df.sample(n=2000, random_state=42)
    fig = px.scatter_matrix(
        sample, dimensions=['Age', 'Flight Distance', 'Inflight entertainment'], color='satisfaction',
        color_discrete_map=SATISFACTION_MAP, title='Pairwise Relationships: Age, Distance, Entertainment',
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=4, opacity=0.5))
    return style_fig(fig, height=620)


def fig_entertainment_facet():
    fig = px.histogram(
        df, x='Inflight entertainment', facet_row='Class', facet_col='satisfaction', color='satisfaction',
        nbins=5, color_discrete_map=SATISFACTION_MAP,
        title='Inflight Entertainment Ratings by Class and Satisfaction',
    )
    fig.update_layout(showlegend=False)
    return style_fig(fig, height=680)


def fig_satisfaction_prob_vs_delay():
    temp = df.dropna(subset=['Arrival Delay in Minutes']).copy()
    temp = temp[temp['Arrival Delay in Minutes'] <= 180]
    temp['delay_bin'] = (temp['Arrival Delay in Minutes'] // 10 * 10).astype(int)
    prob = temp.groupby('delay_bin')['satisfaction'].apply(
        lambda s: 100 * (s == 'satisfied').mean()
    ).reset_index(name='pct_satisfied')
    fig = px.line(
        prob, x='delay_bin', y='pct_satisfied', markers=True, color_discrete_sequence=[ACCENT],
        title='Satisfaction Probability vs Arrival Delay (10-Minute Bins)',
    )
    fig.update_xaxes(title='Arrival Delay (minutes)')
    fig.update_yaxes(title='Percent Satisfied')
    return style_fig(fig)


def fig_parallel_coordinates():
    sample = df.sample(n=800, random_state=42).copy()
    sample['satisfaction_flag'] = (sample['satisfaction'] == 'satisfied').astype(int)
    dims = ['Inflight wifi service', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Cleanliness']
    fig = px.parallel_coordinates(
        sample, dimensions=dims, color='satisfaction_flag',
        color_continuous_scale=[COLOR_DISSATISFIED, COLOR_SATISFIED],
        title='Rating Profiles: Satisfied vs Dissatisfied Passengers',
    )
    return style_fig(fig, height=480)


def fig_age_cumulative_area():
    counts = df.groupby(['age_group', 'Type of Travel'], observed=True).size().reset_index(name='count')
    counts['cumulative'] = counts.groupby('Type of Travel')['count'].cumsum()
    fig = px.area(
        counts, x='age_group', y='cumulative', color='Type of Travel',
        color_discrete_sequence=COLOR_SEQUENCE, title='Cumulative Passenger Count by Age Group and Travel Purpose',
    )
    return style_fig(fig)


def fig_service_waterfall():
    avgs = df[service_cols].mean()
    overall_avg = avgs.mean()
    diffs = (avgs - overall_avg).sort_values(ascending=False)
    fig = go.Figure(go.Waterfall(
        x=diffs.index.tolist(),
        y=diffs.values.tolist(),
        measure=['relative'] * len(diffs),
        connector=dict(line=dict(color='rgb(180,180,180)')),
        increasing=dict(marker=dict(color=COLOR_SATISFIED)),
        decreasing=dict(marker=dict(color=COLOR_DISSATISFIED)),
    ))
    fig.update_layout(title=f'Service Rating Deviation from Global Average ({overall_avg:.2f})')
    return style_fig(fig, height=480)


def kpi_card(label, value, color):
    return dbc.Card(
        dbc.CardBody([
            html.Div(label, className='kpi-label'),
            html.Div(value, className='kpi-value', style={'color': color}),
        ]),
        className='kpi-card',
    )


def chart_card(fig, width=6):
    return dbc.Col(dcc.Graph(figure=fig, config={'displaylogo': False}), width=12, lg=width, className='mb-4')


total_passengers = len(df)
pct_satisfied = round(100 * (df['satisfaction'] == 'satisfied').mean(), 1)
avg_age = round(df['Age'].mean(), 1)
avg_distance = round(df['Flight Distance'].mean(), 0)

kpi_row = dbc.Row([
    dbc.Col(kpi_card('Total Passengers', f'{total_passengers:,}', ACCENT), width=6, lg=3),
    dbc.Col(kpi_card('Satisfied', f'{pct_satisfied}%', COLOR_SATISFIED), width=6, lg=3),
    dbc.Col(kpi_card('Average Age', f'{avg_age}', ACCENT), width=6, lg=3),
    dbc.Col(kpi_card('Average Flight Distance', f'{int(avg_distance):,} mi', ACCENT), width=6, lg=3),
], className='mb-4 g-3')

@lru_cache(maxsize=1)
def build_tab_distributions():
    return dbc.Row([
        chart_card(fig_age_distribution()),
        chart_card(fig_class_donut()),
        chart_card(fig_customer_type_bar()),
        chart_card(fig_distance_box()),
        chart_card(fig_arrival_delay_hist(), width=12),
    ])


@lru_cache(maxsize=1)
def build_tab_bivariate():
    return dbc.Row([
        chart_card(fig_satisfaction_by_class_stacked()),
        chart_card(fig_service_avg_ranked()),
        chart_card(fig_age_violin_travel_type()),
        chart_card(fig_distance_box_by_satisfaction()),
        chart_card(fig_time_convenient_by_delay_bucket(), width=12),
    ])


@lru_cache(maxsize=1)
def build_tab_correlations():
    return dbc.Row([
        chart_card(fig_service_corr_heatmap(), width=12),
        chart_card(fig_delay_scatter_trend()),
        chart_card(fig_satisfaction_heatmap_age_distance()),
        chart_card(fig_service_clustermap(), width=12),
        chart_card(fig_age_delay_density(), width=12),
    ])


@lru_cache(maxsize=1)
def build_tab_deep_dives():
    return dbc.Row([
        chart_card(fig_service_radar_class()),
        chart_card(fig_wifi_likert_diverging()),
        chart_card(fig_seat_comfort_moving_avg()),
        chart_card(fig_legroom_strip_longhaul()),
        chart_card(fig_pairplot(), width=12),
    ])


@lru_cache(maxsize=1)
def build_tab_advanced():
    return dbc.Row([
        chart_card(fig_entertainment_facet(), width=12),
        chart_card(fig_satisfaction_prob_vs_delay()),
        chart_card(fig_age_cumulative_area()),
        chart_card(fig_parallel_coordinates(), width=12),
        chart_card(fig_service_waterfall(), width=12),
    ])


TAB_BUILDERS = {
    'tab-distributions': build_tab_distributions,
    'tab-bivariate': build_tab_bivariate,
    'tab-correlations': build_tab_correlations,
    'tab-deep-dives': build_tab_deep_dives,
    'tab-advanced': build_tab_advanced,
}

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = 'Airline Passenger Satisfaction Dashboard'
server = app.server

app.layout = dbc.Container([
    html.Div([
        html.H1('Airline Passenger Satisfaction Dashboard', className='app-title'),
        html.P('25 interactive Plotly visualizations covering demographics, satisfaction, service ratings, and delays', className='app-subtitle'),
    ], className='app-header'),
    kpi_row,
    dbc.Tabs([
        dbc.Tab(label='Distributions', tab_id='tab-distributions'),
        dbc.Tab(label='Categorical & Bivariate', tab_id='tab-bivariate'),
        dbc.Tab(label='Correlations & Heatmaps', tab_id='tab-correlations'),
        dbc.Tab(label='Service Deep Dives', tab_id='tab-deep-dives'),
        dbc.Tab(label='Advanced & Faceted', tab_id='tab-advanced'),
    ], id='tabs', active_tab='tab-distributions', className='mb-4'),
    dcc.Loading(html.Div(id='tab-content'), type='circle', color=ACCENT),
], fluid=True, className='app-container')


@app.callback(Output('tab-content', 'children'), Input('tabs', 'active_tab'))
def render_tab(active_tab):
    builder = TAB_BUILDERS.get(active_tab, build_tab_distributions)
    return builder()


app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { background-color: #f4f6fa; }
            .app-container { padding: 24px 32px 40px 32px; }
            .app-header { padding: 8px 0 20px 0; }
            .app-title { font-weight: 700; color: #1a1a2e; margin-bottom: 4px; }
            .app-subtitle { color: #5a5f73; font-size: 15px; }
            .kpi-card { border: none; border-radius: 12px; box-shadow: 0 2px 10px rgba(26,26,46,0.08); }
            .kpi-label { font-size: 13px; color: #5a5f73; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }
            .kpi-value { font-size: 28px; font-weight: 700; margin-top: 4px; }
            .nav-tabs .nav-link.active { font-weight: 600; color: #1a1a2e; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=8050)
