from typing import Callable

import numpy as np
from scipy.signal import lfilter
from scipy import signal
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

# experiment wide parameters
dt = 0.1

fs = 1000    # sample rate, Hz
T = 5.0         # seconds
n = int(T * fs)  # total number of samples
t = np.linspace(0, T, n, endpoint=False)
input_signal = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t)

gain_min = 1
gain_max = 10

loc_min = 0.5
loc_max = 1.5

sensit_min = 0.1
sensit_max = 1
sensit_step = 0.1

freqs = np.arange(0, 10**4, 1)

cutoff_min = 1
cutoff_max = 10**4


def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a


def butter_lowpass_filter(cutoff, fs, order=2):
    def filter(data):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y
    return filter


def highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a


def butter_highpass_filter(cutoff, fs, order=2):
    def filter(data):
        b, a = highpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y
    return filter


def get_sigmoid(loc: float, sensitivity: float, gain: float) -> Callable[[np.ndarray], np.ndarray]:
    def sigmoid(x):
        return gain / (1 + np.exp(-(x - loc) / sensitivity))
    return sigmoid


bode_parallel_unity = dbc.Card(
    dbc.CardBody(
        [
            dcc.Graph(id='bode_parallel_unity_graph'),
            dbc.FormGroup([
                dbc.Label('Cutoff1'),
                dcc.Slider(
                    id='filter1-parallel-cutoff-slider',
                    min=cutoff_min,
                    max=cutoff_max,
                    step=10,
                    value=cutoff_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
            dbc.FormGroup([
                dbc.Label('Type1'),
                dcc.Dropdown(
                    id='filter1-parallel-btype-dropdown',
                    options=[
                        {'label': 'lowpass', 'value': 'lowpass'},
                        {'label': 'highpass', 'value': 'highpass'},
                    ],
                    value='lowpass'
                )
            ]),
            dbc.FormGroup([
                dbc.Label('Cutoff2'),
                dcc.Slider(
                    id='filter2-parallel-cutoff-slider',
                    min=cutoff_min,
                    max=cutoff_max,
                    step=10,
                    value=cutoff_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
            dbc.FormGroup([
                dbc.Label('Type2'),
                dcc.Dropdown(
                    id='filter2-parallel-btype-dropdown',
                    options=[
                        {'label': 'lowpass', 'value': 'lowpass'},
                        {'label': 'highpass', 'value': 'highpass'},
                    ],
                    value='lowpass'
                )
            ]),
        ]
    )
)
bode_parallel_scaled = dbc.Card(
    dbc.CardBody(
        [
            dcc.Graph(id='bode_parallel_scaled_graph'),
            dbc.FormGroup([
                dbc.Label('Loc1'),
                dcc.Slider(
                    id='filter1-parallel-loc-slider',
                    min=loc_min,
                    max=loc_max,
                    step=0.1,
                    value=loc_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
            dbc.FormGroup([
                dbc.Label('Sensit1'),
                dcc.Slider(
                    id='filter1-parallel-sensit-slider',
                    min=sensit_min,
                    max=sensit_max,
                    step=sensit_step,
                    value=sensit_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
            dbc.FormGroup([
                dbc.Label('Gain1'),
                dcc.Slider(
                    id='filter1-parallel-gain-slider',
                    min=gain_min,
                    max=gain_max,
                    step=0.1,
                    value=gain_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
            dbc.FormGroup([
                dbc.Label('Loc2'),
                dcc.Slider(
                    id='filter2-parallel-loc-slider',
                    min=loc_min,
                    max=loc_max,
                    step=0.1,
                    value=loc_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
            dbc.FormGroup([
                dbc.Label('Sensit2'),
                dcc.Slider(
                    id='filter2-parallel-sensit-slider',
                    min=sensit_min,
                    max=sensit_max,
                    step=sensit_step,
                    value=sensit_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
            dbc.FormGroup([
                dbc.Label('Gain2'),
                dcc.Slider(
                    id='filter2-parallel-gain-slider',
                    min=gain_min,
                    max=gain_max,
                    step=0.1,
                    value=gain_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
        ]
    )
)
input_parallel = dbc.Card(
    dbc.CardBody(
        [
            dcc.Graph(id='input_parallel_graph'),
        ]
    )
)
output_parallel = dbc.Card(
    dbc.CardBody(
        [
            dcc.Graph(id='output_parallel_graph'),
        ]
    )
)
bode_single_unity = dbc.Card(
    dbc.CardBody(
        [
            dcc.Graph(id='bode_single_unity_graph'),
            dbc.FormGroup([
                dbc.Label('Cutoff'),
                dcc.Slider(
                    id='filter-single-cutoff-slider',
                    min=cutoff_min,
                    max=cutoff_max,
                    step=10,
                    value=cutoff_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
            dbc.FormGroup([
                dbc.Label('Type'),
                dcc.Dropdown(
                    id='filter-single-btype-dropdown',
                    options=[
                        {'label': 'lowpass', 'value': 'lowpass'},
                        {'label': 'highpass', 'value': 'highpass'},
                    ],
                    value='lowpass'
                )
            ]),
        ]
    )
)
bode_single_scaled = dbc.Card(
    dbc.CardBody(
        [
            dcc.Graph(id='bode_single_scaled_graph'),
            dbc.FormGroup([
                dbc.Label('Loc'),
                dcc.Slider(
                    id='filter-single-loc-slider',
                    min=loc_min,
                    max=loc_max,
                    step=0.1,
                    value=loc_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
            dbc.FormGroup([
                dbc.Label('Sensit'),
                dcc.Slider(
                    id='filter-single-sensit-slider',
                    min=sensit_min,
                    max=sensit_max,
                    step=sensit_step,
                    value=sensit_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
            dbc.FormGroup([
                dbc.Label('Gain'),
                dcc.Slider(
                    id='filter-single-gain-slider',
                    min=gain_min,
                    max=gain_max,
                    step=0.1,
                    value=gain_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
        ]
    )
)
input_single = dbc.Card(
    dbc.CardBody(
        [
            dcc.Graph(id='input_single_graph'),
        ]
    )
)
output_single = dbc.Card(
    dbc.CardBody(
        [
            dcc.Graph(id='output_single_graph'),
        ]
    )
)

# Build App
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
app.layout = html.Div([
    html.H1("Reducibility analysis"),
    html.Div([
        dbc.Row([
            dbc.Col([bode_parallel_unity], width=6), dbc.Col(
                [bode_parallel_scaled], width=6)
        ]),
        dbc.Row([
            dbc.Col([input_parallel], width=6), dbc.Col(
                [output_parallel], width=6)
        ]),
        dbc.Row([
            dbc.Col([bode_single_unity], width=6), dbc.Col(
                [bode_single_scaled], width=6)
        ]),
        dbc.Row([
            dbc.Col([input_single], width=6), dbc.Col([output_single], width=6)
        ]),
    ])
])


@app.callback(
    Output('bode_parallel_unity_graph', 'figure'),
    [
        Input('filter1-parallel-cutoff-slider', 'value'),
        Input('filter1-parallel-btype-dropdown', 'value'),
        Input('filter2-parallel-cutoff-slider', 'value'),
        Input('filter2-parallel-btype-dropdown', 'value'),
    ]
)
def draw_bode_parallel_unity(cutoff1, btype1, cutoff2, btype2):
    layout = go.Layout(
        title='Bode plot of parallel network',
        yaxis=dict(
            title='Gain'
        ),
        xaxis=dict(
            title='Freq'
        )
    )
    # compute filter 1
    filter1 = None
    if btype1 == 'highpass':
        b, a = signal.butter(N=2, Wn=cutoff1, btype='highpass', analog=True)
        filter1 = signal.TransferFunction(b, a)
    else:
        b, a = signal.butter(N=2, Wn=cutoff1, btype='lowpass', analog=True)
        filter1 = signal.TransferFunction(b, a)

    _, mag_butter1, _ = signal.bode(filter1, freqs)
    mag_butter1_abs = 10**(mag_butter1/20)

    # compute filter 2
    filter2 = None
    if btype2 == 'highpass':
        b, a = signal.butter(N=2, Wn=cutoff2, btype='highpass', analog=True)
        filter2 = signal.TransferFunction(b, a)
    else:
        b, a = signal.butter(N=2, Wn=cutoff2, btype='lowpass', analog=True)
        filter2 = signal.TransferFunction(b, a)

    _, mag_butter2, _ = signal.bode(filter2, freqs)
    mag_butter2_abs = 10**(mag_butter2/20)

    trace_filter1 = go.Scatter(
        x=freqs,
        y=mag_butter1_abs,
        mode='lines',
        line=dict(
            shape='spline'
        ),
        name='Filter1'
    )
    trace_filter2 = go.Scatter(
        x=freqs,
        y=mag_butter2_abs,
        mode='lines',
        line=dict(
            shape='spline'
        ),
        name='Filter2'
    )

    fig = go.Figure(data=[trace_filter1, trace_filter2], layout=layout)
    fig.update_xaxes(type="log")
    return fig


@app.callback(
    Output('bode_parallel_scaled_graph', 'figure'),
    [
        Input('filter1-parallel-cutoff-slider', 'value'),
        Input('filter1-parallel-btype-dropdown', 'value'),
        Input('filter2-parallel-cutoff-slider', 'value'),
        Input('filter2-parallel-btype-dropdown', 'value'),
        Input('filter1-parallel-loc-slider', 'value'),
        Input('filter1-parallel-sensit-slider', 'value'),
        Input('filter1-parallel-gain-slider', 'value'),
        Input('filter2-parallel-loc-slider', 'value'),
        Input('filter2-parallel-sensit-slider', 'value'),
        Input('filter2-parallel-gain-slider', 'value'),
    ]
)
def draw_bode_parallel_scaled(cutoff1, btype1, cutoff2, btype2, loc1, sensit1, gain1, loc2, sensit2, gain2):
    layout = go.Layout(
        title='Bode plot of parallel network (sigmoid scaled)',
        yaxis=dict(
            title='Gain'
        ),
        xaxis=dict(
            title='Freq'
        )
    )
    # compute filter 1
    filter1 = None
    if btype1 == 'highpass':
        b, a = signal.butter(N=2, Wn=cutoff1,
                             btype='highpass', analog=True)
        filter1 = signal.TransferFunction(b, a)
    else:
        b, a = signal.butter(N=2, Wn=cutoff1,
                             btype='lowpass', analog=True)
        filter1 = signal.TransferFunction(b, a)

    _, mag_butter1, _ = signal.bode(filter1, freqs)
    mag_butter1_abs = 10**(mag_butter1/20)

    # compute filter 2
    filter2 = None
    if btype2 == 'highpass':
        b, a = signal.butter(N=2, Wn=cutoff2,
                             btype='highpass', analog=True)
        filter2 = signal.TransferFunction(b, a)
    else:
        b, a = signal.butter(N=2, Wn=cutoff2,
                             btype='lowpass', analog=True)
        filter2 = signal.TransferFunction(b, a)

    _, mag_butter2, _ = signal.bode(filter2, freqs)
    mag_butter2_abs = 10**(mag_butter2/20)

    sig1 = get_sigmoid(loc1, sensit1, gain1)
    sig2 = get_sigmoid(loc2, sensit2, gain2)

    trace_filter1 = go.Scatter(
        x=freqs,
        y=sig1(mag_butter1_abs),
        mode='lines',
        line=dict(
            shape='spline'
        ),
        name='Filter1'
    )
    trace_filter2 = go.Scatter(
        x=freqs,
        y=sig2(mag_butter2_abs),
        mode='lines',
        line=dict(
            shape='spline'
        ),
        name='Filter2'
    )

    fig = go.Figure(data=[trace_filter1, trace_filter2], layout=layout)
    fig.update_xaxes(type="log")
    return fig


@app.callback(
    Output('input_parallel_graph', 'figure'),
    [
        Input('filter1-parallel-cutoff-slider', 'value'),
    ]
)
def draw_input_parallel(placeholder):
    layout = go.Layout(
        title='Input Signal',
        yaxis=dict(
            title='y'
        ),
        xaxis=dict(
            title='t(s)'
        )
    )

    trace = go.Scatter(
        x=t,
        y=input_signal,
        mode='lines',
        line=dict(
            shape='spline'
        ),
        name='signal'
    )

    fig = go.Figure(data=[trace], layout=layout)
    return fig


@app.callback(
    Output('output_parallel_graph', 'figure'),
    [
        Input('filter1-parallel-cutoff-slider', 'value'),
        Input('filter1-parallel-btype-dropdown', 'value'),
        Input('filter2-parallel-cutoff-slider', 'value'),
        Input('filter2-parallel-btype-dropdown', 'value'),
        Input('filter1-parallel-loc-slider', 'value'),
        Input('filter1-parallel-sensit-slider', 'value'),
        Input('filter1-parallel-gain-slider', 'value'),
        Input('filter2-parallel-loc-slider', 'value'),
        Input('filter2-parallel-sensit-slider', 'value'),
        Input('filter2-parallel-gain-slider', 'value'),
    ]
)
def draw_output_parallel(cutoff1, btype1, cutoff2, btype2, loc1, sensit1, gain1, loc2, sensit2, gain2):
    layout = go.Layout(
        title='Output signal filtered',
        yaxis=dict(
            title='y'
        ),
        xaxis=dict(
            title='t(s)'
        )
    )
    # compute filter 1
    filter1 = None
    if btype1 == 'highpass':
        filter1 = butter_highpass_filter(cutoff1, fs)
    else:
        filter1 = butter_lowpass_filter(cutoff1, fs)

    # compute filter 2
    filter2 = None
    if btype2 == 'highpass':
        filter2 = butter_highpass_filter(cutoff2, fs)
    else:
        filter2 = butter_lowpass_filter(cutoff2, fs)

    y_filtered1 = filter1(input_signal)
    sig1 = get_sigmoid(loc1, sensit1, gain1)

    y_filtered2 = filter2(input_signal)
    sig2 = get_sigmoid(loc2, sensit2, gain2)

    trace_filter1 = go.Scatter(
        x=freqs,
        y=(sig1(y_filtered1) + sig2(y_filtered2)),
        mode='lines',
        line=dict(
            shape='spline'
        ),
        name='Filter1'
    )

    fig = go.Figure(data=[trace_filter1], layout=layout)
    return fig


@app.callback(
    Output('bode_single_unity_graph', 'figure'),
    [
        Input('filter-single-cutoff-slider', 'value'),
        Input('filter-single-btype-dropdown', 'value'),
    ]
)
def draw_bode_single_unity(cutoff, btype):
    layout = go.Layout(
        title='Bode plot of single network',
        yaxis=dict(
            title='Gain'
        ),
        xaxis=dict(
            title='Freq'
        )
    )
    # compute filter 1
    filter1 = None
    if btype == 'highpass':
        b, a = signal.butter(N=2, Wn=cutoff,
                             btype='highpass', analog=True)
        filter1 = signal.TransferFunction(b, a)
    else:
        b, a = signal.butter(N=2, Wn=cutoff,
                             btype='lowpass', analog=True)
        filter1 = signal.TransferFunction(b, a)

    _, mag_butter1, _ = signal.bode(filter1, freqs)
    mag_butter1_abs = 10**(mag_butter1/20)

    trace_filter1 = go.Scatter(
        x=freqs,
        y=mag_butter1_abs,
        mode='lines',
        line=dict(
            shape='spline'
        ),
        name='Filter1'
    )

    fig = go.Figure(data=[trace_filter1], layout=layout)
    fig.update_xaxes(type="log")
    return fig


@app.callback(
    Output('bode_single_scaled_graph', 'figure'),
    [
        Input('filter-single-cutoff-slider', 'value'),
        Input('filter-single-btype-dropdown', 'value'),
        Input('filter-single-loc-slider', 'value'),
        Input('filter-single-sensit-slider', 'value'),
        Input('filter-single-gain-slider', 'value'),
    ]
)
def draw_bode_single_scaled(cutoff, btype, loc, sensit, gain):
    layout = go.Layout(
        title='Bode plot of single network (sigmoid scaled)',
        yaxis=dict(
            title='Gain'
        ),
        xaxis=dict(
            title='Freq'
        )
    )
    # compute filter 1
    filter1 = None
    if btype == 'highpass':
        b, a = signal.butter(N=2, Wn=cutoff,
                             btype='highpass', analog=True)
        filter1 = signal.TransferFunction(b, a)
    else:
        b, a = signal.butter(N=2, Wn=cutoff,
                             btype='lowpass', analog=True)
        filter1 = signal.TransferFunction(b, a)

    _, mag_butter1, _ = signal.bode(filter1, freqs)
    mag_butter1_abs = 10**(mag_butter1/20)

    sig = get_sigmoid(loc, sensit, gain)

    trace_filter1 = go.Scatter(
        x=freqs,
        y=sig(mag_butter1_abs),
        mode='lines',
        line=dict(
            shape='spline'
        ),
        name='Filter1'
    )

    fig = go.Figure(data=[trace_filter1], layout=layout)
    fig.update_xaxes(type="log")
    return fig


@app.callback(
    Output('input_single_graph', 'figure'),
    [
        Input('filter-single-cutoff-slider', 'value'),
    ]
)
def draw_input_single(placeholder):
    layout = go.Layout(
        title='Input Signal',
        yaxis=dict(
            title='y'
        ),
        xaxis=dict(
            title='t(s)'
        )
    )

    trace = go.Scatter(
        x=t,
        y=input_signal,
        mode='lines',
        line=dict(
            shape='spline'
        ),
        name='signal'
    )

    fig = go.Figure(data=[trace], layout=layout)
    return fig


@app.callback(
    Output('output_single_graph', 'figure'),
    [
        Input('filter-single-cutoff-slider', 'value'),
        Input('filter-single-btype-dropdown', 'value'),
        Input('filter-single-loc-slider', 'value'),
        Input('filter-single-sensit-slider', 'value'),
        Input('filter-single-gain-slider', 'value'),
    ]
)
def draw_output_single(cutoff, btype, loc, sensit, gain):
    layout = go.Layout(
        title='Output signal filtered',
        yaxis=dict(
            title='y'
        ),
        xaxis=dict(
            title='t(s)'
        )
    )
    # compute filter 1
    filter1 = None
    if btype == 'highpass':
        filter1 = butter_highpass_filter(cutoff, fs)
    else:
        filter1 = butter_lowpass_filter(cutoff, fs)

    y_filtered = filter1(input_signal)
    sig = get_sigmoid(loc, sensit, gain)

    trace_filter1 = go.Scatter(
        x=freqs,
        y=sig(y_filtered),
        mode='lines',
        line=dict(
            shape='spline'
        ),
        name='Filter1'
    )

    fig = go.Figure(data=[trace_filter1], layout=layout)
    return fig


if __name__ == '__main__':
    # Run app and display result inline in the notebook
    app.run_server(use_reloader=True)
