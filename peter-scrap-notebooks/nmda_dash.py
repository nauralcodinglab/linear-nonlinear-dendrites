from typing import Callable

import numpy as np
import matplotlib.gridspec as gs
from ezephys import stimtools as st
from ezephys import pltools
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import plotly.express as px

from jupyter_dash import JupyterDash
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

# experiment wide parameters
dt = 0.1

# timescales
ca_tau = 40 # ms

# ranges for sigmoid parameters
na_loc_min = 0.1
na_loc_max = 2
na_sensit_min = 0.1
na_sensit_max = 1
na_gain_min = 1
na_gain_max = 6

ca_loc_min = 0.1
ca_loc_max = 2
ca_sensit_min = 0.1
ca_sensit_max = 1
ca_gain_min = 1
ca_gain_max = 6

nmda_loc_min = 0.1
nmda_loc_max = 2
nmda_sensit_min = 0.1
nmda_sensit_max = 1
nmda_gain_min = 1
nmda_gain_max = 6

sensit_step = 0.01


def DesignExponentialFilter(tau_ms, filter_length_ms):
    t = np.arange(0, filter_length_ms, dt)
    IRF_filter = np.exp(-t / tau_ms)
    IRF_filter = IRF_filter/sum(IRF_filter)
    IRF_filter[0] = 0
    return IRF_filter, t


def get_sigmoid(loc: float, sensitivity: float, gain: float) -> Callable[[np.ndarray], np.ndarray]:
    def sigmoid(x):
        return gain / (1 + np.exp(-(x - loc) / sensitivity))
    return sigmoid


def get_filter(kernel: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Get a function that convolves the kernel with an input."""
    return lambda input_: lfilter(kernel, 1, input_, axis=-1, zi=None)


def get_linear_nonlinear_model(
        membrane_kernel: np.ndarray, nonlinear_kernel: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    membrane_filter = get_filter(membrane_kernel)
    sodium_filter = get_filter(nonlinear_kernel)

    def linear_nonlinear_model(dendritic_input: np.ndarray, loc: float, sensit: float, gain: float) -> [np.ndarray, np.ndarray]:
        assert np.ndim(dendritic_input) == 1
        sigmoid = get_sigmoid(loc, sensit, gain)

        return [sigmoid(sodium_filter(dendritic_input)) + membrane_filter(dendritic_input), membrane_filter(dendritic_input)]

    return linear_nonlinear_model


def delayed_pulse(delay_ms: float, peak_amplitude: float, total_duration: float = 500.) -> st.ConvolvedStimulus:
    """Create ConvolvedStimulus containing a synaptic pulse with a delayed start."""
    synaptic_kernel = st.BiexponentialSynapticKernel(
        peak_amplitude, 0.1, 5, duration=25., dt=dt)

    pulse_time = np.zeros(int(total_duration / dt - 0.5))
    pulse_time[int(delay_ms / dt - 0.5)] = 1.
    pulse = st.ConvolvedStimulus(0, synaptic_kernel)
    pulse.generate(pulse_time, dt)
    return pulse


# na kernel
na_membrane_kernel, _ = DesignExponentialFilter(
    tau_ms=5, filter_length_ms=100)
na_nl_kernel, _ = DesignExponentialFilter(tau_ms=5, filter_length_ms=100)

na_kernel = get_linear_nonlinear_model(na_membrane_kernel, na_nl_kernel)

# ca kernel
ca_membrane_kernel, _ = DesignExponentialFilter(
    tau_ms=40, filter_length_ms=100)
ca_nl_kernel, _ = DesignExponentialFilter(tau_ms=40, filter_length_ms=200)

ca_kernel = get_linear_nonlinear_model(ca_membrane_kernel, ca_nl_kernel)

# nmda kernel
nmda_membrane_kernel, _ = DesignExponentialFilter(
    tau_ms=80, filter_length_ms=200)
nmda_nl_kernel, _ = DesignExponentialFilter(tau_ms=80, filter_length_ms=200)

nmda_kernel = get_linear_nonlinear_model(nmda_membrane_kernel, nmda_nl_kernel)

# input signal
input_epsp = delayed_pulse(delay_ms=20, peak_amplitude=10)

input_signal_pane = dbc.Card(
    dbc.CardBody(
        [
            dcc.Graph(id='input_graph'),
            dbc.FormGroup([
                dbc.Label('Peak'),
                dcc.Slider(
                    id='input-peak-slider',
                    min=1,
                    max=15,
                    step=0.1,
                    value=1,
                    updatemode='drag',
                    tooltip=dict(placement='topLeft')
                )
            ]),
        ]
    )
)
na_pane = dbc.Card(
    dbc.CardBody(
        [
            dcc.Graph(id='na_graph'),
            dbc.FormGroup([
                dbc.Label('Loc'),
                dcc.Slider(
                    id='na-loc-slider',
                    min=na_loc_min,
                    max=na_loc_max,
                    step=0.1,
                    value=na_loc_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
            dbc.FormGroup([
                dbc.Label('Sensit'),
                dcc.Slider(
                    id='na-sensit-slider',
                    min=na_sensit_min,
                    max=na_sensit_max,
                    step=sensit_step,
                    value=na_sensit_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
            dbc.FormGroup([
                dbc.Label('Gain'),
                dcc.Slider(
                    id='na-gain-slider',
                    min=na_gain_min,
                    max=na_gain_max,
                    step=0.1,
                    value=na_gain_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
        ]
    )
)
ca_pane = dbc.Card(
    dbc.CardBody(
        [
            dcc.Graph(id='ca_graph'),
            dbc.FormGroup([
                dbc.Label('Loc'),
                dcc.Slider(
                    id='ca-loc-slider',
                    min=ca_loc_min,
                    max=ca_loc_max,
                    step=0.1,
                    value=ca_loc_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
            dbc.FormGroup([
                dbc.Label('Sensit'),
                dcc.Slider(
                    id='ca-sensit-slider',
                    min=ca_sensit_min,
                    max=ca_sensit_max,
                    step=sensit_step,
                    value=ca_sensit_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
            dbc.FormGroup([
                dbc.Label('Gain'),
                dcc.Slider(
                    id='ca-gain-slider',
                    min=ca_gain_min,
                    max=ca_gain_max,
                    step=0.1,
                    value=ca_gain_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
        ]
    )
)
nmda_pane = dbc.Card(
    dbc.CardBody(
        [
            dcc.Graph(id='nmda_graph'),
            dbc.FormGroup([
                dbc.Label('Loc'),
                dcc.Slider(
                    id='nmda-loc-slider',
                    min=nmda_loc_min,
                    max=nmda_loc_max,
                    step=0.1,
                    value=nmda_loc_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
            dbc.FormGroup([
                dbc.Label('Sensit'),
                dcc.Slider(
                    id='nmda-sensit-slider',
                    min=nmda_sensit_min,
                    max=nmda_sensit_max,
                    step=sensit_step,
                    value=nmda_sensit_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
            dbc.FormGroup([
                dbc.Label('Gain'),
                dcc.Slider(
                    id='nmda-gain-slider',
                    min=nmda_gain_min,
                    max=nmda_gain_max,
                    step=0.1,
                    value=nmda_gain_min,
                    tooltip=dict(placement='topLeft')
                )
            ]),
        ]
    )
)

blocker_pane = dbc.Card(
    dbc.CardBody(
        [
            dcc.Graph(id='blocker_graph'),
        ]
    )
)

# Build App
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
app.layout = html.Div([
    html.H1("NMDA Spikes analysis"),
    html.Div(id='param-container'),
    html.Div([
        dbc.Row([
            dbc.Col([input_signal_pane], width=6), dbc.Col([na_pane], width=6)
        ]),
        dbc.Row([
            dbc.Col([ca_pane], width=6), dbc.Col([nmda_pane], width=6)
        ]),
        dbc.Row([
            dbc.Col([blocker_pane], width=6)
        ])
    ])
])

# Define callback to update graph
@app.callback(
    Output('param-container', 'children'),
    [
        Input('input-peak-slider', 'value'),
        Input('na-loc-slider', 'value'),
        Input('na-sensit-slider', 'value'),
        Input('na-gain-slider', 'value'),
        Input('ca-loc-slider', 'value'),
        Input('ca-sensit-slider', 'value'),
        Input('ca-gain-slider', 'value'),
        Input('nmda-loc-slider', 'value'),
        Input('nmda-sensit-slider', 'value'),
        Input('nmda-gain-slider', 'value'),
    ]
)
def display_hash(peak, na_loc, na_sensit, na_gain, ca_loc, ca_sensit, ca_gain, nmda_loc, nmda_sensit, nmda_gain):
    return 'peak: ' + str(peak) + ' na: ' + str((na_loc, na_sensit, na_gain)) + ' ca: ' + str((ca_loc, ca_sensit, ca_gain)) + ' ndma: ' + str((nmda_loc, nmda_sensit, nmda_gain))

  # Define callback to update graph
@app.callback(
    Output('blocker_graph', 'figure'),
    [
        Input('input-peak-slider', 'value'),
        Input('na-loc-slider', 'value'),
        Input('na-sensit-slider', 'value'),
        Input('na-gain-slider', 'value'),
        Input('ca-loc-slider', 'value'),
        Input('ca-sensit-slider', 'value'),
        Input('ca-gain-slider', 'value'),
        Input('nmda-loc-slider', 'value'),
        Input('nmda-sensit-slider', 'value'),
        Input('nmda-gain-slider', 'value'),
    ]
)
def compute_blocker_graph(peak, na_loc, na_sensit, na_gain, ca_loc, ca_sensit, ca_gain, nmda_loc, nmda_sensit, nmda_gain):
  # create input intensities to evaluate
  min_intensity = 0
  max_intensity = 15

  intensities = np.arange(min_intensity, max_intensity, 0.4)
  peaks_control = np.zeros(len(intensities))
  peaks_blocked = np.zeros(len(intensities))

  # calculate peak amplitudes for non blocked Ca and Na channels
  for e, intensity in enumerate(intensities):
    input_epsp = delayed_pulse(
        delay_ms=20, peak_amplitude=intensity).command.flatten()
    y_ca, _= ca_kernel(input_epsp, ca_loc, ca_sensit, ca_gain)
    y_na, _= na_kernel(input_epsp, na_loc, na_sensit, na_gain)
    y_nmda_sum, _ = nmda_kernel(
        y_ca+y_na, nmda_loc, nmda_sensit, nmda_gain)
    peak = np.max(y_nmda_sum)
    peaks_control[e] = peak


  # calculate peak amplitudes for blocked Ca and Na channels (set sigmoid gain to 0)
  for e, intensity in enumerate(intensities):
    input_epsp = delayed_pulse(
        delay_ms=20, peak_amplitude=intensity).command.flatten()
    y_ca, _= ca_kernel(input_epsp, ca_loc, ca_sensit, 0)
    y_na, _= na_kernel(input_epsp, na_loc, na_sensit, 0)
    y_nmda_sum, _ = nmda_kernel(
        y_ca+y_na, nmda_loc, nmda_sensit, nmda_gain)
    peak = np.max(y_nmda_sum)
    peaks_blocked[e] = peak

  # graph both traces
  non_blocked = go.Scatter(
        x=intensities,
        y=peaks_control,
        mode='lines+markers',
        name='Control'
  )

  blocked = go.Scatter(
      x=intensities,
      y=peaks_blocked,
      mode='lines+markers',
      name='TTX+cadmium'
  )

  layout = go.Layout(
        title='TTX+cadmium block vs control',
        yaxis=dict(
            title='Peak amplitude'
        ),
        xaxis=dict(
            title='Input Intensity'
        )
  )
  fig= go.Figure(data=[non_blocked, blocked], layout=layout)
  return fig

# Define callback to update graph


@ app.callback(
    Output('input_graph', 'figure'),
    [
        Input('input-peak-slider', 'value'),
    ]
)
def update_input_figure(peak):
    print('hello')
    layout = go.Layout(
        title='EPSP Input',
        yaxis=dict(
            title='V'
        ),
        xaxis=dict(
            title='t(ms)'
        )
    )

    input_epsp = delayed_pulse(
        delay_ms=20, peak_amplitude=peak).command.flatten()
    trace = go.Scatter(
        y=input_epsp,
        mode='lines',
        line=dict(
            shape='spline'
        ),
        name='Lin+NL'
    )
    fig= go.Figure(data=[trace], layout=layout)
    return fig

# Define callback to update graph


@ app.callback(
    Output('na_graph', 'figure'),
    [
        Input('input-peak-slider', 'value'),
        Input('na-loc-slider', 'value'),
        Input('na-sensit-slider', 'value'),
        Input('na-gain-slider', 'value'),
    ]
)
def update_na_figure(peak, loc, sensit, gain):
    layout = go.Layout(
        title='Na kernel',
        yaxis=dict(
            title='voltage'
        ),
        xaxis=dict(
            title='time(ms)'
        )
    )

    input_epsp = delayed_pulse(
        delay_ms=20, peak_amplitude=peak).command.flatten()
    y_sum, y_lin= na_kernel(input_epsp, loc, sensit, gain)

    trace_sum = go.Scatter(
        y=y_sum,
        mode='lines',
        line=dict(
            shape='spline'
        ),
        name='Lin+NL'
    )
    trace_lin = go.Scatter(
        y=y_lin,
        mode='lines',
        line=dict(
            shape='spline'
        ),
        name='Lin'
    )
    fig= go.Figure(data=[trace_sum, trace_lin], layout=layout)
    return fig

# Define callback to update graph
@ app.callback(
    Output('ca_graph', 'figure'),
    [
        Input('input-peak-slider', 'value'),
        Input('ca-loc-slider', 'value'),
        Input('ca-sensit-slider', 'value'),
        Input('ca-gain-slider', 'value'),
    ]
)
def update_ca_figure(peak, loc, sensit, gain):
    layout = go.Layout(
        title='Ca kernel',
        yaxis=dict(
            title='voltage'
        ),
        xaxis=dict(
            title='time(ms)'
        )
    )

    input_epsp = delayed_pulse(
        delay_ms=20, peak_amplitude=peak).command.flatten()
    y_sum, y_lin= ca_kernel(input_epsp, loc, sensit, gain)

    trace_sum = go.Scatter(
        y=y_sum,
        mode='lines',
        line=dict(
            shape='spline'
        ),
        name='Lin+NL'
    )
    trace_lin = go.Scatter(
        y=y_lin,
        mode='lines',
        line=dict(
            shape='spline'
        ),
        name='Lin'
    )
    fig= go.Figure(data=[trace_sum, trace_lin], layout=layout)
    return fig

# Define callback to update graph


@ app.callback(
    Output('nmda_graph', 'figure'),
    [
        Input('input-peak-slider', 'value'),
        Input('na-loc-slider', 'value'),
        Input('na-sensit-slider', 'value'),
        Input('na-gain-slider', 'value'),
        Input('ca-loc-slider', 'value'),
        Input('ca-sensit-slider', 'value'),
        Input('ca-gain-slider', 'value'),
        Input('nmda-loc-slider', 'value'),
        Input('nmda-sensit-slider', 'value'),
        Input('nmda-gain-slider', 'value'),
    ]
)
def update_nmda_figure(peak, na_loc, na_sensit, na_gain, ca_loc, ca_sensit, ca_gain, nmda_loc, nmda_sensit, nmda_gain):
    layout = go.Layout(
        title='NMDA kernel',
        yaxis=dict(
            title='voltage'
        ),
        xaxis=dict(
            title='time(ms)'
        )
    )

    input_epsp = delayed_pulse(
        delay_ms=20, peak_amplitude=peak).command.flatten()
    y_ca, _= ca_kernel(input_epsp, ca_loc, ca_sensit, ca_gain)
    y_na, _= na_kernel(input_epsp, na_loc, na_sensit, na_gain)
    y_nmda_sum, y_nmda_lin = nmda_kernel(
        y_ca+y_na, nmda_loc, nmda_sensit, nmda_gain)

    trace_sum = go.Scatter(
        y=y_nmda_sum,
        mode='lines',
        line=dict(
            shape='spline'
        ),
        name='Lin+NL'
    )
    trace_lin = go.Scatter(
        y=y_nmda_lin,
        mode='lines',
        line=dict(
            shape='spline'
        ),
        name='Lin'
    )
    fig= go.Figure(data=[trace_sum, trace_lin], layout=layout)
    return fig


if __name__ == '__main__':
    # Run app and display result inline in the notebook
    app.run_server(use_reloader=True)
