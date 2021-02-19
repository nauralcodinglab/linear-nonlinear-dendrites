close all
clear

dt_ms = 0.01; % step size for alpha signal

% create input alpha signal
tau_ms = 2;
[t_alpha, y_alpha] = generate_alpha(tau_ms, dt_ms);
alpha_signal = transpose([t_alpha; y_alpha]);

% create decay exponential just in case
[t_exp, y_exp] = generate_decay_exp(tau_ms, dt_ms);
exp_signal = transpose([t_exp; y_exp]);

% static parameters
pulse_width_ms = 1;
t_end_ms = 30; % length of simulink simulatio in ms
t_step = 0.1; % max step size of solver

% multi simulation parameters
peak = 1.95; % peak amplitude of alpha signal
loc = 2; % loc of sigmoid
gain = 3; % gain of sigmoid
sensitivity = 0.35; % sensit of sigmoid
pulse_height = 1.5; % pulse height
soma_threshold = 0.23; % soma threshold
input_signal = alpha_signal;

model = 'na_dendrite_soma';
load_system(model);
sim(model);




