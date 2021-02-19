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
t_end_ms = 50; % length of simulink simulatio in ms
t_step = 0.01; % max step size of solver

% multi simulation parameters
peak_arr = 1:0.05:2; % peak amplitude of alpha signal
loc_arr = 1.9:0.1:2.1; % loc of sigmoid
gain_arr = 3:1:3; % gain of sigmoid
sensit_arr = 0.3:0.01:0.35; % sensit of sigmoid
pulse_height_arr = 1.5:0.5:1.5; % pulse height
soma_thresh_arr = 0.23:0.01:0.25; % soma threshold


idx = 1;
for i = 1:length(peak_arr)
    peak = peak_arr(i);
    for j = 1:length(loc_arr)
        loc = loc_arr(j);
        for k = 1:length(gain_arr)
            gain = gain_arr(k);
            for l = 1:length(sensit_arr)
                sensit = sensit_arr(l);     
                for m = 1:length(pulse_height_arr)
                    pulse_height = pulse_height_arr(m);
                    for n = 1:length(soma_thresh_arr)
                        soma_thresh = soma_thresh_arr(n);
                        params(idx).peak = peak;
                        params(idx).loc = loc;
                        params(idx).gain = gain;
                        params(idx).sensit = sensit;
                        params(idx).pulse_height = pulse_height;
                        params(idx).soma_thresh = soma_thresh;
                        idx = idx + 1;
                    end
                end
            end
        end
    end
end

model = 'na_dendrite_soma';
load_system(model)

for i = 1:length(params)
    in(i) = Simulink.SimulationInput(model);
    in(i) = in(i).setVariable('peak', params(i).peak);
    in(i) = in(i).setVariable('loc', params(i).loc);
    in(i) = in(i).setVariable('sensitivity', params(i).sensit);
    in(i) = in(i).setVariable('gain', params(i).gain);
    in(i) = in(i).setVariable('pulse_height', params(i).pulse_height);
    in(i) = in(i).setVariable('soma_threshold', params(i).soma_thresh);
end
in = in.setVariable('pulse_width_ms', pulse_width_ms);
in = in.setVariable('t_step', t_step);
in = in.setVariable('t_end_ms', t_end_ms);
in = in.setVariable('input_signal', alpha_signal);
out = parsim(in, 'UseFastRestart', 'on');

for i = 1:length(params)
    sim_results = out(i);
    signal_logs = sim_results.logsout;

    Vd = signal_logs.getElement('Vd');
    input = signal_logs.getElement('input');
    Vs = signal_logs.getElement('Vs');
    
    Vd_lin = signal_logs.getElement('Vd_lin');
    Vd_lin_vals = Vd_lin.Values.Data(:);

    Vd_time = Vd.Values.Time(:);
    Vd_vals = Vd.Values.Data(:);

    Vs_time = Vs.Values.Time(:);
    Vs_vals = Vs.Values.Data(:);
    
    out_mat(i).time = Vd_time;
    out_mat(i).Vd = Vd_vals;
    out_mat(i).Vs = Vs_vals;
    out_mat(i).input = input.Values.Data(:);
    out_mat(i).Vd_lin = Vd_lin_vals;
    
    par = params(i);
    out_mat(i).peak = par.peak;
    out_mat(i).loc = par.loc;
    out_mat(i).gain = par.gain;
    out_mat(i).sensit = par.sensit;
    out_mat(i).pulse_height = par.pulse_height;
    out_mat(i).soma_thresh = par.soma_thresh;
end

save('./output/output.mat', 'out_mat');



