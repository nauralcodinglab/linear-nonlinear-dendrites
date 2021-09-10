close all
clear

dt_ms = 0.01;
t = 0:dt_ms:200;

t_end_ms = 30; % length of simulink simulatio in ms
t_step = 0.1; % max step size of solver

% create input alpha signal
tau_ms = 2;
C = exp(1) / tau_ms;
output = C * t .* exp(-t/tau_ms);
input_signal = transpose([t; output]);

% model parameters
soma_threshold = 0.27; % threshold of soma before it spikes

% soma nonlinear parameters
pulse_width_ms = 1;
pulse_height = 4;

% dendrite sigmoid parameters
loc = 1.7;
gain = 5;
sensitivity = 0.3868;

peak = 2.0;

sim('lnl_working4.slx');

% load_system(model)
% Simulink.BlockDiagram.buildAcceleratorTarget(model);
% for i = 1:length(peak_arr)
%     peak = peak_arr(i);
%     for j = 1:length(loc_arr)
%         loc = loc_arr(j);
%         
%          % run simulink simulation
% %         in = Simulink.SimulationInput(model);
% %         in = in.setModelParameter('SimulationMode', 'rapid-accelerator');
% %         in = in.setModelParameter('RapidAcceleratorUpToDateCheck', 'off');
% %         sim_results = parsim(in);
% %         paramNameValStruct.SimulationMode = 'rapid-accelerator';
% %         paramNameValStruct.RapidAcceleratorUpToDateCheck = 'off';
% %         sim_results = sim(model);
% 
%         % out results
% %         signal_logs = sim_results.logsout;
% % 
% %         Vd = signal_logs.getElement('Vd');
% %         Vd_lin = signal_logs.getElement('Vd_lin');
% %         Vd_nl = signal_logs.getElement('Vd_nl');
% %         input = signal_logs.getElement('input');
% %         Vs = signal_logs.getElement('Vs');
% % 
% %         Vd_time = Vd.Values.Time(:);
% %         Vd_vals = Vd.Values.Data(:);
% % 
% %         Vs_time = Vs.Values.Time(:);
% %         Vs_vals = Vs.Values.Data(:);
% % 
% %         Vd_lin_time = Vd_lin.Values.Time(:);
% %         Vd_lin_vals = Vd_lin.Values.Data(:);
% % 
% %         Vd_nl_time = Vd_nl.Values.Time(:);
% %         Vd_nl_vals = Vd_nl.Values.Data(:);
% %         
% %         loc_output(j).time = Vd_time;
% %         loc_output(j).Vd = Vd_vals;
% %         loc_output(j).Vs = Vs_vals;
%     end
% %     input_vals = input.Values.Data(:);
% %     peak_output(i).input = input_vals;
% %     peak_output(i).peak = peak;
% %     peak_output(i).trials = loc_output;
% end

% multi simulation parameters
peak_arr = 1:0.5:2.5; % peak amplitude of alpha signal
loc_arr = 1.5:0.5:2.5; % loc of sigmoid
gain_arr = 2:1:4; % gain of sigmoid
sensit_arr = 0.3:0.1:0.6; % sensit of sigmoid
pulse_height_arr = 1:0.5:2; % pulse height
soma_thresh_arr = 0.24:0.01:0.29; % soma threshold


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

model = 'lnl_working4';
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
in = in.setVariable('pulse_width_ms', 1);
in = in.setVariable('t_step', 0.01);
in = in.setVariable('t_end_ms', 50);
in = in.setVariable('input_signal', input_signal);
out = parsim(in, 'UseFastRestart', 'on');

for i = 1:length(params)
    sim_results = out(i);
    signal_logs = sim_results.logsout;

    Vd = signal_logs.getElement('Vd');
    input = signal_logs.getElement('input');
    Vs = signal_logs.getElement('Vs');

    Vd_time = Vd.Values.Time(:);
    Vd_vals = Vd.Values.Data(:);

    Vs_time = Vs.Values.Time(:);
    Vs_vals = Vs.Values.Data(:);
    
    out_mat(i).time = Vd_time;
    out_mat(i).Vd = Vd_vals;
    out_mat(i).Vs = Vs_vals;
    out_mat(i).input = input.Values.Data(:);
    
    par = params(i);
    out_mat(i).peak = par.peak;
    out_mat(i).loc = par.loc;
    out_mat(i).gain = par.gain;
    out_mat(i).sensit = par.sensit;
    out_mat(i).pulse_height = par.pulse_height;
    out_mat(i).soma_thresh = par.soma_thresh;
end

save('output.mat', 'out_mat');



