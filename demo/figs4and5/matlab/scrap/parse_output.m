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