function [t, y] = generate_alpha(tau_ms, t_step)
    t = 0:t_step:200;
    C = exp(1) / tau_ms;
    output = C * t .* exp(-t/tau_ms);
    y = output;
end