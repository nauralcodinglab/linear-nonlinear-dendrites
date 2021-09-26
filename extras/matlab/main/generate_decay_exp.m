function [t, y] = generate_decay_exp(tau_ms, t_step)
	t = 0:t_step:200;
    y = exp(-t/tau_ms);
end