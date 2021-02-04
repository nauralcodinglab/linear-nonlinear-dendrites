dt_ms = 0.1;
t = 0:dt_ms:200;

tau_ms = 2;

C = exp(1) / tau_ms;

output = C * t .* exp(-t/tau_ms);

input_signal = transpose([t; output]);
