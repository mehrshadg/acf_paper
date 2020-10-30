clc, clear;

fs = 508;
seconds = 60 * 5;
iniHz = 1.3;
endHz = 45;
nfft = 2 ^ 15;
n_signal = 5000; %360 * 4 * 10;

t=linspace(0,seconds,seconds*fs);  % Time vector

types = [
  1,0,0;
  0,1,0;
  0,0,1;
  1,1,0;
  1,0,1;
  0,1,1;
  1,1,1;
];

n_type = size(types, 1);
n_points = length(t);

signals = zeros(n_type, n_signal, n_points);
for type = 1 : size(types, 1)
    disp(type);
    a = types(type, 1);
    b = types(type, 2);
    c = types(type, 3);
    
    parfor iter = 1 : n_signal
        my_freq = (endHz - iniHz) .* rand + iniHz;

        % Patient prestim parameters
        A=a*2.5+0.1*randn;    % pink noise amplitude
        B=b*2.5+0.1*randn;    % white noise amplitude
        C=c*2+0.1*randn;      % sinusoidal wave amplitude,,,,,,,,,,,,,
        myrand=1;%+0.1*randn;
        Microvolts=10^-6;

        signal=Microvolts*(A*pinknoise(seconds*fs,1)'+(randn(1,length(t))*B)+C*sin(2*pi*my_freq*myrand*t));
        signals(type, iter, :) = signal;
    end
end

save('signals.mat', 'signals', 't', 'types', '-v7.3');
