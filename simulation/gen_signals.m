
clear; clc;

length = 300;
fs = 1;

signals = zeros(700, length);
for i = 1 : size(signals, 1) 
    disp(i);
    t = linspace(0,length,fs*length);
    signal1 = randn(fs * length,1)';   
    % Pink noise
    signal2 = pinknoise(fs*length,1)';    
    % Chirp
    signal3 = chirp(t,0,fs*length,1000);
    % Sen
    signal4 = sin(2*pi*t*10);
    % Mixture
    signals(i, :) = 0.5 * randn * signal1 + 0.5 * randn * signal2  + 0.5 * randn * signal3;
end

save('signals.mat', 'signals');