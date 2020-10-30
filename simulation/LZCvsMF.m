

clear,


for i=1:100 % For tests, keep 100 repetitions. For tests, you can reduce it
    i
    % White noise
    fs=1;
    signal1_1min=randn(fs*60,1);
    t=linspace(0,60,fs*60);
    [PSD1,f]=pwelch(signal1_1min,hamming(100),[],[],fs);
    MF_WhiteNoise(i)=CalculoMF(PSD1, f, [1 30]);
    LZC_WhiteNoise(i)=lzcomplexity_tramas(signal1_1min,'mediana',2,0);
    
    
    % Pink noise
    fs=500;
    signal2_1min=pinknoise(fs*60,1);
    t=linspace(0,60,fs*60);
    [PSD2,f]=pwelch(signal2_1min,hamming(100),[],[],fs);
    MF_PinkNoise(i)=CalculoMF(PSD2, f, [1 30]);
    LZC_PinkNoise(i)=lzcomplexity_tramas(signal2_1min,'mediana',2,0);
    
    
    % Chirp
    fs=500;
    t=linspace(0,60,fs*60);
    signal3_1min=chirp(t,0,fs*60,1000);
    signal3_1min=signal3_1min+0.01*signal1_1min'; % Adding low level of white
                                                  % noise to obtain different
                                                  %(but similar) values of MF and LZC
                                                  % This level of noise
                                                  % must be very low or
                                                  % zero to avoid influence
                                                  % in the LZC
                                                  
    [PSD3,f]=pwelch(signal3_1min,hamming(100),[],[],fs);
    MF_Chirp(i)=CalculoMF(PSD3, f, [1 30]);
    LZC_Chirp(i)=lzcomplexity_tramas(signal3_1min,'mediana',2,0);
    
    
    % Sen
    fs=500;
    t=linspace(0,60,fs*60);
    signal4_1min=sin(2*pi*t*10);
    signal4_1min=signal4_1min+0.01*signal1_1min';   % Adding low level of white
                                                    % noise to obtain different
                                                    %(but similar) values of MF and LZC
                                                    % This level of noise
                                                    % must be very low or
                                                    % zero to avoid influence
                                                    % in the LZC
    [PSD4,f]=pwelch(signal4_1min,hamming(100),[],[],fs);
    MF_Sen(i)=CalculoMF(PSD4, f, [1 30]);
    LZC_Sen(i)=lzcomplexity_tramas(signal4_1min,'mediana',2,0);
end
