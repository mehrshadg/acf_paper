

clear, 


for i=1:100
close all


% White noise
fs=1;
sig_len = 60 * 10;
signal1_1min=randn(fs*sig_len,1);
t=linspace(0,sig_len,fs*sig_len);
[PSD,f]=pwelch(signal1_1min,hamming(100),[],[],fs);

figure
h41=subplot(3,4,1);
plot(t,signal1_1min)
xlim([0 sig_len])
ylim([-5 5])
set(gca,'fontsize',14)
ylabel('Amplitude')
xlabel('Time (seconds)')
box off
h41.LineWidth=2;

h45=subplot(3,4,5);
plot(f,log10(PSD),'linewidth',2)
xlim([0 50])
ylim([-4 0])
set(gca,'fontsize',14)
ylabel('log(PSD)')
xlabel('Frequency (Hz)')
box off
h45.LineWidth=2;

h49=subplot(3,4,9);
ACW_white(i)=ACW_estimation_figure(signal1_1min',fs,0,50,50)
h49.LineWidth=2;






% Pink noise
fs=500;
signal2_1min=pinknoise(fs*60,1);
t=linspace(0,60,fs*60);
[PSD,f]=pwelch(signal2_1min,hamming(100),[],[],fs);

h42=subplot(3,4,2);
plot(t,signal2_1min)
xlim([0 60])
ylim([-5 5])
set(gca,'fontsize',14)
ylabel('Amplitude')
xlabel('Time (seconds)')
box off
h42.LineWidth=2;

h46=subplot(3,4,6);
plot(f,log10(PSD),'linewidth',2)
xlim([0 50])
ylim([-4 0])
set(gca,'fontsize',14)
ylabel('log(PSD)')
xlabel('Frequency (Hz)')
box off
h46.LineWidth=2;

h410=subplot(3,4,10);
ACW_pink(i)=ACW_estimation_figure(signal2_1min',fs,0,50,50)
h410.LineWidth=2;




% Chirp
fs=500;
t=linspace(0,60,fs*60);
signal3_1min=chirp(t,0,fs*60,1000);
[PSD,f]=pwelch(signal3_1min,hamming(100),[],[],fs);

h43=subplot(3,4,3);
plot(t,signal3_1min)
xlim([0 60])
ylim([-1.1 1.1])
set(gca,'fontsize',14)
ylabel('Amplitude')
xlabel('Time (seconds)')
box off
h43.LineWidth=2;

h47=subplot(3,4,7);
plot(f,log10(PSD),'linewidth',2)
xlim([0 50])
ylim([-7 0])
set(gca,'fontsize',14)
ylabel('log(PSD)')
xlabel('Frequency (Hz)')
box off
h47.LineWidth=2;

h411=subplot(3,4,11);
ACW_chirp(i)=ACW_estimation_figure(signal3_1min',fs,0,50,50)
h411.LineWidth=2;



% Sen
fs=500;
t=linspace(0,60,fs*60);
signal4_1min=sin(2*pi*t*1)+1;
[PSD,f]=pwelch(signal4_1min,hamming(100),[],[],fs);

h44=subplot(3,4,4);
plot(t,signal4_1min)
xlim([0 60])
ylim([-.1 2.1])
set(gca,'fontsize',14)
ylabel('Amplitude')
xlabel('Time (seconds)')
box off
h44.LineWidth=2;

h48=subplot(3,4,8);
plot(f,log10(PSD),'linewidth',2)
xlim([0 50])
ylim([-7 0])
set(gca,'fontsize',14)
ylabel('log(PSD)')
xlabel('Frequency (Hz)')
box off
h48.LineWidth=2;

h412=subplot(3,4,12);
ACW_sen(i)=ACW_estimation_figure(signal4_1min',fs,0,50,50)
h412.LineWidth=2;

end

% parseval








