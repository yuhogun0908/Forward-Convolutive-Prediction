clc; clear; close all;
[mix, fs] = audioread('mix.wav');
figure(1)
spectrogram(mix(:,1), 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('Input');
[clean_s0, ~] = audioread('ref_s0.wav');
[clean_s1, ~] = audioread('ref_s1.wav');
figure(2)
subplot(2,1,1);
spectrogram(clean_s0, 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('Clean Source1')
subplot(2,1,2);
spectrogram(clean_s1, 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('Clean Source2')


[MISO1_s0,~] = audioread('MISO1_s0.wav');
[MISO1_s1,~] = audioread('MISO1_s1.wav');
figure(3);
subplot(2,1,1);
spectrogram(MISO1_s0(:,1), 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('MISO1 Separate Model Output Source1');
subplot(2,1,2);
spectrogram(MISO1_s1(:,2), 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('MISO1 Separate Model Output Source2');



[Beamform_s0, ~] = audioread('Origin_Beamform_s0.wav');
[Beamform_s1, ~] = audioread('Origin_Beamform_s1.wav');

figure(4); 
subplot(2,1,1);
spectrogram(Beamform_s0, 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('Beamforming Output Source1');
subplot(2,1,2);
spectrogram(Beamform_s1, 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('Beamforming Output Source2');



[DNN_WPE_s0, ~] = audioread('DNN_WPE_s0.wav');
[DNN_WPE_s1, ~] = audioread('DNN_WPE_s1.wav');
figure(5);
subplot(2,1,1);
spectrogram(DNN_WPE_s0(:,1), 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('DNN WPE Dereverberation Output Source1');
subplot(2,1,2);
spectrogram(DNN_WPE_s1(:,2), 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('DNN WPE Dereverberation Output Source2');


[DNN_WPE_Beamform_s0, ~] = audioread('DNN_WPE_Beamform_s0.wav');
[DNN_WPE_Beamform_s1, ~] = audioread('DNN_WPE_Beamform_s1.wav');
figure(6);
subplot(2,1,1);
spectrogram(DNN_WPE_Beamform_s0, 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('DNN WPE & Beamforming Output Source1');
subplot(2,1,2);
spectrogram(DNN_WPE_Beamform_s1, 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('DNN WPE & Beamforming Output Source2');


[FCP_s0, ~] = audioread('FCP_s0.wav');
[FCP_s1, ~] = audioread('FCP_s1.wav');
figure(7);
subplot(2,1,1);
spectrogram(FCP_s0(:,1), 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('FCP Dereverberation Output Source1');
subplot(2,1,2);
spectrogram(FCP_s1(:,2), 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('FCP Dereverberation Output Source2');


[FCP_Beamform_s0, ~] = audioread('FCP_Beamform_s0.wav');
[FCP_Beamform_s1, ~] = audioread('FCP_Beamform_s1.wav');
figure(8);
subplot(2,1,1);
spectrogram(FCP_Beamform_s0, 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('FCP & Beamforming Output Source1');
subplot(2,1,2);
spectrogram(FCP_Beamform_s1, 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('FCP & Beamforming Output Source2');


[cFCP_s0, ~] = audioread('cFCP_s0.wav');
[cFCP_s1, ~] = audioread('cFCP_s1.wav');
figure(9);
subplot(2,1,1);
spectrogram(cFCP_s0(:,1), 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('combine FCP Dereverberation Output Source1');
subplot(2,1,2);
spectrogram(cFCP_s1(:,2), 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('combine FCP Dereverberation Output Source2');
5

[cFCP_Beamform_s0, ~] = audioread('cFCP_Beamform_s0.wav');
[cFCP_Beamform_s1, ~] = audioread('cFCP_Beamform_s1.wav');
figure(10); 
subplot(2,1,1);
spectrogram(cFCP_Beamform_s0, 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('combine FCP & Beamforming Output Source1');
subplot(2,1,2);
spectrogram(cFCP_Beamform_s1, 512, 384, 512, fs, 'yaxis'); caxis([-130 -50]); colormap jet;
title('combine FCP & Beamforming Output Source2');







