%    clc;close all; clear all;
    name = 'NewWave_1';
    load(name);
    [XI,YI] = meshgrid(0:1:100);
    figure(18); mesh(XI,YI,Data); hold on;
    Vec = 35:5:70;
    figure(18); contour3(XI,YI,Data,Vec);
