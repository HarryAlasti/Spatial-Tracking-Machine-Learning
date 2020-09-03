% Analysis of spanning the signal strength range is included.
% Lloyd-Max approach for progressive learning is removed.
% Signal tracking for equally-spaced contour levels included.
% reinforcement learning included for equally spaced contour levels.
% initial delta adaptation is applied.
% compensating the initial Delta by considering the signal range span

clc; close all; pause(1); clear; 
L0 = 35;               
Lend = 45;             
L = 10000;             
Nlevels0 = 10;          
n_std = 0.3;    
Navg = 1;             % maximum 50
jmp = 3;
jmp0 = jmp;
%Npnt = fix(35/jmp); 
Npnt = 25;
rw = 0;                   % rw = 0 for no reward. rw= 1 for full reward.

self = 1;                % 1 for using previous step as reference in calculation of error
nFlag = 1;             % 0 for norm-2 and 1 for norm-1
LFlag = 1;             % 0 for just the immediate level-based construction & 1 for all so-far selected marginal 

MA =10;               % moving average window size
perc = 5/100;
step = 0.01;
Thr = 0.2;
THR = 5;
mu = 1;
dyn_end = 300;

Error_opt = zeros(1, Npnt);
Error_unif = zeros(1, Npnt);
Error_learn_unif = zeros(1, Npnt);
Delta_unif_avg = zeros(1, Npnt);
Slevel = zeros(1, Npnt);
Sensors = zeros(L,7+MA+2);

Cost_opt = zeros(size(Slevel));
Cost_unif = zeros(size(Slevel));
Span_u = zeros(size(Slevel));
[XI,YI] = meshgrid(0:1:100);
Slevel = zeros(2 , Npnt);
Peak = 100;

Sensors(:,1:2) = Peak * rand(L,2);

% ================================================================= Uniform Adapted
for rept = 1 : Navg
    error_1 = 0;
    error_2 = 0;
    span_factor = 1;
    LU_bank = zeros(1, Npnt);    
    Levels_u = zeros(1, Npnt);  
    Nlevels = Nlevels0;
    Slevel(1,1) = Nlevels0;
    
    % Generation of the large scale Gaussians
    name = sprintf('NewWave_%d',1);
    load(name);
    X = Sensors(:,1);
    Y = Sensors(:,2);
    Z = zeros(L,1);
    for k = 1: N
        Z = Z + 2*Peak * (P(k,3) - 0.5) / sqrt(2*pi) / Sigma * exp(-((X-P(k,1)).^2 + (Y-P(k,2)).^2)/2/Sigma^2);
    end

    % Genmeration of small scale Gaussians
    for k = 1 : Ns
        Z = Z + 5 * (Ps(k,3) - 0.5) / sqrt(2*pi) / Ss * exp(-((X-Ps(k,1)).^2 + (Y-Ps(k,2)).^2)/2/Ss^2);
    end
   
    % Snapshot normalization
    Z = Peak / 2 * Z / (max(max(Z)) - min(min(Z)))+50;
    Sensors(:,3) = Z;        
    Sensors(:,4) = Z + n_std * randn(size(Z));   
%{    
    Nx = 2;
    m1 = fix(L * rand(Nx , 1));
    L1 = mean(Sensors(m1,4));
    p = 0;
    while (p == 0)
        m2 = fix(L * rand(Nx , 1));
        L2 = mean(Sensors(m2,4));
        if (abs(L1 - L2) > THR)
            p = 1;
            L0 = min(L1 , L2);
            Lend = max(L1 , L2);
        end
    end
  %}
    
    [p,xx] = hist(Data(:),500);
    Full_span = max(xx) - min(xx);
    Span_u(1) = (Lend - L0) / Full_span;
    
    Levels = linspace(L0,Lend,Nlevels+2);
    Levels = Levels(2:end-1);
    Delta = 0.5 * (Levels(2) - Levels(1));     % Delta adaptation is applied here.
    
    Sensors(:,5:7) = 0; 
    for pnt = 1 : Npnt
        if (pnt < 3)
            if (rw == 0)
                jmp = jmp0;
                jmp_old = jmp0;
            else    
                jmp = 1;
                jmp_old = 1;
            end
        end        
        
        Sensors(:,6:7) = 0;         
        for clevel = 1 : length(Levels)
            index = find(abs(Sensors(:,4) - Levels(clevel)) <= Delta);        
            Sensors(index, 6) = 1;
            Sensors(index, 7) = Levels(clevel);            
        end
        index = find(Sensors(:,6) == 1);        
        
        Filtered = sum(abs(Sensors(index, 6) - Sensors(index, 5)));
        Sensors(index, 5) = 1;
        
        Cost_unif(pnt) = Cost_unif(pnt) + Filtered;

        if (LFlag == 1)        
            Index_select = find(Sensors(:,5) == 1);
        elseif (LFlag == 0)
            Index_select = find(Sensors(:,6) == 1);
        end
        X0 = Sensors(Index_select, 1);
        Y0 = Sensors(Index_select, 2);
        Z0 = Sensors(Index_select, 4);

        ZI = griddata2(X0,Y0,Z0,XI,YI,'v4');          % linear   nearest      cubic     v4(biharmonic interpolation)
        if (nFlag == 0) 
            error_1 = sqrt(sum(sum((ZI - Data).^2)/(101*101)));
        elseif (nFlag == 1)
            error_1 = sum(sum(abs(ZI - Data))/(101*101));
        end
        Error_unif(pnt) = Error_unif(pnt) + error_1;
        
        if (pnt > 1)
           if (nFlag == 0)
                error_2 = sqrt(sum(sum((ZI - Data_old).^2)/(101*101)));
           elseif (nFlag == 1)
                error_2 = sum(sum(abs(ZI - Data_old))/(101*101));
           end
           Error_learn_unif(pnt) = Error_learn_unif(pnt) + error_2;       
        end
        Data_old = ZI; 
        
        Levels_u(1,pnt) = pnt+Nlevels0-1;
        LU_bank(1,pnt) = Delta;

       [p,xx] = hist(ZI(:),500);
        if (pnt < Npnt)
            Span_u(pnt+1) = Span_u(pnt+1) + (max(xx) - min(xx)) / Full_span;
        end  
        
        Nlevels = Nlevels + jmp;
        if (pnt < Npnt)
            Slevel(1 , pnt+1) = Nlevels;
        end
        
        L_0 = min(min(ZI));
        L_end = max(max(ZI));
        Levels = linspace(L_0,L_end,Nlevels+2);
        Levels = Levels(2:end-1);
        
        if (self == 0)
            error = error_1;
            if (pnt ==1)
                error_old = error;
            end
        end        
        if (self == 1)
            error = error_2;
            if (pnt <= 2)
                error_old = error;
            end
        end

        if (pnt > 1)
            Delta = Delta * span_factor * (1 + mu*(error - error_old)/(error + error_old + 0.01));        
        end
                
        if (pnt < Npnt)
            span_factor = Span_u(pnt+1) / Span_u(pnt);
        end        
        
        if (pnt >= 3)            
            jmp = jmp_old * (1 + 0.5 * rw * abs(log10(error_old) - log10(error))/abs(log10(error) + log10(error_old)));
            jmp = round(jmp);
            jmp_old = jmp;
        end              
        
        error_old = error;
    end
    Delta_unif_avg = Delta_unif_avg + LU_bank(1,:);
end

Span_u = Span_u / Navg;
Cost_unif = Cost_unif / Navg;
Error_unif = Error_unif / Navg;
Delta_unif_avg = Delta_unif_avg / Navg;
Error_learn_unif = Error_learn_unif / Navg;

Delta_done = Delta;

figure(20); mesh(XI,YI,ZI); grid on;
figure(19); mesh(XI,YI,Data); grid on;

figure(25); plot(Slevel(1,:), Span_u, 'g-s'); ylabel('Signal range span (%)'); xlabel('# of levels');grid on;

figure(21); plot(Slevel(1,:), LU_bank(1,:),'g');grid;ylabel('\Delta');
figure(22); plot(Slevel(1,:), Delta_unif_avg,'g');grid; ylabel('Average \Delta');
Error_learn_unif(1) = Error_learn_unif(2);
figure(23); plot(Slevel(1,:), 20*log10(Error_learn_unif),'-sg');grid on; ylabel('Learning Error (dB)'); xlabel('# of Levels');
pause(1);

% ===========================================
% Tracking
% ===========================================
x_s = step;     % x direction shift increment
y_s = step;     % y direction shift increment
reps = 0;
Sensors(:,end) = Sensors(:,4);
Cost = zeros(ceil(dyn_end/MA),1);
Error = zeros(ceil(dyn_end/MA),1);
m = 1;
Error_0 = Error_unif(end);
xx0 = 0;
yy0 = 0;
Delta = Delta_done;
for dyn = 1:dyn_end
% Generation of the large scale Gaussians
    X = Sensors(:,1);
    Y = Sensors(:,2);
    Z = zeros(L,1);
    Data_new = zeros(size(XI));
    
    if (abs(cos(2*pi*dyn*step)) > Thr)
        xx0 = xx0 + x_s;
        yy0 = yy0 + y_s;
        reps = reps + 1;
    end
    
    for k = 1: N
        Z = Z + 2*Peak * (P(k,3) - 0.5) / sqrt(2*pi) / Sigma * exp(-((X-P(k,1)-xx0).^2 + (Y-P(k,2)-yy0).^2)/2/Sigma^2);
        Data_new = Data_new + 2*Peak * (P(k,3) - 0.5) / sqrt(2*pi) / Sigma * exp(-((XI-P(k,1)-xx0).^2 + (YI-P(k,2)-yy0).^2)/2/Sigma^2);
    end
    
% Generation of small scale Gaussians
    for k = 1 : Ns
        Z = Z + 5 * (Ps(k,3) - 0.5) / sqrt(2*pi) / Ss * exp(-((X-Ps(k,1)).^2 + (Y-Ps(k,2)).^2)/2/Ss^2);
        Data_new = Data_new + 5 * (Ps(k,3) - 0.5) / sqrt(2*pi) / Ss * exp(-((XI-Ps(k,1)).^2 + (YI-Ps(k,2)).^2)/2/Ss^2);
    end
    Z = Peak / 2 * Z / (max(Z) - min(Z))+50;
    Data_new = Peak / 2 * Data_new / (max(max(Data_new)) - min(min(Data_new)))+50;
    
    Sensors(:,4) = Z + n_std * randn(size(Z));     
    Sensors(:,5:7) = 0;
    
    if (dyn > MA)
        if (rem(dyn,MA) == 0)
            for clevel = 1 : length(Levels)
                index = find(abs(Sensors(:,4) - Levels(clevel)) <= Delta);  
                Sensors(index, 6) = 1;
            end
            Index = find(Sensors(:, 6) == 1);
            Cost(m) = length(Index); %+ length(diss);

            X1 = Sensors(Index, 1);
            Y1 = Sensors(Index, 2);
            Z1 = Sensors(Index, 4);
            Zx = griddata2(X1,Y1,Z1,XI,YI,'v4');          % linear   nearest    cubic     v4(biharmonic interpolation)
            if (nFlag == 0)
                Error(m) = sqrt(sum(sum((Zx - Data_new).^2)/(101*101)));  
            elseif (nFlag == 1)
                Error(m) = sum(sum(abs(Zx - Data_new))/(101*101));
            end
            
            L_0 = min(min(ZI));
            L_end = max(max(ZI));
            Levels = linspace(L_0,L_end,Nlevels+2);
            Levels = Levels(2:end-1);
        end
        
        m = m+1;                
    end    
end

figure(101); plot(20*log10(Error),'r*'); grid;title('performance'); ylabel('MSE in dB'); xlabel('update instant');
figure(100); plot(Cost/L*100,'bs'); grid;title('cost'); ylabel('Percentage of sensors'); xlabel('update instant');
pause(1);

% ======================================================================== Optimal

jmp0 = jmp;
for rept = 1 : Navg
    [p,xx] = hist(Data(:),500);
    p = p / sum(p);
    Nlevels = Nlevels0;
    Slevel(2 , 1) = Nlevels0;
    
    % Generation of the large scale Gaussians
    name = sprintf('NewWave_%d',1);
    load(name);
    X = Sensors(:,1);
    Y = Sensors(:,2);
    Z = zeros(L,1);
    for k = 1: N
        Z = Z + 2*Peak * (P(k,3) - 0.5) / sqrt(2*pi) / Sigma * exp(-((X-P(k,1)).^2 + (Y-P(k,2)).^2)/2/Sigma^2);
    end

    % Genmeration of small scale Gaussians
    for k = 1 : Ns
        Z = Z + 5 * (Ps(k,3) - 0.5) / sqrt(2*pi) / Ss * exp(-((X-Ps(k,1)).^2 + (Y-Ps(k,2)).^2)/2/Ss^2);
    end

    % Snapshot normalization
    Z = Peak / 2 * Z / (max(max(Z)) - min(min(Z)))+50;
    Sensors(:,3) = Z;        
    Sensors(:,4) = Z + n_std * randn(size(Z));   
    Sensors(:,5:7) = 0;
    for pnt = 1 : Npnt
        
        Levels = Lloyd_Max_2(p, xx, Nlevels);
        Delta = 0.5*min(abs(Levels(1:end-1) - Levels(2:end)));     % added on 06/03/2020 for delta adaptation
        
        Sensors(:,6:7) = 0;         
        for clevel = 1 : length(Levels)
            index = find(abs(Sensors(:,4) - Levels(clevel)) <= Delta);        
            Sensors(index, 6) = 1;
            Sensors(index, 7) = Levels(clevel);            
        end
        index = find(Sensors(:,6) == 1);        
     
        Filtered = sum(abs(Sensors(index, 6) - Sensors(index, 5)));
        Sensors(index, 5) = 1;
        
        Cost_opt(pnt) = Cost_opt(pnt) + Filtered;

        if (LFlag == 1)        
            Index_select = find(Sensors(:,5) == 1);
        elseif (LFlag == 0)
            Index_select = find(Sensors(:,6) == 1);
        end
        X0 = Sensors(Index_select, 1);
        Y0 = Sensors(Index_select, 2);
        Z0 = Sensors(Index_select, 4);

        ZI = griddata2(X0,Y0,Z0,XI,YI,'v4');          % linear   nearest      cubic     v4(biharmonic interpolation)
        if (nFlag == 0)
            error = sqrt(sum(sum((ZI - Data).^2)/(101*101)));
        elseif (nFlag == 1)
            error = sum(sum(abs(ZI - Data))/(101*101));
        end
        Error_opt(pnt) = Error_opt(pnt) + error;

        Nlevels = Nlevels + jmp;
        if (pnt < Npnt)
            Slevel(2 , pnt+1) = Nlevels;        
        end
    end
end

Cost_opt = Cost_opt / Navg;
Error_opt = Error_opt / Navg;

Cost = [Cost_unif ; Cost_unif ; Cost_opt];
for kk = 2 : length(Slevel)
    Cost(: , kk) = Cost(: , kk) + Cost(: , kk-1);
end
Cost = Cost/L*100;

% ========================================================================
figure(10); plot(Slevel(1,:), 20*log10(Error_unif(1:end)),'-dg', Slevel(2,:), 20*log10(Error_opt(1:end)), '-ob'); grid on;ylabel('Mean Error in dB'); xlabel('# of levels');
figure(12); plot(Slevel(1,:), Cost_unif(1:end)/L*100,'g', Slevel(2,:), Cost_opt(1:end)/L*100,'b'); grid on; ylabel('Percentage of sensors'); xlabel('# of levels');
figure(24); plot(Slevel(1,:), Cost(2,:),'g', Slevel(2,:), Cost(3,:),'b'); grid on; ylabel('Percentage of sensors'); xlabel('# of levels');

Relative_costs = [sum(Cost_opt)/L    sum(Cost_unif)/L];
pause(1);

%
Z_all = griddata2(Sensors(:,1),Sensors(:,2),Sensors(:,4),XI,YI,'v4');          % linear   nearest      cubic     v4(biharmonic interpolation)
if (nFlag == 0)
    error_v4_all = sqrt(sum(sum((Z_all - Data).^2)/(101*101)));
elseif (nFlag == 1)
    error_v4_all = sum(sum(abs(Z_all - Data))/(101*101));
end
All_Sensor_Error = [20*log10(error_v4_all)]

%figure(1); plot(Slevel(2:end), 20*log10(Error_learn_prop(1:end-1)),'r', Slevel(2:end), 20*log10(Error_learn_unif(1:end-1)),'g'); grid on;
%}