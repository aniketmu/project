function cantilever

global damping1
global omega_0_square
global freq_sound
global F_ext
global alpha
global beta
global gamma

global a_f_neu
global R1
global u_dc
global kalib
global tau


%%%Canti-werte%%%%%%%%%%%%%%%
%%%canti1%%%%%%%
Q1=40;
R1=25;
freq1=14000;
omega_0=freq1*2*pi;
omega_0_square=omega_0^2;
damp1=1/(2*Q1);
damping1=damp1*2*omega_0;

alpha=749.3702;
beta=1.0066e+03;
gamma=4.2588e+07;

kalib=1.008e6;
tau=0.001;

options = odeset('MaxStep',1e-6);

%%%%%%%%%%Schall-Werte%%%%%%%%%%%
F_ext=3.15;
freq_sound_list=[264, 297, 330, 352, 396, 440, 495, 528];
number_runs=length(freq_sound_list)


%%%%%%%%%Reservoir Werte%%%%%%%%%%%%%%%%%%
number_neurons=10
theta=2*Q/omega_0
a_f0=0.9
const_a=0.01
const_m=0.1
settling=1

x_out=[];
a_f_list=[];
delta_a_list=[];
m_list=[];
freq_sound_list2=[];

rng(0,'twister');
m_initial=randi([0 1], 1,number_neurons)

if settling==1
    y_start=[0 1e-9 0 0];
    t_final = 0.1;
    zeit = [0 t_final];
    [T,Y] = ode15s(@cantimodell2a,zeit,y_start,options);
    figure
    plot(T,Y(:,2),T,Y(:,4))
end

M = 10000;


a_f_neu=a_f0+const_m*m_initial(1)
a_f_list=[a_f_list,a_f_neu];
m_list=[m_list,const_m*m_initial(1)];
delta_a_list=[delta_a_list, 0];

time_start=t_final;
time_end=t_final;
for time=1:number_runs%20    
    freq_sound=freq_sound_list(number_run)
    
    time_end=time_end+time*theta*number_neurons;
    time_start=time_start+(time-1)*theta*number_neurons;
    
    for neuron=1:number_neurons
            freq_sound_list2=[freq_sound_list2,freq_sound];
            t_final = time_start+n*theta;
            t_start=time_start+(n-1)*theta;
            %dt=t_final/M;
            zeit = [t_start t_final];
            %if t_start>0
                y_start=Y(end,:);
            %else
            %    y_start=[0 1e-9 0];
            %end
            %[T,Y] = ode15s(@cantimodell1b,zeit,y_start2,options); %%single cantilevers
            [T,Y] = ode15s(@cantimodell2a,zeit,y_start,options); %%two cantilevers

            [env1a,env2a]=envelope(Y(:,4),5000,'peak');
            %amptest1a=min(Y(startzeit2,4));
            %amptest2a=max(Y(startzeit2,4));
            %amptesta=amptest2a-amptest1a;
            ampchecka=mean(env1a-env2a);
            
            delta_a=const_a*x_out2(end)
            delta_a_list=[delta_a_list,delta_a];
            
            m_value=m_initial(neuron)*const_m
            m_list=[m_list,m_value];
            
            a_f_neu=a_f0+delta_a+const_m*m_initial(dummy_m)
            a_f_list=[a_f_list,a_f_neu];
            
            x_out=[x_out,ampchecka];
            x_out2=[x_out2,env1a];
            
            figure
            hold on
            plot(T,Y)
            plot(T,env1a)
            plot(T,env2a)
            fdgsga
    end
end

figure
hold on
plot(freq_sound_list2)
plot(m_list)
plot(delta_a_list)
plot(a_f_list)
plot(x_out)





% fshadhsha
% startzeit=find(T>0.15);
% startzeit2=find(T>0.15);
% [env1a,env2a]=envelope(Y(:,4),5000,'peak');
% amptest1a=min(Y(startzeit2,4));
% amptest2a=max(Y(startzeit2,4));
% amptesta=amptest2a-amptest1a;
% ampchecka=mean(env1a(startzeit2)-env2a(startzeit2));
% 
% [env1b,env2b]=envelope(Y(:,8),5000,'peak');
% amptest1b=min(Y(startzeit2,8));
% amptest2b=max(Y(startzeit2,8));
% amptestb=amptest2b-amptest1b;
% ampcheckb=mean(env1b(startzeit2)-env2b(startzeit2));
% 
% %text=strcat('x_AC1+2 u_Dc =',mat2str(u_dc1),'_Fext_',mat2str(F_ext));
% %plot(T(:),Y(:,4),'gx')
% %plot(T(:),Y(:,8),'ro')
% %plot(T(startzeit2),env1a(startzeit2),T(startzeit2),env2a(startzeit2));%,T(startzeit),0.5*(env1b(startzeit)-env2b(startzeit))+mean(env2b))
% 
% if abs(amptesta)<abs(ampchecka)
%     amp1=[amp1,amptesta];
%     x_DC1=[x_DC1,mean(Y(startzeit,2))];
% else
%     amp1=[amp1,ampchecka];
%     x_DC1=[x_DC1,0.5*mean(env1a(startzeit)-env2a(startzeit))+mean(env2a(startzeit))];
% end
% if abs(amptestb)<abs(ampcheckb)
%     amp2=[amp2,amptestb];
%     x_DC2=[x_DC2,mean(Y(startzeit,6))];
% else
%     amp2=[amp2,ampcheckb];
%     x_DC2=[x_DC2,0.5*mean(env1b(startzeit)-env2b(startzeit))+mean(env2b(startzeit))];
% end
% laenge=length(startzeit2);
% % 
% dt2=mean(T(2:end)-T(1:end-1));
% %std(T(2:end)-T(1:end-1))
% 
% NFFT=2^nextpow2(laenge);
% FFT_sensor1=fft(Y(startzeit2,4),NFFT);
% FFT_sensor2=fft(Y(startzeit2,8),NFFT);
% % Y_laut=sin(2*pi*freq.*T);
% % FFT_laut=fft(Y_laut(end-laenge+1:end),NFFT);
% Px_sensor1=FFT_sensor1.*conj(FFT_sensor1)/(laenge*NFFT); %Power of each freq components
% Px_sensor2=FFT_sensor2.*conj(FFT_sensor2)/(laenge*NFFT); %Power of each freq components
% fVals2=1/dt2.*(0:NFFT/2-1)/NFFT;
% size(fVals2)
% size(Px_sensor1)
% 
% amp1_fft=[amp1_fft,max(Px_sensor1(10:NFFT/2))];
% amp2_fft=[amp2_fft,max(Px_sensor2(10:NFFT/2))];
% % fiamp1_fft=[amp1_fft,max(Px_sensor1(10:NFFT/2))];gure
% 
% % figure
% % hold on
% % plot(fVals2',Px_sensor1(1:NFFT/2),'b')
% % plot(fVals2',Px_sensor2(1:NFFT/2),'r')
% % ghjgkh
% freq_soundlist=[freq_soundlist,freq_sound];
% end
% 
% %text=strcat('b = ',mat2str(b_f));
% %plot(freq_sounbdlist,amp,'DisplayName',text)
% % 
% 
% %%%%%maximal amplitude etc%%%%%%%
% [amp_maxfind,maxind]=max(amp1);
% amp_max1=[amp_max1,amp_maxfind];
% x_DC1_max=[x_DC1_max,x_DC1(maxind)];
% freq_res1=[freq_res1,freq_soundlist(maxind)];
% amp_min1=[amp_min1,amp1(end)];
% 
% [amp_maxfind2,maxind2]=max(amp2);
% amp_max2=[amp_max2,amp_maxfind2];
% x_DC2_max=[x_DC2_max,x_DC2(maxind2)];
% freq_res2=[freq_res2,freq_soundlist(maxind2)];
% amp_min2=[amp_min2,amp2(1)];
% 
% %%%%%maximal amplitude in fft etc%%%%%%%
% [amp_maxfind3,maxind3]=max(amp1_fft);
% amp_max1_fft=[amp_max1_fft,amp_maxfind3];
% freq_res1_fft=[freq_res1_fft,freq_soundlist(maxind3)];
% [amp_maxfind4,maxind4]=max(amp2_fft);
% amp_max2_fft=[amp_max2_fft,amp_maxfind4];
% freq_res2_fft=[freq_res2_fft,freq_soundlist(maxind4)];
% 
% b_value=[b_value,b_f1];
% 
% datei_amp=fopen(char(strcat('ode-solver/coupled_cantis_2021-01-04/amplitude_vs_freq_2cantis+feedback+schall_bWert_',mat2str(b_f1),'_uDC1_',mat2str(u_dc1),'_freq1_',mat2str(freq1),'_Q1_',mat2str(Q1),'_R1_',mat2str(R1),'_uDC2_',mat2str(u_dc2),'_freq2_',mat2str(freq2),'_Q2_',mat2str(Q2),'_R2_',mat2str(R2),'_mit_Hochpass.dat')),'w');
% for dummy=1:length(amp1)
% %    %fprintf(datei_amp,'%.2e %.2e %.2e %.2e %.2e %.2e\n', F_ext,a_value(dummy), amp_max1(dummy),amp_min(dummy),ampdiff(dummy), x_DC2(dummy));
%     fprintf(datei_amp,'%.2e %.2e %.2e %.2e %.2e\n', freq_soundlist(dummy),amp1(dummy),amp2(dummy),amp1_fft(dummy),amp2_fft(dummy)); 
% end
% fclose(datei_amp);
% end
% 
% %ampd
% figure
% hold on
% plot(b_list,amp_max1,'g')
% plot(b_list,amp_max2,'b')
% plot(b_list,freq_res1,'g')
% plot(b_list,freq_res2,'b')
% plot(b_list,amp_max1_fft,'r')
% plot(b_list,amp_max2_fft,'m')
% plot(b_list,freq_res1_fft,'g')
% plot(b_list,freq_res2_fft,'b')
% 
% datei_amp=fopen(char(strcat('ode-solver/coupled_cantis_2021-01-04/amplitude+freq_res+DC_vs_b-value_2cantis+feedback+schall_Fext_',mat2str(100/0.656*(F_ext-3.144)+100),'_uDC1_',mat2str(u_dc1),'_freq1_',mat2str(freq1),'_Q1_',mat2str(Q1),'_R1_',mat2str(R1),'_uDC2_',mat2str(u_dc2),'_freq2_',mat2str(freq2),'_Q2_',mat2str(Q2),'_R2_',mat2str(R2),'_mit_Hochpass.dat')),'a');
% for dummy=1:length(amp_max1)
% %    %fprintf(datei_amp,'%.2e %.2e %.2e %.2e %.2e %.2e\n', F_ext,a_value(dummy), amp_max1(dummy),amp_min(dummy),ampdiff(dummy), x_DC2(dummy));
%     fprintf(datei_amp,'%.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e\n', F_ext, 100/0.656*(F_ext-3.144)+100,b_value(dummy), amp_max1(dummy),x_DC1_max(dummy), freq_res1(dummy),amp_min1(dummy),amp_max1_fft(dummy),freq_res1_fft(dummy),amp_max2(dummy),x_DC2_max(dummy),freq_res2(dummy),amp_min2(dummy),amp_max2_fft(dummy),freq_res2_fft(dummy)); 
% end
% fclose(datei_amp);
% sdgadfgad
% 
% 
% 
% diff=[ampdiff,abs(amp(2)-amp(1))];
% [amp_maxfind2,maxind2]=max(amp_max1);
% a_crit=[a_crit,a_value(maxind2(1))];
% amp_max=[amp_max,amp_maxfind2(1)];
% DC_max=[DC_max,x_DC2(maxind2(1))];
% amp_min2=[amp_min2,amp_min(maxind2(1))];
% u_dc_list=[u_dc_list,u_dc];
% fext_list=[fext_list,F_ext];
% 
% 
% end
% %freq_res2
% %amp_max1
% 
% %hgfjgf
% 
% %text=strcat('x_AC u_Dc =',mat2str(u_dc),'_Fext_',mat2str(F_ext));
% %plot(a_value,amp_max,'DisplayName',text);
% %plot(a_value,amp_min2)
% % text2=strcat('x_DC u_Dc =',mat2str(u_dc),'_Fext_',mat2str(F_ext));
% % plot(a_value,x_DC,'DisplayName',text2);
% 
% %  datei_amp2=fopen(char(strcat('ode-solver/a-crit+ampmax+DC_vs_uDC_versch_Fext_canti+feedback+schall__Fext_',mat2str(F_ext),'_freq_',mat2str(freq_res),'_Q_',mat2str(Q),'_R_',mat2str(R),'_mitHochpass.dat')),'a');
% %  for dummy=1:length(amp_max)
% %  fprintf(datei_amp2,'%.2e %.2e %.2e %.2e %.2e .2%e\n', fext_list(dummy),u_dc_list(dummy), a_crit(dummy), amp_max(dummy),amp_min2(dummy),DC_max(dummy));
% %  end
% %  fclose(datei_amp2);
%  end
%  
% %hold off
% %savefig(figamp,char(strcat('canti_modell_matlab_results/amplitude_vs_fext_canti+feedback+schall_uDC_',mat2str(u_dc),'_freq_',mat2str(freq_res),'_Q_',mat2str(Q),'_R_',mat2str(R),'_2.fig')));
% %close(zeitplot);
% safDSA
% maxfig=figure;
% hold on
% text=strcat('a_crit');
% plot(u_dc_list,a_crit,'DisplayName',text)
% text=strcat('AC amp_max');
% plot(u_dc_list,amp_max,'DisplayName',text)
% text=strcat('DC amp');
% plot(u_dc_list,DC_max,'DisplayName',text)
% 
% %end
% hold off
% savefig(maxfig,char(strcat('canti_modell_matlab_results/acrit+amp-max+DC_vs_uDC_canti+feedback+schall_Fext_',mat2str(F_ext),'_freq_',mat2str(freq_res),'_Q_',mat2str(Q),'_R_',mat2str(R),'.fig')));
% %end
% datei_max=fopen(char(strcat('canti_modell_matlab_results/acrit+amp-max+DC_vs_uDC_canti+feedback+schall_Fext_',mat2str(F_ext),'_freq_',mat2str(freq_res),'_Q_',mat2str(Q),'_R_',mat2str(R),'.dat')),'w');
% for dummy=1:length(amp_max)
% fprintf(datei_max,'%f %f %f %f\n', u_dc_list(dummy), a_crit(dummy),amp_max(dummy),DC_max(dummy));
% end
% fclose(datei_max);
% 
% % %%fft-Analyse%%
% % try
% %     laenge=length(Y_neu(end-laenge+1:end));
% % 
% % NFFT=2^nextpow2(laenge);
% % FFT_sensor=fft(Y_neu(end-laenge+1:end),NFFT);
% % Y_laut=sin(2*pi*freq.*T);
% % FFT_laut=fft(Y_laut(end-laenge+1:end),NFFT);
% % Px_sensor=FFT_sensor.*conj(FFT_sensor)/(laenge*NFFT); %Power of each freq components
% % Px_laut=FFT_laut.*conj(FFT_laut)/(laenge*NFFT); %Power of each freq components
% % fVals2=1/dt.*(0:NFFT/2-1)/NFFT;
% 
% % p_start1=find(fVals2<freq_start);
% % p_ende1=find(fVals2>freq_ende);
% % [pkf,lcf,wf] = findpeaks(Px_sensor(p_start1(end):p_ende1(1)),fVals2(p_start1(end):p_ende1(1)),'MinPeakDistance',1000);
% % [peakmaxf,peakindf]=max(pkf);
% % fft_freq(test,1)=lcf(peakindf(1));
% % fft_max(test,1)=pkf(peakindf(1));
% % fft_fwhm(test,1)=wf(peakindf(1));
% % pstartb=find(fVals2>frequenz(test)-50);

'ende'
end
% %-------------------------------------------------
% function dy = cantimodell1(t,y)
% 
% global damping
% global omega_0_square
% global freq_sound
% global F_ext
% global alpha
% global beta
% global gamma
% 
% global a_f
% global R
% global u_dc
% global kalib
% global u_ac
% global freq_act
% 
% 
% dy(1) = -beta.*y(1)+gamma.*(tanh(a_f.*kalib.*y(2)+u_dc)/R)^2;
% 
% %dy(4)=-y(4)/tau+y(3);
% 
% dy(2) = y(3);
% dy(3) = -damping*y(3)-omega_0_square.*y(2)+alpha*y(1)+F_ext.*sin(2*pi*freq_sound*t);
% 
% dy = dy';
% %t(end)
% end
%--------------------------------
% function dy = cantimodell1b(t,y)
% 
% global damping1
% global omega_01_square
% %global damping2
% %global omega_02_square
% global freq_sound
% global F_ext
% global alpha
% global beta
% global gamma
% global tau
% global a_f1
% global a_f2
% %global b_f2
% global R1
% %global R2
% global u_dc1
% %global u_dc2
% global kalib
% %global u_ac
% %global freq_act
% global offset
% 
% 
% dy(1) = -beta.*y(1)+gamma.*(tanh(a_f1.*(kalib.*y(4)+offset)+u_dc1)/R1)^2;
% dy(4)=-y(4)/tau+y(3);
% dy(2) = y(3);
% dy(3) = -damping1*y(3)-omega_01_square.*y(2)+alpha*y(1)+F_ext.*sin(2*pi*freq_sound*t);
% % 
% % dy(5) = -beta.*y(5)+gamma.*(tanh(b_f2.*(kalib.*y(8)+offset)+u_dc2)/R2)^2;
% % dy(8)=-y(8)/tau+y(7);
% % dy(6) = y(7);
% % dy(7) = -damping2*y(7)-omega_02_square.*y(6)+alpha*y(5)+F_ext.*sin(2*pi*freq_sound*t);
% 
% dy = dy';
% %t(end)
% end
% %-------------------------------------------------
function dy = cantimodell2a(t,y)

global damping1
global omega_0_square
global freq_sound
global F_ext
global alpha
global beta
global gamma
global tau
global a_f_neu
global R1
global u_dc1
global kalib

dy(1) = -beta.*y(1)+gamma.*(tanh(a_f_neu.*kalib.*y(4)+u_dc1)/R1)^2;
dy(4)=-y(4)/tau+y(3);
dy(2) = y(3);
dy(3) = -damping1*y(3)-omega_01_square.*y(2)+alpha*y(1)+F_ext.*sin(2*pi*freq_sound*t);

dy = dy';
%t(end)
end
% %-------------------------------------------------
% function dy = cantimodell2b(t,y)
% 
% global damping1
% global omega_01_square
% global damping2
% global omega_02_square
% global freq_sound
% global F_ext
% global alpha
% global beta
% global gamma
% global tau
% global b_f1
% global b_f2
% global a_f1
% global a_f2
% global R1
% global R2
% global u_dc1
% global u_dc2
% global kalib
% 
% dy(1) = -beta.*y(1)+gamma.*(tanh(a_f1.*kalib.*y(4)+b_f1.*kalib.*y(8)+u_dc1)/R1)^2;
% dy(4)=-y(4)/tau+y(3);
% dy(2) = y(3);
% dy(3) = -damping1*y(3)-omega_01_square.*y(2)+alpha*y(1)+F_ext.*sin(2*pi*freq_sound*t);
% 
% dy(5) = -beta.*y(5)+gamma.*(tanh(a_f2.*kalib.*y(8)+b_f2.*kalib.*y(4)+u_dc2)/R2)^2;
% dy(8)=-y(8)/tau+y(7);
% dy(6) = y(7);
% dy(7) = -damping2*y(7)-omega_02_square.*y(6)+alpha*y(5)+F_ext.*sin(2*pi*freq_sound*t);
% 
% dy = dy';
% %t(end)
% end
% % %-------------------------------------------------
% function dy = cantimodell2(t,y)
% 
% global damping
% global omega_0_square
% global freq_sound
% global F_ext
% global alpha
% global beta
% global gamma
% 
% global a_f
% global R
% global u_dc
% global kalib
% global freq_act
% global u_ac
% 
% 
% dy(1) = -beta.*y(1)+gamma.*((u_ac*sin(2*pi*freq_act*t)+u_dc)/R)^2;
% 
% dy(2) = y(3);
% dy(3) = -damping*y(3)-omega_0_square.*y(2)+alpha*y(1)+F_ext.*sin(2*pi*freq_sound*t);
% dy = dy';
% t(end)
% end
