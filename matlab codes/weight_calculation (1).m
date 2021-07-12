function weight_calculation

%omega=14kHz
%Q_0=50

T=1
a=0.1   %%%typically in the range [0...1]
%---> T(a+delta_a(i)+m(i))

% --> first step: T(a)=Q_0/omega_0*exp(a*slope)

%--> second step: mask matrix (m for each node) e.g. m=[0, 1,1,0.1,...,
%N]*m_const; calculate weights with T(a+m(i)), %caluclate weights and spectralrad

%--> final step: delta_a=amp*delta_a_const; calcualte T(a+m+delta_a);
%caluclate weights and spectralrad


spectralstep=[];
theta=[];
for thetastep=1:10
theta(thetastep)=(0.1*thetastep)*T
alpha=0.2

weight_slope= (w_max-w_min)/(theta_min-theta_max)

weight=[];
for i=1:N
    for j=1:N
        if i==j
            weight(i,j)=alpha
        else
        timediff=(j-i)*theta(thetastep)
        weight(i,j)=weight_slope*(timediff-theta_min)+w_max
        %y=y2-y1/x2-x1*(x-x1)+y1
        end
    end
end
EV=eig(weight)
spectralrad(thetastep)=max(abs(EV))
end
figure
plot(theta,spectralrad)