syms theta_1(t) theta_2(t) theta_3(t) real
%syms q_dot_1 q_dot_2 q_dot_3 real
%syms q_2dot_1 q_2dot_2 q_2dot_3
syms d4 d2 a2 a3 real

q_2dot = [diff(theta_1(t),2);diff(theta_2(t),2);diff(theta_3(t),2)]
q_dot  = [diff(theta_1(t));diff(theta_2(t));diff(theta_3(t))]

d2 = 149.09/1000; %m
d4 = 433.07/1000; %m
a2 = 431.8/1000; %m
a3 = -20.32/1000; %m

d(1,1) = 2.4574 + 1.7181*cos(theta_2(t))*cos(theta_2(t))+.443*sin(theta_2(t)+theta_3(t))*sin(theta_2(t)+theta_3(t))-.0324*cos(theta_2(t))*cos(theta_2(t)+theta_3(t))-.0415*cos(theta_2(t)+theta_3(t))*sin(theta_2(t)+theta_3(t))+.9378*cos(theta_2(t))*sin(theta_2(t)+theta_3(t));
d(1,2) = 2.2312*sin(theta_2(t)) - .0068*sin(theta_2(t)+theta_3(t))-.1634*cos(theta_2(t)+theta_3(t));
d(1,3) = -.0068*sin(theta_2(t)+theta_3(t))-.1634*cos(theta_2(t)+theta_3(t));
d(2,1) = d(1,2);
d(2,2) = 5.1285+.9378*sin(theta_3(t))-.0324*cos(theta_3(t));
d(2,3) = .4424+.4689*sin(theta_3(t))-.0162*cos(theta_3(t));
d(3,1) = d(1,3);
d(3,2) = d(2,3);
d(3,3) = 1.0236;

c111 = 0;
c121 = 0.0207 - 1.2752*cos(theta_2(t))*sin(theta_2(t)) + 0.4429*cos(theta_3(t))*sin(theta_3(t)) - 0.8859*sin(theta_2(t))*sin(theta_3(t))*sin(theta_2(t)+theta_3(t)) +0.0325*cos(theta_2(t))*sin(theta_2(t)+theta_3(t)) + 0.4689*cos(theta_2(t))*cos(theta_2(t)+theta_3(t)) - 0.4689*sin(theta_2(t))*sin(theta_2(t)+theta_3(t))-0.0461*cos(theta_2(t)+theta_2(t)) - 0.0415*cos(theta_2(t)+theta_3(t))*cos(theta_2(t)+theta_3(t)) - 0.0163*sin(theta_3(t));
c131 = 0.0207 + 0.4429*cos(theta_2(t))*sin(theta_2(t)) + 0.4429*cos(theta_3(t))*sin(theta_3(t)) - 0.8859*sin(theta_2(t))*sin(theta_3(t))*sin(theta_2(t)+theta_3(t)) +0.0163*cos(theta_2(t))*sin(theta_2(t)+theta_3(t)) + 0.4689*cos(theta_2(t))*cos(theta_2(t)+theta_3(t)) - 0.0415*cos(theta_2(t)+theta_3(t))*cos(theta_2(t)+theta_3(t));
c211 = c121;
c221 = 1.8181*cos(theta_2(t)) + 0.1634*sin(theta_2(t)+theta_3(t)) - 0.0068*cos(theta_2(t)+theta_3(t));
c231 = 0.1634*sin(theta_2(t)+theta_3(t)) - 0.0068*cos(theta_2(t)+theta_3(t));
c311 = c131;
c321 = c231;
c331 = 0.1634*sin(theta_2(t)+theta_3(t)) - 0.0068*cos(theta_2(t)+theta_3(t));
c112 = -c121;
c122 = 0;
c132 = 0;
c212 = c122;
c222 = 0;
c232 = 0.4689*cos(theta_3(t)) + 0.0162*sin(theta_3(t));
c312 = 0;
c322 = c232;
c332 = 0.4689*cos(theta_3(t)) + 0.0162*sin(theta_3(t));
c113 = -c131;
c123 = -c132;
c133 = 0;
c213 = c123;
c223 = -c232;
c233 = 0;
c313 = c133;
c323 = c233;
c333 = 0;

c(1,1) = c111*q_dot(1)+c211*q_dot(2)+c311*q_dot(3);
c(2,1) = c112*q_dot(1)+c212*q_dot(2)+c312*q_dot(3);
c(3,1) = c113*q_dot(1)+c213*q_dot(2)+c313*q_dot(3);
c(1,2) = c121*q_dot(1)+c221*q_dot(2)+c321*q_dot(3);
c(2,2) = c122*q_dot(1)+c222*q_dot(2)+c322*q_dot(3);
c(3,2) = c123*q_dot(1)+c223*q_dot(2)+c323*q_dot(3);
c(1,3) = c131*q_dot(1)+c231*q_dot(2)+c331*q_dot(3);
c(2,3) = c132*q_dot(1)+c232*q_dot(2)+c332*q_dot(3);
c(3,3) = c133*q_dot(1)+c233*q_dot(2)+c333*q_dot(3);

syms g h real

g(1) = 0;
g(2) = -48.5564*cos(theta_2(t)) + 1.0462*sin(theta_2(t))+.3683*cos(theta_2(t)+theta_3(t))-10.6528*sin(theta_2(t)+theta_3(t));
g(3) = .3683*cos(theta_2(t)+theta_3(t)) - 10.6528*sin(theta_2(t)+theta_3(t));
g = g';

h(1) = a3*cos(theta_1(t))*cos(theta_2(t)+theta_3(t)) + d4*cos(theta_1(t))*sin(theta_2(t)+theta_3(t))+a2*cos(theta_1(t))*cos(theta_2(t))-d2*sin(theta_1(t));
h(2) = a3*sin(theta_1(t))*cos(theta_2(t)+theta_3(t)) + d4*sin(theta_1(t))*sin(theta_2(t)+theta_3(t))+a2*sin(theta_1(t))*cos(theta_2(t))+d2*cos(theta_1(t));
h(3) = -a3*sin(theta_2(t)+theta_3(t))+d4*cos(theta_2(t)+theta_3(t))-a2*sin(theta_2(t));
h = h';

tau = d*q_2dot+c*q_dot+g;

% plotODESolve([0,0,0],tau,[theta_1(t) == 0; theta_2(t) == 0; theta_2(t) == 0])
% plotODESolve([1,0,0],tau,[theta_1(t) == 0; theta_2(t) == 0; theta_2(t) == 0])
% plotODESolve([0,1,0],tau,[theta_1(t) == 0; theta_2(t) == 0; theta_2(t) == 0])
% plotODESolve([0,0,1],tau,[theta_1(t) == 0; theta_2(t) == 0; theta_2(t) == 0])



tauTest = [0,0,1]
Theta = [0,0,0,0,0,0]

clearvars Data

for i = 1:100
    LinearizeSpot = [theta_1(t) == Theta(1); theta_2(t) == Theta(3) ;theta_3(t) == Theta(5);]
plot([1:length(Data(:,1))],Data)    [Theta,Meaning] = plotODESolve(tauTest,tau,LinearizeSpot,Theta)
    Data(i,:) = [Theta,Meaning']
end

plot([1:length(Data(:,1))],Data(:,[1,3,5]))

function [out,S] = plotODESolve(tauInput,tau,Linearization,Conditions)
    tauTest = tauInput%[0;0;0]

    tauEqns = [tauTest(1) == tau(1);tauTest(2) == tau(2);tauTest(3) == tau(3)]
    [SymSys,S] = odeToVectorField(tauEqns,Linearization);
    Sys = matlabFunction(SymSys,'vars',{'t','Y'})
    tspan = [0 1];
    Y0 = Conditions %[0 0 0 0 0 0]
    %Sys1 = @(t,Y) Sys(Y)
    %options = odeset('RelTol',1e-5,'Stats','on')%,'OutputFcn',@odeplot)
    [t,y] = ode45(@(t,Y) Sys(t,Y), tspan, Y0)%, options)

    out = y(length(y(:,1)),:)%y(length(y(:,1)),[1,3,5])
    
%     figure('Name',strcat('Tau Test: ' , num2str(tauInput)))
%     plot(t, y)%y(:,[1,3,5]))
%     grid
end