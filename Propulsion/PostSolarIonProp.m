clear
clc
%Note:
%m_0 is final mass (dry mass)
%m_1 is mass before station keeping
%m_2 is mass before deep space propulsion
%m_3 is mass before solar take off
%m_4 is mass before jupiter oberth
%m_5 is mass before burn to jupiter (or initial mass!)

%% Ion Prop Calcs
%Spacecraft Properties
m_d = 101.8; %kg, dry mass

%NEXT Ion Prop properties
u_eq = [4000 3735 3525 3240 3015 2885 2745 2450 2400 1855 1585].*9.81; %m/s - equivalent velocity for NEXT ion prop
P = [2.585 2.300 2.090 1.825 1.635 1.520 1.415 1.210 1.175 0.865 0.740].*1000; %watts - power of the NEXT ion prop
m_dot = 2.05*10^-6; %kg/s - mass flow rate of the ion prop

%Energy Sizing Calcs
t_max = 40; %years - max years allowed
t = 0:0.1:t_max; %years - discretation of time
for i = 1:length(u_eq)
m_gen(i,:) = P(i)./(4.2*0.984.^t); %kg - mass of the generator
m_0(i,:) = m_d + m_gen(i,:); %kg - dry mass of the craft including generator
end

%Station Keeping
y_s = 365.25*24*3600;
T = 8*y_s; %station keeping time in seconds
R = 250*10^6; % Radius of rotation in meters
V_t = 2*pi*R/y_s; % tangential velocity
R_1 = exp(V_t^2*T./(R.*u_eq)); %first mass ratio
for i = 1:length(u_eq)
m_1(i,:) = m_0(i,:).*R_1(i); %kg
end

%Deep Space Prop
delta_m = t*y_s*m_dot; %kg - change in mass
m_2 = m_1 + delta_m; %kg - mass with post solar burn fuel
R2 = m_2./m_1; %post solar burn mass ratio
for i = 1:length(u_eq)
delta_v_2(i,:) = u_eq(i).*log(m_2(i,:)./m_1(i,:)); %m/s - delta V gain from that mass
v_bar(i,:) = u_eq(i) - delta_v_2(i,:).*(m_1(i,:)./delta_m); %m/s - average velocity from Sun to imaging
v_bar(i,1) = 0;
v_inf(i,:) = 77.017*10^3 - (v_bar(i,:).*t+delta_v_2(i,:).*(t_max-t))/t_max; %m/s - effective v_inf needed after solar Oberth
end


%% Solar Sail Min 
%variables
AU_m = 1.496*10^11;
rho = 0.9; %sail reflectivity
mu_s = 1.327*10^20; %m^3/s^2 - Sun gravitational parameter
E_r = 0.5*(102.8*10^3)^2-mu_s/(0.14*AU_m); %m^2/s^2 - Heliocentric energy at perhelion
%E_r = 5.677*10^9; %m^2/s^2
r_p = 0.14*AU_m; %m - perhelion radius
v_p = sqrt(2*(E_r+mu_s/r_p)); %m/s - velocity at perhelion
rho_s = 0.001; %kg/m^2 - areal density of sail
m_limit = 16800; %kg - limit mass (max for Falcon Heavy Mars)

%Paper Calculations
mu_eff_req = 0.5.*r_p.*(v_p.^2-v_inf.^2); %m^3/s^2 - effective gravitational parameter needed to reach imaging within 40 years
Am3_req = (mu_s-mu_eff_req)/((1.016*10^17)*(1+rho)); %m^2/kg - total area to mass ratio required
Am2_req = Am3_req./(1-rho_s.*Am3_req); %m^2/kg - area to mass without sail ratio required
A_req = Am2_req.*m_2; %m^2 - sail area required
m_sail = A_req.*rho_s; %kg - mass of the sail
Fake = m_sail < 0; %checks for invalid cases
m_sail(Fake) = NaN; %gets rid of them
m_3 = m_2 + m_sail; %kg - mass with sail
R3 = m_3./m_2; %mass ratio from sail
 
%Final Plots
close all
figure
R_solar = R3.*R2; %mass ratio from sail and post solar ion burn
for i = 1:2:11
plot(t,R_solar(i,:),"LineWidth",1.4)
hold on
end
hold on
% plot(t,m_limit*ones(length(rho_s)),"LineWidth",1.4,"LineStyle","--")
xlabel("Burn time (y)","FontSize",17)
ylabel("Solar Mass Ratio","FontSize",17)
title("Solar Mass Ratio as Burn Time Increases at " + rho_s*1000 + " g/m^2 and 0.14 AU")
grid on
ax = gca;
ax.FontSize = 14;
legend("2.585 kW", "2.090 kW", "1.635 kW", "1.415 kW", "1.175 kW", "0.740 kW")

figure
for i = 1:2:11
plot(t,A_req(i,:),"LineWidth",1.4)
hold on
end
xlabel("Burn time (y)","FontSize",17)
ylabel("Sail Area (m^2)","FontSize",17)
title("Sail Area as Burn Time Increases at " + rho_s*1000 + " g/m^2 and 0.14 AU")
grid on
ax = gca;
ax.FontSize = 14;
legend("2.585 kW", "2.090 kW", "1.635 kW", "1.415 kW", "1.175 kW", "0.740 kW")