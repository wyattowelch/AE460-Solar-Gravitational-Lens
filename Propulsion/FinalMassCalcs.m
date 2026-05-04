clear
clc

%m0 = Mass at launch (initial mass)
%m1 = Mass after GEO escape
%m2 = Mass after jupiter oberth
%m3 = Mass after sun dive
%m4 = Mass after imaging (dry mass)

%% Setup

%DELTA V's

dV1 = 12486.938942 - 5376.534624;
dV2 = 3607.670000;
dV3 = 41553.838106;

%Conversion Values

AU_m = 1.496*10^11; %AU to meters conversion
y_s = 3600*24*365.25; %year to second conversion

%Spacecraft
m4 = 101.8; %kg
Ueq_I = 22563; %m/s - U_eq for the deep space ion propulsion based of BET-300-P Config B in vaccum
m_I = 1; %kg - mass of the ion propulsion system
Ueq_C = 3615; %m/s - U_eq for methalox chemical propulsion in vaccum
eps_C = 0.08; %kg - structural coefficient of methalox
Ueq_H = 220*9.81; %m/s - U_eq for hydrazine
rho_A = 0.001; %kg/m^2 - sail areal density
rho = 0.9; %sail reflectivity
k = (1.016*10^17)*(1+rho); %kg-m/s^2 (Newtons) - sail parameter

%Sun to Imaging

mu_S = 1.327*10^20; %m^3/s^2 - gravitational parameter of the sun
rp_h = 0.14*AU_m; %m - perhelion
t_SuToIm = 40*y_s; %s - time from sun to imaging
v_inf_req = 650*AU_m/t_SuToIm; %m/s - approximate v_inf needed to reach imaging within t_SuToIm years
E_req = v_inf_req^2/2; %m^2/s^2 - Energy required at perhelion
% v_ph_req = sqrt(2*(E_req+mu_S/rp_h));
% v_ph = v_ph_req - dV3;

v_ph_req = 144391.576; %m/s - final perhelion velocity required
v_ph = v_ph_req - dV3; %m/s - entrance perhelion velocity

%% MASS CALCS

R1 = exp(dV1/Ueq_C); %mass ratio from burn from Earth GEO
R2 = exp(dV2/Ueq_C); %mass ratio from Jupiter Oberth

%Run different cases for sundive

dV3_fuel = linspace(0,dV3,1000); %m/s - variation in DeltaV from chemical burn
E_helio = (v_ph+dV3_fuel).^2./2 - mu_S/rp_h; %m^2/s^2 - it's the specific energy relative to Sun after fuel burns.
% E_req = v_inf_req^2/2; %m^2/s^2 - energy needed to get the required v_inf
E_req = v_ph_req^2/2 - mu_S/rp_h; %m^2/s^2 - energy needed to get the required v_inf
delta_E = E_req-E_helio; %m^2/s^2 - change in energy needed to get the required v_inf
R_sail_req = delta_E.*rp_h./k; %m^2/kg - area to mass ratio needed to get the required v_inf
R3_Sail = (1./(1-rho_A.*R_sail_req)); %mass ratio for the sail
R3_Prop = exp(dV3_fuel./Ueq_C); %mass ratio from the chemical burn

%Spiral/correction

t_imaging = (900-650)*AU_m/v_inf_req; %s - time spent imaging
R = 250*10^6; %m - Radius of rotation in meters
V_t = 2*pi*R/y_s; %m/s - tangential velocity
R4 = exp(V_t^2*t_imaging/(R*Ueq_I)); %first mass ratio
dVH = 100; %m/s
R4p5 = exp(dVH/Ueq_H);

%Mass ratio calcs

m3p5 = (m4+m_I)*R4; %kg - mass with the ion prop and fuel
m3 = m3p5*R4p5; %kg - mass with hydrazine
m2p5 = m3*R3_Sail; %kg - mass with sail
m2 = m2p5.*(R3_Prop.*(1-eps_C)./(1-R3_Prop.*eps_C)); %kg - mass with solar burn (if applicable)
m1 = m2*(R2*(1-eps_C)/(1-R2*eps_C)); %kg - mass with Jupiter burn
m0 = m1*(R1*(1-eps_C)/(1-R1*eps_C)); %kg - mass with Earth burn

dV3_fuel_max = interp1(m0,dV3_fuel,16800); %m/s - max allowable delta V from the chemical burn
A_sail = R_sail_req.*m2./exp(dV3_fuel./Ueq_C); %m^2 - area of the sail
m_sail = A_sail*rho_A; %kg - mass of the sail
mp_1 = (m0-m1)*(1-eps_C); %kg - mass of the methalox from Earth GEO burn
ms_1 = (m0-m1)*eps_C; %kg - structural mass for GEO burn
mp_2 = (m1-m2)*(1-eps_C); %kg - methalox fuel mass for Jupiter Oberth
ms_2 = (m1-m2)*eps_C; %kg - structural mass for Jupiter Oberth
mp_3 = ((m2-m_sail) - m3)*(1-eps_C); %kg - methalox fuel mass for aided solar oberth
ms_3 = ((m2-m_sail) - m3)*eps_C; %kg - structural mass for aided solar oberth
mp_4 = (m3p5-m_I)-m4; %kg - mass of the ion propulsion fuel for imaging
mp_H = m3-m3p5; %kg - mass of hydrazine fuel

close all

plot(dV3_fuel./1000,m0,"LineWidth",1)
grid on
title("Spacecraft Mass with Increasing Solar Burn")
xlabel("Delta V (km/s)")
ylabel("Mass (kg)")
yline(16800,Color='r',LineWidth=1)
xlim([0,dV3_fuel_max/1000])
ax = gca;
ax.FontSize = 14;

figure
plot(dV3_fuel./1000,A_sail,"LineWidth",1)
grid on
title("Sail Size with Increasing Solar Burn")
xlabel("Delta V (km/s)")
ylabel("Sail Area (m^2)")
xlim([0,dV3_fuel_max/1000])
ax = gca;
ax.FontSize = 14;

%For case where dV3_fuel = 0

clc

disp("Cruise/Imaging:")
disp("  Ion Prop Fuel Mass: " + mp_4 + " kg")
disp("  Hydrazine Fuel Mass: " + mp_H + " kg")
disp("Solar Oberth (No Aid):")
disp("  Sail Mass: " + m_sail(1) + " kg")
disp("  Sail Area: " + A_sail(1) + " m^2")
disp("Jupiter Oberth:")
disp("  Methalox Fuel Mass: " + mp_2(1) + " kg")
disp("  Methalox Structural Mass: " + ms_2(1) + " kg")
disp("Earth GEO:")
disp("  Methalox Fuel Mass: " + mp_1(1) + " kg")
disp("  Methalox Structural Mass: " + ms_1(1) + " kg")
disp("Total Mass: " + m0(1) + " kg")

