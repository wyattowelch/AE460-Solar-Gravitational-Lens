clear
clc

%% Solar Sail Min 
%variables
AU_m = 1.496*10^11;
rho = 0.9;
v_p = 113.67*10^3; %m/s
v_inf = 123.17*10^3; %m/s
r_p = 0.14*AU_m; %m
I_sp = 338; %s
g_0 = 9.81; %m/s^2
rho_s = 0.00105:0.000001:0.0012 %kg/m^2
k_s = (1.016*10^17)*(1+rho)
S = (rho_s*r_p*I_sp*g_0/k_s).^2
a = r_p*rho_s.^2
b = 2*S*k_s - 2*r_p*rho_s
c = r_p - S*r_p*v_inf^2 - 2*S*1.327*10^20
Am_min = (-b - sqrt(b.^2-4.*a.*c))./(2.*a)
m_d = 150 %kg, dry mass
m_limit = 8900 %kg, limit mass (max for Atlas V GEO)

%Paper Calculations
mu_eff_min = 1.327*10^20 - (1.016*10^17)*(1+rho).*Am_min;
delta_V_r = sqrt(v_inf^2 + 2*mu_eff_min./r_p)-v_p;
m3_mc_min = exp(delta_V_r/I_sp/g_0);
m3_m4_min = m3_mc_min.*(1./(1-rho_s.*Am_min));

am_3_min = Am_min/(1-rho_s.*Am_min)

%Sail Density Plot
plot(rho_s,m3_m4_min,"LineWidth",1.4)
xlabel("Sail Density (kg/m^2)","FontSize",17)
ylabel("(m_3/m_4)_{min}","FontSize",17)
title("Solar Oberth Mass Ratio as Sail Density Increases")
grid on
ax = gca;
ax.FontSize = 14;
 
%Final Mass Plot
figure
m1 = 1.41*7.61*m3_m4_min*150
plot(rho_s,m1,"LineWidth",1.4)
hold on
plot(rho_s,m_limit*ones(length(rho_s)),"LineWidth",1.4,"LineStyle","--")
xlabel("Sail Density (kg/m^2)","FontSize",17)
ylabel("(m_1)_{min} (kg)","FontSize",17)
title("Total Mass as Solar Sail Density Increases")
grid on
ax = gca;
ax.FontSize = 14;