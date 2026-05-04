
clear
clc

%assumptions made:
% Solar panels don't degrade, have an areal density of 2.5 kg/m^2, and an efficiency of 32% BOL
% The path between Jupiter and the Sun is a straight line (although not true, in this case the velocity is super hyperbolic so it's a decent approximation)
% No space drag
% Thrust points exactly in the direction of motion at all times
% NEXT ion prop uses 2.6 kW, with Isp 4000s and mdot 2.05 mg/s
% Sun is the main influence with no regard to other masses when considering gravity
% Prop itself has negligible mass
% Batteries not included in mass

%OrbitalSetup
AU_m = 1.496*10^11; %AU to meter conversion
V_J = 1*10^3; %m/s - Heliocentric exit velocity from Jupiter.
r_J = 5.2*AU_m; %m - Average distance Jupiter is from the Sun
r_p = 0.14*AU_m; %m - perihelion distance
mu_s = 1.327*10^20; %m^3/s^2 - sun gravitation parameter
r = linspace(r_J,r_p,200000); %m - Just discretizing the distance between Jupiter and the Sun. Note, for relevant accuracy, high amount of discretization is required.

%IonPropSetup
s_yr = 3600*24*365.25; %seconds per year
t_b = 1.7*s_yr; %s - burn time from Jup to Sun
m_dot = 2.05*10^-6; %kg/s - mass flow rate of the NEXT ion prop
m_b = t_b*m_dot; %kg - mass of the fuel
V = zeros(1,length(r)); %m/s - velocity at every point
t = zeros(1,length(r)); %s- time at every point
m = zeros(1,length(r)); %kg - mass at every point
m_p = zeros(1,length(r)); %kg - propellant mass at every point
V(1) = V_J; %m/s - first velocity
t(1) = 0; %s - t = 0
m(1) = 483 + 320 + m_b; %kg - 483 comes from everything from the solar dive and beyond, 320 from the solar panels, and last one is propellant mass.
m_p(1) = m_b; %kg - first propellant mass

%Velocity at Perihelion (V_p) prediction (theoretical calcs)
R = (483 + 320 + m_b)/(483 + 320); %mass ratio
dV_pred = 4000*9.81*log(R); %m/s - deltaV from that amount of fuel
V_p_noburn = sqrt((V_J)^2+2*mu_s*(1/r_p-1/r_J)); %m/s - v_p if you did not burn at all
V_p_pred_best = dV_pred + V_p_noburn; %m/s - assumes burn right at perihelion
V_p_pred_worst = sqrt((V_J+dV_pred)^2+2*mu_s*(1/r_p-1/r_J)); %m/s - assumes burn right after jupiter

%acceleration integration
for i = 1:length(r)-1
    dt = -(r(i+1) - r(i))/V(i); %s - change in time between positional steps
    dm = dt*m_dot; %kg - change in mass between positional steps
    m_p(i+1) = m_p(i) - dm; %kg - propellant mass on the next step
    if m_p(i+1) < 0 %kg - make sure we don't get negative propellant mass
        m_p(i+1) = 0;
    end
    m(i+1) = 483 + 320 + m_p(i+1); %kg - mass on next step
    if m_p(i) > 0
        V(i+1) = V(i) + (mu_s/r(i)^2 + 0.084/m(i))*dt; %m/s - new velocity, with gravity and thrust from propulsion
    else
        V(i+1) = V(i) + (mu_s/r(i)^2)*dt; %m/s - new velocity, with gravity and no thrust from propulsion
    end
    t(i+1) = t(i) + dt; %s- time at the next position
end

t_y = t/s_yr; %yr - time in years
t_y_end = t_y(length(r)); %yr - time it took
V_p = V(length(r)); %m/s - actual velocity at perihelion

%User Friendly(er)
disp("Velocity at Perihelion:")
disp(V_p + " m/s")
disp("Time Taken:")
disp(t_y_end + " years")
disp("Mass Ratio:")
disp(R)
disp("Delta V Gained:")
disp(V_p-V_p_noburn + "m/s")
