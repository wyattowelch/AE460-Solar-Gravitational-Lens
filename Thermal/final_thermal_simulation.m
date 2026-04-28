%% AE 460B Thermal Subsystem Final Simulation
% Two-node transient thermal model
%
% Nodes:
%   1) Spacecraft Bus
%   2) OBDH
%
% Outputs:
%   - Command window summary
%   - Plot 1: Temperature vs log(time)
%   - Plot 2: Temperature vs log(distance)
%

clear; clc; close all;

%% 
% 1) Constants
%% 
sigma = 5.670374419e-8;   % Stefan-Boltzmann constant [W/m^2/K^4]
S0    = 1361;             % Solar constant at 1 AU [W/m^2]

%% 
% 2) Mission / subsystem values
%% 
r_peri_AU    = 0.046;     % Perihelion distance [AU]
P_heater_max = 12;        % Thermal heater allocation [W]

% OBDH allowable operating range
T_obdh_min_C = -20;
T_obdh_max_C =  50;

%% 
% 3) Geometry / thermal assumptions
%% 
% Bus node
m_bus   = 120;            % effective thermal mass participation [kg]
cp_bus  = 900;            % [J/kg/K]
C_bus   = m_bus * cp_bus; % [J/K]

% OBDH node
m_obdh  = 7.3;            % [kg]
cp_obdh = 900;            % [J/kg/K]
C_obdh  = m_obdh * cp_obdh;

% Coupling between OBDH and bus
G_cond = 0.45;            % [W/K]

% Bus external thermal properties
alpha_bus = 0.20;         % effective solar absorptance
Aproj_hot = 0.60;         % projected area [m^2]

% Hot-case parameters
eta_shield  = 0.0039;     % effective solar attenuation factor
epsilon_hot = 0.70;       % hot-case emissivity
Arad_hot    = 0.90;       % hot-case effective radiator area [m^2]

% Cold-case parameters
epsilon_cold = 0.05;      % cold-case effective emissivity
Arad_cold    = 2.60;      % cold-case effective radiator area [m^2]

% OBDH radiation to bus / panel
epsilon_obdh = 0.70;      % effective emissivity
A_obdh_rad   = 0.25;      % [m^2]

Tspace = 3;               % deep-space sink temperature [K]

%% 
% 4) Initial conditions and heater settings
%% 
T0_bus  = 293.15;         % 20 C
T0_obdh = 293.15;         % 20 C

T_heater_on  = 289.3;     % [K]
T_heater_off = 294.3;     % [K]

%% 
% 5) Internal heat assumptions
%% 
% Hot case
Qbus_hot  = 25;           % [W]
Qobdh_hot = 14;           % [W]

% Cold case
Qbus_cold  = 10;          % [W]
Qobdh_cold = 10;          % [W]

%% 
% 6) HOT CASE transient at perihelion
%% 
t_hot_days = 3;                         % simulate 3 days
dt_hot     = 1;                         % [s]
t_hot      = 0:dt_hot:t_hot_days*24*3600;

T_bus_hot  = zeros(size(t_hot));
T_obdh_hot = zeros(size(t_hot));

T_bus_hot(1)  = T0_bus;
T_obdh_hot(1) = T0_obdh;

S_hot = S0 / (r_peri_AU^2);

for i = 1:length(t_hot)-1
    
    % Solar loading on bus after shielding / pointing
    Qsolar = alpha_bus * S_hot * Aproj_hot * eta_shield;
    
    % Bus radiation to space
    Qrad_bus = epsilon_hot * sigma * Arad_hot * (T_bus_hot(i)^4 - Tspace^4);
    
    % OBDH radiation to bus/panel environment
    Qrad_obdh_to_bus = epsilon_obdh * sigma * A_obdh_rad * ...
        (T_obdh_hot(i)^4 - T_bus_hot(i)^4);
    
    % Conduction between bus and OBDH
    Qcond = G_cond * (T_bus_hot(i) - T_obdh_hot(i));
    
    % Energy balances
    dT_bus  = (Qsolar + Qbus_hot - Qrad_bus - Qcond + Qrad_obdh_to_bus) / C_bus;
    dT_obdh = (Qobdh_hot + Qcond - Qrad_obdh_to_bus) / C_obdh;
    
    % Euler update
    T_bus_hot(i+1)  = T_bus_hot(i)  + dT_bus  * dt_hot;
    T_obdh_hot(i+1) = T_obdh_hot(i) + dT_obdh * dt_hot;
end

%% 
% 7) COLD CASE transient in deep space
%% 
t_cold_days = 120;                      % simulate 120 days
dt_cold     = 10;                       % [s]
t_cold      = 0:dt_cold:t_cold_days*24*3600;

T_bus_cold  = zeros(size(t_cold));
T_obdh_cold = zeros(size(t_cold));

T_bus_cold(1)  = T0_bus;
T_obdh_cold(1) = T0_obdh;

heater_on = false;
Qheater_arr = zeros(size(t_cold));

for i = 1:length(t_cold)-1
    
    % Heater thermostat logic
    if ~heater_on && T_bus_cold(i) <= T_heater_on
        heater_on = true;
    elseif heater_on && T_bus_cold(i) >= T_heater_off
        heater_on = false;
    end
    
    if heater_on
        Qheater = P_heater_max;
    else
        Qheater = 0;
    end
    
    % Bus radiation to space
    Qrad_bus = epsilon_cold * sigma * Arad_cold * (T_bus_cold(i)^4 - Tspace^4);
    
    % OBDH radiation to bus/panel environment
    Qrad_obdh_to_bus = epsilon_obdh * sigma * A_obdh_rad * ...
        (T_obdh_cold(i)^4 - T_bus_cold(i)^4);
    
    % Conduction
    Qcond = G_cond * (T_bus_cold(i) - T_obdh_cold(i));
    
    % Energy balances
    dT_bus  = (Qbus_cold + Qheater - Qrad_bus - Qcond + Qrad_obdh_to_bus) / C_bus;
    dT_obdh = (Qobdh_cold + Qcond - Qrad_obdh_to_bus) / C_obdh;
    
    % Euler update
    T_bus_cold(i+1)  = T_bus_cold(i)  + dT_bus  * dt_cold;
    T_obdh_cold(i+1) = T_obdh_cold(i) + dT_obdh * dt_cold;
    
    % Save heater history
    Qheater_arr(i) = Qheater;
end

Qheater_arr(end) = Qheater_arr(end-1);

%% 
% 8) Convert to Celsius
%% 
T_bus_hot_C   = T_bus_hot  - 273.15;
T_obdh_hot_C  = T_obdh_hot - 273.15;

T_bus_cold_C  = T_bus_cold  - 273.15;
T_obdh_cold_C = T_obdh_cold - 273.15;

t_hot_hr    = t_hot  / 3600;
t_cold_hr   = t_cold / 3600;

%% 
% 9) Print summary
%% 
fprintf('\n--- HOT CASE SUMMARY (0.046 AU) ---\n');
fprintf('Bus max temperature   : %.2f C\n', max(T_bus_hot_C));
fprintf('OBDH max temperature  : %.2f C\n', max(T_obdh_hot_C));
fprintf('Bus final temperature : %.2f C\n', T_bus_hot_C(end));
fprintf('OBDH final temperature: %.2f C\n', T_obdh_hot_C(end));

fprintf('\n--- COLD CASE SUMMARY ---\n');
fprintf('Bus min temperature   : %.2f C\n', min(T_bus_cold_C));
fprintf('OBDH min temperature  : %.2f C\n', min(T_obdh_cold_C));
fprintf('Bus final temperature : %.2f C\n', T_bus_cold_C(end));
fprintf('OBDH final temperature: %.2f C\n', T_obdh_cold_C(end));
fprintf('Peak heater power     : %.2f W\n', max(Qheater_arr));

fprintf('\n--- OBDH LIMIT CHECK ---\n');
fprintf('Required OBDH range   : %.1f to %.1f C\n', T_obdh_min_C, T_obdh_max_C);
fprintf('Hot-case margin to +50 C  : %.2f C\n', T_obdh_max_C - max(T_obdh_hot_C));
fprintf('Cold-case margin to -20 C : %.2f C\n', min(T_obdh_cold_C) - T_obdh_min_C);

%% 
% 10) Representative distance vectors for final-report plotting
%% 
% These are used ONLY to visualize the separate hot- and cold-case
% simulations against heliocentric distance on a logarithmic axis.

% Hot case: representative approach from 1 AU to perihelion
r_hot_plot = logspace(log10(1.0), log10(r_peri_AU), length(t_hot));

% Cold case: representative outbound cruise from 1 AU to 900 AU
r_cold_plot = logspace(log10(1.0), log10(900.0), length(t_cold));

%% 
% 11) Plot 1: Temperature vs Time (logarithmic time axis)
%% 
figure;
semilogx(t_hot_hr,  T_bus_hot_C,   'LineWidth', 2.0); hold on;
semilogx(t_hot_hr,  T_obdh_hot_C,  '--', 'LineWidth', 2.0);
semilogx(t_cold_hr, T_bus_cold_C,  'LineWidth', 2.0);
semilogx(t_cold_hr, T_obdh_cold_C, '--', 'LineWidth', 2.0);

grid on;
xlabel('Time [hr]');
ylabel('Temperature [^{\circ}C]');
title('Bounding-Case Thermal Response vs Time');
legend('Bus - Hot Case', 'OBDH - Hot Case', ...
       'Bus - Cold Case', 'OBDH - Cold Case', ...
       'Location', 'best');

%% 
% 12) Plot 2: Temperature vs Distance (logarithmic distance axis)
%% 
figure;
semilogx(r_hot_plot,  T_bus_hot_C,   'LineWidth', 2.0); hold on;
semilogx(r_hot_plot,  T_obdh_hot_C,  '--', 'LineWidth', 2.0);
semilogx(r_cold_plot, T_bus_cold_C,  'LineWidth', 2.0);
semilogx(r_cold_plot, T_obdh_cold_C, '--', 'LineWidth', 2.0);

grid on;
xlabel('Heliocentric Distance [AU]');
ylabel('Temperature [^{\circ}C]');
title('Bounding-Case Thermal Response vs Heliocentric Distance');
legend('Bus - Hot Case', 'OBDH - Hot Case', ...
       'Bus - Cold Case', 'OBDH - Cold Case', ...
       'Location', 'best');