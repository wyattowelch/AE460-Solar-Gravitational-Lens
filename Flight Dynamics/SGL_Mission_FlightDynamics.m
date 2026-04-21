%% SGL Mission Flight Dynamics Analysis
% AE 460 Spacecraft Design
% Author: Parham Khodadi
% December 2024
%
% Trajectory: GEO -> Jupiter GA -> Solar Oberth (0.046 AU) -> 650 AU
% Target: Antipode of Proxima Centauri b (RA=37.43°, Dec=+62.68°)

clear; clc; close all;

%% Constants

% Gravitational parameters [km^3/s^2]
mu_Sun = 1.32712440018e11;
mu_Earth = 3.986004418e5;
mu_Jupiter = 1.26686534e8;

% Physical constants
AU = 1.495978707e8;           % km
R_Sun = 696340;               % km
R_Jupiter = 71492;            % km

% Planetary orbits (circular coplanar approximation)
r_Earth = 1.0 * AU;
r_Jupiter = 5.2 * AU;
v_Earth = sqrt(mu_Sun/r_Earth);
v_Jupiter = sqrt(mu_Sun/r_Jupiter);

%% Mission Parameters

r_perihelion = 0.046 * AU;    % Solar perihelion distance
r_target = 650 * AU;          % Target distance (SGL focal region)
RA_target = 37.4289;          % Target right ascension [deg]
Dec_target = 62.6795;         % Target declination [deg]

r_GEO = 42164;                % GEO radius [km]
v_GEO = sqrt(mu_Earth/r_GEO); % GEO velocity [km/s]

%% Phase 1: Earth Departure (Trans-Jupiter Injection)

% Hohmann transfer to Jupiter
a_transfer_EJ = (r_Earth + r_Jupiter) / 2;
v_departure_helio = sqrt(mu_Sun * (2/r_Earth - 1/a_transfer_EJ));
v_inf_Earth = v_departure_helio - v_Earth;
C3_required = v_inf_Earth^2;

% Delta-V from GEO
v_escape_needed = sqrt(v_inf_Earth^2 + 2*mu_Earth/r_GEO);
DV1_TMI = v_escape_needed - v_GEO;

%% Phase 2: Earth-Jupiter Transfer

T_transfer_EJ = pi * sqrt(a_transfer_EJ^3 / mu_Sun);
T_transfer_EJ_years = T_transfer_EJ / (365.25 * 24 * 3600);

v_arrival_helio = sqrt(mu_Sun * (2/r_Jupiter - 1/a_transfer_EJ));
v_inf_Jupiter = v_Jupiter - v_arrival_helio;

%% Phase 3: Jupiter Gravity Assist (Powered Flyby)

% Post-flyby orbit parameters (Sun-diving ellipse)
a_dive = (r_Jupiter + r_perihelion) / 2;
e_dive = (r_Jupiter - r_perihelion) / (r_Jupiter + r_perihelion);
v_post_Jupiter = sqrt(mu_Sun * (2/r_Jupiter - 1/a_dive));

% Flyby geometry
h_periapsis_J = 1.0 * R_Jupiter;  % Flyby altitude = 1 Jupiter radius
r_periapsis_J = R_Jupiter + h_periapsis_J;

v_inf_J_mag = abs(v_inf_Jupiter);
v_periapsis_J = sqrt(v_inf_J_mag^2 + 2*mu_Jupiter/r_periapsis_J);

v_inf_out_needed = v_Jupiter - v_post_Jupiter;
v_periapsis_out = sqrt(v_inf_out_needed^2 + 2*mu_Jupiter/r_periapsis_J);
DV2_Jupiter = abs(v_periapsis_out - v_periapsis_J);

% Turn angle (for reference)
delta_turn = 2 * asin(1 / (1 + r_periapsis_J * v_inf_J_mag^2 / mu_Jupiter));

%% Phase 4: Coast to Solar Perihelion

T_dive = pi * sqrt(a_dive^3 / mu_Sun);
T_dive_years = T_dive / (365.25 * 24 * 3600);

v_perihelion = sqrt(mu_Sun * (2/r_perihelion - 1/a_dive));
v_escape_Sun = sqrt(2 * mu_Sun / r_perihelion);

%% Phase 5: Solar Oberth Maneuver

target_mission_time = 25;  % Desired time to reach 650 AU [years]
avg_velocity_needed = (r_target/AU) / target_mission_time;  % [AU/year]
v_inf_target = avg_velocity_needed * AU / (365.25 * 24 * 3600);  % [km/s]

v_required_perihelion = sqrt(v_inf_target^2 + 2*mu_Sun/r_perihelion);
DV3_Solar = v_required_perihelion - v_perihelion;

% Oberth benefit: delta-V saved compared to burn at infinity
DV_savings = v_inf_target - DV3_Solar;

%% Phase 6: Hyperbolic Escape to 650 AU

a_escape = -mu_Sun / v_inf_target^2;
e_escape = 1 - r_perihelion / abs(a_escape);

% Time to reach 650 AU (hyperbolic trajectory)
r_final = r_target;
cosh_H = (1 - r_final/a_escape) / e_escape;
H = acosh(cosh_H);
t_650AU = sqrt((-a_escape)^3/mu_Sun) * (e_escape * sinh(H) - H);
t_650AU_years = t_650AU / (365.25 * 24 * 3600);

v_at_650AU = sqrt(v_inf_target^2 + 2*mu_Sun/r_target);

%% Summary Calculations

DV_total = DV1_TMI + DV2_Jupiter + DV3_Solar;
T_total = T_transfer_EJ_years + T_dive_years + t_650AU_years;

%% Display Results

fprintf('\n');
fprintf('===== SGL MISSION FLIGHT DYNAMICS RESULTS =====\n\n');

fprintf('Target Coordinates:\n');
fprintf('  RA  = %.4f deg\n', RA_target);
fprintf('  Dec = %.4f deg\n', Dec_target);
fprintf('  Distance = %d AU\n\n', r_target/AU);

fprintf('Delta-V Budget:\n');
fprintf('  TMI (GEO departure):  %6.2f km/s  (%5.1f%%)\n', DV1_TMI, 100*DV1_TMI/DV_total);
fprintf('  Jupiter Oberth:       %6.2f km/s  (%5.1f%%)\n', DV2_Jupiter, 100*DV2_Jupiter/DV_total);
fprintf('  Solar Oberth:         %6.2f km/s  (%5.1f%%)\n', DV3_Solar, 100*DV3_Solar/DV_total);
fprintf('  -----------------------------------------\n');
fprintf('  Total:                %6.2f km/s\n\n', DV_total);

fprintf('Mission Timeline:\n');
fprintf('  Earth to Jupiter:     %5.2f years\n', T_transfer_EJ_years);
fprintf('  Jupiter to Sun:       %5.2f years\n', T_dive_years);
fprintf('  Sun to 650 AU:        %5.2f years\n', t_650AU_years);
fprintf('  -----------------------------------------\n');
fprintf('  Total:                %5.2f years\n\n', T_total);

fprintf('Key Parameters:\n');
fprintf('  C3 at Earth:          %.2f km^2/s^2\n', C3_required);
fprintf('  Jupiter V_inf:        %.2f km/s\n', v_inf_J_mag);
fprintf('  Jupiter flyby alt:    %.0f km (%.1f R_J)\n', h_periapsis_J, h_periapsis_J/R_Jupiter);
fprintf('  Solar perihelion:     %.4f AU (%.1f R_Sun)\n', r_perihelion/AU, r_perihelion/R_Sun);
fprintf('  Perihelion velocity:  %.2f km/s\n', v_perihelion);
fprintf('  Escape V_inf:         %.2f km/s (%.1f AU/yr)\n', v_inf_target, avg_velocity_needed);
fprintf('  Oberth benefit:       %.2f km/s saved\n', DV_savings);
fprintf('\n');