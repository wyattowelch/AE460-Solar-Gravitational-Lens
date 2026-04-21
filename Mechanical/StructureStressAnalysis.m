%% Mechanical Subsystem – 4 Deep C-Channel Longerons, 1356 kg
clear; clc;

%% Material properties (Al 6061-T6)
E       = 68.9e9;      % Young's modulus (Pa)
sigma_y = 276e6;       % Yield strength (Pa)

%% Spacecraft mass and launch load
mass_sc = 1356;       
g_load  = 10;         
F_total = mass_sc * g_load * 9.81;   
F_long  = F_total / 4;               

%% === BASELINE SECTION ===
t  = 0.008;     
h  = 0.060;      
b  = 0.040;     

A = t * (2*b + h);    


I_full = (b * h^3) / 12;
I_hole = ((b - t) * (h - 2*t)^3) / 12;
I = I_full - I_hole;   % m^4

% Axial stress
sigma = F_long / A;    

L_eff = 2.0;           
K = 1.0;              

P_cr = (pi^2 * E * I) / (K * L_eff)^2;  

buckling_SF = P_cr / F_long;
yield_SF    = sigma_y / sigma;

% Global first bending mode (equivalent beam)
D_truss = 1.0;        
R = D_truss / 2;

I_eq = 4 * (I + A * R^2);  
L_truss = 8.0;
m_eq = mass_sc;

omega1 = (1.875^2) * sqrt(E * I_eq / (m_eq * L_truss^4));
f1 = omega1 / (2*pi);      

fprintf("=== Baseline – 60x40x8 mm C-channels, 4 longerons ===\n");
fprintf("Spacecraft mass:               %.1f kg\n", mass_sc);
fprintf("Total axial launch load:       %.1f N\n", F_total);
fprintf("Axial load per longeron:       %.1f N\n", F_long);
fprintf("Axial stress in longeron:      %.2f MPa\n", sigma/1e6);
fprintf("Yield safety factor:           %.2f\n", yield_SF);
fprintf("Buckling load per longeron:    %.1f N\n", P_cr);
fprintf("Buckling safety factor:        %.2f\n", buckling_SF);
fprintf("First bending-mode frequency:  %.2f Hz\n\n", f1);


A_sail = 100;             
p_rad  = 9e-6;             
F_sail = p_rad * A_sail;
F_boom = F_sail / 4;

L_boom = 7.0;
Do_b   = 0.020;
t_b    = 0.0003;
Di_b   = Do_b - 2*t_b;

I_boom = pi/64 * (Do_b^4 - Di_b^4);
P_cr_boom = pi^2 * E * I_boom / L_boom^2;
boom_SF = P_cr_boom / F_boom;

fprintf("=== Solar Sail Boom Check ===\n");
fprintf("Load per boom (radiation):     %.3e N\n", F_boom);
fprintf("Boom buckling capacity:        %.2f N\n", P_cr_boom);
fprintf("Boom buckling safety factor:   %.1e\n\n", boom_SF);

%% === FIGURE 1 & 2: Thickness Sweep  ===
t_vals = linspace(0.004, 0.012, 25);  
sigma_vals       = zeros(size(t_vals));
buckling_SF_vals = zeros(size(t_vals));

for i = 1:length(t_vals)
    t_i = t_vals(i);
    A_i = t_i * (2*b + h);
    I_full_i = (b * h^3) / 12;
    I_hole_i = ((b - t_i) * (h - 2*t_i)^3) / 12;
    I_i = I_full_i - I_hole_i;
    
    sigma_vals(i) = F_long / A_i;
    Pcr_i = pi^2 * E * I_i / (K * L_eff)^2;
    buckling_SF_vals(i) = Pcr_i / F_long;
end

figure;
plot(t_vals*1000, sigma_vals/1e6, 'LineWidth', 1.5);
xlabel('C-channel wall thickness (mm)');
ylabel('Axial stress (MPa)');
title('Longeron Axial Stress vs Wall Thickness');
grid on;

figure;
plot(t_vals*1000, buckling_SF_vals, 'LineWidth', 1.5);
xlabel('C-channel wall thickness (mm)');
ylabel('Buckling Safety Factor');
title('Buckling Safety Factor vs Wall Thickness');
yline(1,'--','SF = 1','LabelHorizontalAlignment','left');
grid on;

%% === FIGURE 3: Depth Sweep ===
h_vals = linspace(0.04, 0.10, 13);  
buckling_SF_h = zeros(size(h_vals));

for j = 1:length(h_vals)
    h_j = h_vals(j);
    I_full_j = (b * h_j^3) / 12;
    I_hole_j = ((b - t) * (h_j - 2*t)^3) / 12;
    I_j = I_full_j - I_hole_j;
    
    Pcr_j = pi^2 * E * I_j / (K * L_eff)^2;
    buckling_SF_h(j) = Pcr_j / F_long;
end

figure;
plot(h_vals*1000, buckling_SF_h, 'LineWidth', 1.5);
xlabel('C-channel depth h (mm)');
ylabel('Buckling Safety Factor');
title('Buckling Safety Factor vs C-Channel Depth (t = 8 mm)');
yline(1,'--','SF = 1','LabelHorizontalAlignment','left');
grid on;
