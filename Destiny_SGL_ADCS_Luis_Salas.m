%% Destiny_SGL_ADCS_Luis_Salas.m
% Destiny: Solar Gravitational Lens (SGL) mission ADCS closed-loop simulation
%
% The script created here implements a MATLAB-style attitude determination and control
% simulation based on our teams PDR architecture:
%   - 4 star trackers (Sodern SED16 model)
%   - 4 reaction wheels in a pyramid configuration (Honeywell HR04 model)
%   - OBC closes the loop using tracker/gyro inputs
%   - mission mode here requires ultra-fine pointing
%
% Our model is intentionally first-order and includes :
%   1) rigid-body spacecraft attitude dynamics
%   2) gyro + star tracker measurement loop
%   3) quaternion PD control in the OBC
%   4) 4-wheel torque allocation
%   5) simple wheel momentum / saturation model
%
% Things to note with this code:
% Our PDR provides architecture, selected components, and requirements,
% BUT not complete inertia matrices, wheel inertia's, or detailed sensor
% noise tables. My values below are engineering assumptions that can be
% refined with our team's detailed design data.

clear; clc; close all;

%% -------------------- Destiny Mission / PDR Inputs ----------------
% Derived values for extra context:
req.pointing_arcsec = 0.01;      % science pointing requirement
req.power_limit_W   = 80;        % spacecraft power requirement
req.dry_mass_kg     = 200;       % dry mass limit
req.science_start_AU = 650;
req.science_end_AU   = 900;

% Final-stage spacecraft parameters for ADCS simulation (Assumptions)
sc.mass_kg = 127;                            % final stage dry mass
sc.J = diag([42, 38, 30]);                  % assumed spacecraft inertia [kg-m^2]
sc.Jinv = inv(sc.J);

% Four-wheel pyramid geometry (Destiny ADCS Reaction Wheels configuration)
beta = deg2rad(54.7356); % tetra/pyramid half-angle
wheelAxes = [ ...
     cos(beta),  0,          sin(beta);
    -cos(beta),  0,          sin(beta);
     0,          cos(beta), -sin(beta);
     0,         -cos(beta), -sin(beta)]';
wheelAxes = wheelAxes ./ vecnorm(wheelAxes);

rw.N = wheelAxes;                % 3x4 wheel axis matrix
rw.Jw = 0.0035;                  % assumed wheel rotor inertia [kg-m^2]
rw.maxTorque = 0.05;             % assumed max torque per wheel [N-m]
rw.maxSpeed_rpm = 6000;          % assumed wheel speed limit [rpm]
rw.maxSpeed = rw.maxSpeed_rpm * 2*pi/60; % [rad/s]
rw.h = zeros(4,1);               % wheel angular momentum state

% Sensor assumptions used in this case
gyro.sigma_rad_s = deg2rad(0.002);         % gyro white noise [rad/s]
gyro.bias0 = deg2rad([0.002; -0.0015; 0.001]); % initial gyro bias [rad/s]
gyro.bias_rw = deg2rad(2e-5);              % bias random walk [rad/s/sqrt(s)]

tracker.dt = 0.5;                           % star tracker update period [s]
tracker.sigma_arcsec = 0.2;                 % assumed measurement noise [arcsec]
tracker.sigma_rad = tracker.sigma_arcsec/206265;

% OBC/controller assumptions used 
obc.dt = 0.05;                              % control task period [s]
ctrl.Kp = diag([3e-4 3e-4 2.5e-4]);         % quaternion vector-error gain
ctrl.Kd = diag([0.25 0.25 0.20]);           % rate damping gain
ctrl.rate_cmd = [0;0;0];                    % science mode zero-rate pointing
ctrl.deadband_arcsec = 0.002;               % optional fine-pointing deadband

% Disturbance assumptions used (these are small deep-space environmental torques)
dist.base = [3e-7; -2e-7; 1e-7];            % constant bias disturbance [N-m]
dist.sin_amp = [2e-7; 1.5e-7; 1e-7];        % periodic disturbance amplitude
dist.sin_w = [0.007; 0.011; 0.005];         % disturbance frequency [rad/s]

%% --------------------- ADCS Simulation Setup -------------------------
tFinal = 600;                       % seconds
dt = obc.dt;
t = 0:dt:tFinal;
N = numel(t);

% Initial attitude error (about 0.15 deg equivalent misalignment)
q = [0.999997; 0.0015; -0.0008; 0.0010];
q = q / norm(q);                    % quaternion scalar-first [q0;q1;q2;q3]
w = deg2rad([0.02; -0.015; 0.01]);  % body rates [rad/s]

% Desired science pointing attitude
qd = [1;0;0;0];

% Estimator states
gyroBias = gyro.bias0;
q_est = q;                          % initialize estimate
lastTrackerUpdate = -inf;

% Logging
q_log = zeros(4,N);
qest_log = zeros(4,N);
w_log = zeros(3,N);
wmeas_log = zeros(3,N);
tau_cmd_log = zeros(3,N);
tau_rw_log = zeros(3,N);
wheel_torque_log = zeros(4,N);
wheel_speed_log = zeros(4,N);
pointing_err_arcsec = zeros(1,N);
tracker_used = false(1,N);

%% ------------------------- ADCS Main Loop ----------------------------
for k = 1:N
    tk = t(k);

    %  Disturbance torque model 
    tau_dist = dist.base + dist.sin_amp .* sin(dist.sin_w * tk);

    %  Gyro measurement 
    gyroBias = gyroBias + gyro.bias_rw * sqrt(dt) * randn(3,1);
    w_meas = w + gyroBias + gyro.sigma_rad_s * randn(3,1);

    %  Star tracker update (has lower rate than control loop)
    if (tk - lastTrackerUpdate) >= tracker.dt - 1e-12
        dq_noise = [1;
                    0.5*tracker.sigma_rad*randn(3,1)];
        dq_noise = dq_noise / norm(dq_noise);
        q_tracker = quatMultiply(q, dq_noise);
        q_est = q_tracker / norm(q_tracker);     % simple fused estimate
        lastTrackerUpdate = tk;
        tracker_used(k) = true;
    else
        % propagate estimated attitude using gyro between tracker updates
        q_est = quatPropagate(q_est, w_meas, dt);
    end

    %  Attitude error in OBC 
    q_err = quatMultiply(quatConj(qd), q_est);
    if q_err(1) < 0
        q_err = -q_err; % shortest rotation
    end

    err_arcsec = 2*norm(q_err(2:4))*206265;
    pointing_err_arcsec(k) = err_arcsec;

    % Deadband for very fine pointing (Star Tracker)
    qv = q_err(2:4);
    if err_arcsec < ctrl.deadband_arcsec
        qv = zeros(3,1);
    end

    % Quaternion PD control law
    tau_cmd = -ctrl.Kp*qv - ctrl.Kd*(w_meas - ctrl.rate_cmd);

    %  4-wheel torque allocation
    wheel_torque = pinv(rw.N) * tau_cmd;
    wheel_torque = max(min(wheel_torque, rw.maxTorque), -rw.maxTorque);

    % Back out actual body torque after saturation
    tau_rw = rw.N * wheel_torque;

    % Reaction Wheel speed / momentum updated 
    rwOmega = rw.h / rw.Jw;
    rwOmega = rwOmega + (wheel_torque / rw.Jw) * dt;

    % Reaction Wheel speed saturation 
    rwOmega = max(min(rwOmega, rw.maxSpeed), -rw.maxSpeed);
    rw.h = rw.Jw * rwOmega;

    %  Destiny Spacecraft rotational dynamics 
    wdot = sc.Jinv * (tau_rw + tau_dist - cross(w, sc.J*w));
    w = w + wdot*dt;
    q = quatPropagate(q, w, dt);

    %  Log 
    q_log(:,k) = q;
    qest_log(:,k) = q_est;
    w_log(:,k) = w;
    wmeas_log(:,k) = w_meas;
    tau_cmd_log(:,k) = tau_cmd;
    tau_rw_log(:,k) = tau_rw;
    wheel_torque_log(:,k) = wheel_torque;
    wheel_speed_log(:,k) = rwOmega;
end

%% ---------------------- ADCS Requirement Check ------------------------
steady_idx = t > 0.8*tFinal;
steady_mean = mean(pointing_err_arcsec(steady_idx));
steady_max  = max(pointing_err_arcsec(steady_idx));
req_met = steady_max <= req.pointing_arcsec;

fprintf('\nSGL ADCS Simulation Summary\n');
fprintf('---------------------------------------------\n');
fprintf('Mission science pointing requirement : %.5f arcsec\n', req.pointing_arcsec);
fprintf('Steady-state mean pointing error     : %.5f arcsec\n', steady_mean);
fprintf('Steady-state max pointing error      : %.5f arcsec\n', steady_max);
fprintf('Requirement met?                     : %s\n', string(req_met));
fprintf('Wheel max speed used                 : %.1f rpm\n', max(abs(wheel_speed_log(:)))*60/(2*pi));
fprintf('Wheel max torque commanded           : %.4f N-m\n', max(abs(wheel_torque_log(:))));
fprintf('Tracker updates used                 : %d samples\n', nnz(tracker_used));

%% -------------------------- ADCS Plots -------------------------------
figure;
plot(t, pointing_err_arcsec, 'LineWidth', 1.4); hold on;
yline(req.pointing_arcsec, '--', 'Requirement');
xlabel('Time [s]');
ylabel('Pointing Error [arcsec]');
title('SGL ADCS Fine Pointing Error');
grid on;

figure;
plot(t, rad2deg(w_log(1,:)), 'LineWidth', 1.2); hold on;
plot(t, rad2deg(w_log(2,:)), 'LineWidth', 1.2);
plot(t, rad2deg(w_log(3,:)), 'LineWidth', 1.2);
xlabel('Time [s]');
ylabel('Body Rates [deg/s]');
title('Spacecraft Angular Rates');
legend('\omega_x','\omega_y','\omega_z','Location','best');
grid on;

figure;
plot(t, wheel_speed_log(1,:)*60/(2*pi), 'LineWidth', 1.1); hold on;
plot(t, wheel_speed_log(2,:)*60/(2*pi), 'LineWidth', 1.1);
plot(t, wheel_speed_log(3,:)*60/(2*pi), 'LineWidth', 1.1);
plot(t, wheel_speed_log(4,:)*60/(2*pi), 'LineWidth', 1.1);
xlabel('Time [s]');
ylabel('Wheel Speed [rpm]');
title('Reaction Wheel Speeds');
legend('RW1','RW2','RW3','RW4','Location','best');
grid on;

figure;
plot(t, wheel_torque_log(1,:), 'LineWidth', 1.1); hold on;
plot(t, wheel_torque_log(2,:), 'LineWidth', 1.1);
plot(t, wheel_torque_log(3,:), 'LineWidth', 1.1);
plot(t, wheel_torque_log(4,:), 'LineWidth', 1.1);
xlabel('Time [s]');
ylabel('Wheel Torque [N-m]');
title('Wheel Torque Commands from OBC');
legend('RW1','RW2','RW3','RW4','Location','best');
grid on;

%% ------------------------ Buses -------------------------------
% Example: rough ADCS power placeholder for future use
% PDR ADCS allocation is ~28 W total. I can later replace this with
% detailed mode-dependent wheel + tracker + OBC power models.
adcsPowerApprox_W = 28;
fprintf('Approximate ADCS power allocation from PDR: %.1f W\n', adcsPowerApprox_W);

%% ---------------------- Helper Functions -------------------------
function qn = quatPropagate(q, w, dt)
    Omega = [0    -w(1) -w(2) -w(3);
             w(1)  0     w(3) -w(2);
             w(2) -w(3)  0     w(1);
             w(3)  w(2) -w(1)  0   ];
    qn = q + 0.5 * Omega * q * dt;
    qn = qn / norm(qn);
end

function qc = quatConj(q)
    qc = [q(1); -q(2:4)];
end

function q = quatMultiply(q1, q2)
    s1 = q1(1); v1 = q1(2:4);
    s2 = q2(1); v2 = q2(2:4);
    q = [s1*s2 - dot(v1,v2);
         s1*v2 + s2*v1 + cross(v1,v2)];
end
