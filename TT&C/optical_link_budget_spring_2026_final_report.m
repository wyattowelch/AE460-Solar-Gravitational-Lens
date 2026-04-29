%% SGL Deep-Space Optical Downlink Budget
%
% Purpose:
%   This script evaluates a photon-counting, PPM-based optical downlink
%   for a Solar Gravitational Lens (SGL) spacecraft at 950 AU.
%
% Method:
%   The analysis uses a physically motivated optical model based on:
%     1) diffraction-limited beam divergence,
%     2) geometric interception by the receive aperture,
%     3) pointing-jitter coupling loss,
%     4) atmospheric and internal optical throughput losses,
%     5) detector quantum efficiency,
%     6) background and dark-count statistics,
%     7) PPM symbol/slot timing and coded throughput,
%     8) detected photons per information bit as the primary link metric.
%
% Notes:
%   This is intentionally not an RF-style kTB/G/T link budget.
%   Equivalent gain/loss terms are reported only for comparison to older
%   presentation styles and legacy communications budget tables.

clc; clear;

%% ========================================================================
%% 1. CONSTANTS
%% ========================================================================
c  = 299792458;                 % Speed of light [m/s]
h  = 6.62607015e-34;            % Planck constant [J*s]
AU = 1.495978707e11;            % Astronomical Unit [m]

%% ========================================================================
%% 2. MISSION GEOMETRY
%% ========================================================================
distance_AU = 950;              % Worst-case Earth-spacecraft range [AU]
distance_m  = distance_AU * AU; % Range [m]

%% ========================================================================
%% 3. OPTICAL CARRIER PARAMETERS
%% ========================================================================
lambda = 1.064e-6;              % Downlink wavelength [m]
f_opt  = c / lambda;            % Optical carrier frequency [Hz]
E_photon = h * f_opt;           % Single-photon energy [J]

%% ========================================================================
%% 4. TRANSMITTER PARAMETERS
%% ========================================================================
Pt_W = 15.0;                    % Average transmitted optical power [W]
Pt_dBW = 10*log10(Pt_W);        % [dBW]

D_tx = 1.0;                     % Spacecraft transmit aperture [m]
eta_tx_opt = 0.70;              % Tx optical throughput efficiency [-]

%% ========================================================================
%% 5. RECEIVER PARAMETERS
%% ========================================================================
D_rx = 30.0;                    % Ground receive aperture diameter [m]
A_rx = pi * (D_rx/2)^2;         % Receive aperture area [m^2]
eta_rx_opt = 0.70;              % Rx optical throughput efficiency [-]
eta_detector = 0.80;            % Detector quantum efficiency [-]

%% ========================================================================
%% 6. ATMOSPHERIC / INTERNAL OPTICAL LOSSES
%% ========================================================================
L_optics_dB = 2.0;              % Internal optics/filter/coupling losses [dB]
L_atmosphere_dB = 2.5;          % Atmosphere transmission loss [dB]

eta_misc_optics = 10^(-L_optics_dB/10);
eta_atmosphere  = 10^(-L_atmosphere_dB/10);

%% ========================================================================
%% 7. POINTING MODEL
%% ========================================================================
% Effective RMS pointing error representing the residual beam-pointing
% uncertainty after spacecraft pointing control and terminal fine tracking.
sigma_point_rad = 0.30e-6;      % [rad]

%% ========================================================================
%% 8. MODULATION AND CODING ASSUMPTIONS
%% ========================================================================
R_b = 2640;                     % Information bit rate [bit/s]
M_ppm = 16;                     % PPM order [-]
code_rate = 0.50;               % FEC code rate [-]

bits_per_symbol_coded = log2(M_ppm);
bits_per_symbol_info  = code_rate * bits_per_symbol_coded;
R_sym  = R_b / bits_per_symbol_info;   % Symbol rate [symbols/s]
R_slot = M_ppm * R_sym;                % Slot rate [slots/s]
T_slot = 1 / R_slot;                   % Slot duration [s]

%% ========================================================================
%% 9. BACKGROUND / DARK COUNT MODEL
%% ========================================================================
% Background optical spectral irradiance after sky brightness, halo leakage,
% and scene filtering assumptions.
%
% This is a placeholder engineering input and will eventually be replaced
% by a site-specific sky radiance / solar halo / filter-bandwidth analysis.
background_irradiance_W_m2_nm = 1e-15; % [W/m^2/nm]
filter_bandwidth_nm = 0.10;            % Effective optical filter width [nm]

% Background power incident on aperture
P_bkg_in_W = background_irradiance_W_m2_nm * filter_bandwidth_nm * A_rx;

% Background power at detector input after receive-side losses
P_bkg_det_in_W = P_bkg_in_W * eta_rx_opt * eta_misc_optics * eta_atmosphere;

% Detected background count rate
N_bkg_det = (P_bkg_det_in_W / E_photon) * eta_detector;   % [photons/s]

% Detector dark counts
N_dark = 100;                   % [counts/s]

%% ========================================================================
%% 10. DIFFRACTION-LIMITED TRANSMIT BEAM
%% ========================================================================
% Airy-pattern first-null half-angle:
theta_div_rad = 1.22 * lambda / D_tx;   % [rad]

% Far-field beam footprint approximation:
beam_radius_m = theta_div_rad * distance_m;  % [m]
beam_diameter_m = 2 * beam_radius_m;         % [m]
A_beam = pi * beam_radius_m^2;               % [m^2]

%% ========================================================================
%% 11. GEOMETRIC INTERCEPTION
%% ========================================================================
% First-order intercepted beam fraction:
eta_geo = A_rx / A_beam;

%% ========================================================================
%% 12. POINTING LOSS
%% ========================================================================
% Gaussian-like beam coupling penalty:
eta_point = exp(-2 * (sigma_point_rad / theta_div_rad)^2);
L_pointing_dB = -10 * log10(eta_point);

%% ========================================================================
%% 13. RECEIVED SIGNAL POWER AND DETECTED PHOTON RATE
%% ========================================================================
eta_total_signal = eta_tx_opt * eta_geo * eta_point * ...
                   eta_atmosphere * eta_rx_opt * eta_misc_optics;

Pr_W = Pt_W * eta_total_signal;       % Received optical power at detector input [W]
Pr_dBW = 10 * log10(Pr_W);            % [dBW]

N_sig_det = (Pr_W / E_photon) * eta_detector;   % Detected signal photon rate [photons/s]

%% ========================================================================
%% 14. SLOT-LEVEL PHOTON STATISTICS
%% ========================================================================
n_sig_slot  = N_sig_det * T_slot;     % Signal photons/slot
n_bkg_slot  = N_bkg_det * T_slot;     % Background photons/slot
n_dark_slot = N_dark    * T_slot;     % Dark counts/slot
n_noise_slot = n_bkg_slot + n_dark_slot;

%% ========================================================================
%% 15. BIT- AND SYMBOL-LEVEL PHOTON EFFICIENCY
%% ========================================================================
photons_per_symbol     = N_sig_det / R_sym;
photons_per_info_bit   = N_sig_det / R_b;
photons_per_coded_bit  = N_sig_det / (R_b / code_rate);

%% ========================================================================
%% 16. REQUIRED PERFORMANCE THRESHOLD
%% ========================================================================
% This value will eventually be replaced with a mission-specific threshold
% derived from selected modulation/coding performance curves.
required_photons_per_info_bit = 10.0;

LinkMargin_dB = 10 * log10(photons_per_info_bit / required_photons_per_info_bit);

%% ========================================================================
%% 17. EQUIVALENT REFERENCE TERMS (FOR COMPARISON ONLY)
%% ========================================================================
Gt_dBi_equiv = 10 * log10((pi * D_tx / lambda)^2 * eta_tx_opt);
Gr_dBi_equiv = 10 * log10((pi * D_rx / lambda)^2 * eta_rx_opt);
L_fs_equiv   = 20 * log10(4 * pi * distance_m / lambda);
EIRP_dBW_equiv = Pt_dBW + Gt_dBi_equiv;

%% ========================================================================
%% 18. REPORT HEADER
%% ========================================================================
fprintf('\n');
fprintf('====================================================================\n');
fprintf('   SGL DEEP-SPACE OPTICAL DOWNLINK BUDGET REPORT\n');
fprintf('   Photon-Counting / Pulse-Position Modulation (PPM) Formulation\n');
fprintf('====================================================================\n');

fprintf('\nMission context:\n');
fprintf(['This analysis evaluates a worst-case 950 AU optical downlink using a ', ...
         'photon-counting, PPM-based receiver architecture.\n']);
fprintf(['The calculation is intended for deep-space optical communications ', ...
         'where detected signal photons, background photons,\n']);
fprintf(['pointing stability, and optical throughput are the dominant ', ...
         'performance drivers.\n']);

%% ========================================================================
%% 19. ASSUMPTIONS SUMMARY
%% ========================================================================
fprintf('\n--------------------------------------------------------------------\n');
fprintf('1. ASSUMPTIONS SUMMARY\n');
fprintf('--------------------------------------------------------------------\n');
fprintf('Range                                 : %8.1f AU\n', distance_AU);
fprintf('Optical wavelength                    : %8.1f nm\n', lambda*1e9);
fprintf('Transmit optical power                : %8.3f W\n', Pt_W);
fprintf('Transmit aperture diameter            : %8.3f m\n', D_tx);
fprintf('Receive aperture diameter             : %8.3f m\n', D_rx);
fprintf('Transmit optical efficiency           : %8.3f\n', eta_tx_opt);
fprintf('Receive optical efficiency            : %8.3f\n', eta_rx_opt);
fprintf('Detector quantum efficiency           : %8.3f\n', eta_detector);
fprintf('Miscellaneous optics loss             : %8.3f dB\n', L_optics_dB);
fprintf('Atmospheric transmission loss         : %8.3f dB\n', L_atmosphere_dB);
fprintf('Effective RMS pointing error          : %8.3f urad\n', sigma_point_rad*1e6);
fprintf('Information bit rate                  : %8.1f bit/s\n', R_b);
fprintf('PPM order                             : %8d\n', M_ppm);
fprintf('Code rate                             : %8.3f\n', code_rate);
fprintf('Background irradiance                 : %8.3e W/m^2/nm\n', background_irradiance_W_m2_nm);
fprintf('Optical filter bandwidth              : %8.3f nm\n', filter_bandwidth_nm);
fprintf('Dark count rate                       : %8.3f counts/s\n', N_dark);
fprintf('Required photons per information bit  : %8.3f\n', required_photons_per_info_bit);

%% ========================================================================
%% 20. ANALYTICAL METHOD SUMMARY
%% ========================================================================
fprintf('\n--------------------------------------------------------------------\n');
fprintf('2. ANALYTICAL METHOD SUMMARY\n');
fprintf('--------------------------------------------------------------------\n');
fprintf('The link budget is evaluated using the following sequence:\n');
fprintf('  (a) Compute photon energy from wavelength.\n');
fprintf('  (b) Compute diffraction-limited beam divergence from transmitter size.\n');
fprintf('  (c) Compute beam diameter at Earth at 950 AU.\n');
fprintf('  (d) Estimate receive-aperture interception of the beam footprint.\n');
fprintf('  (e) Apply pointing-jitter loss as a beam-coupling penalty.\n');
fprintf('  (f) Apply atmospheric and internal optical throughput losses.\n');
fprintf('  (g) Convert received optical power to detected signal photon rate.\n');
fprintf('  (h) Estimate detected background and dark-count rates.\n');
fprintf('  (i) Convert detected signal rate into photons per slot, symbol, and\n');
fprintf('      information bit for the selected PPM / coding architecture.\n');
fprintf('  (j) Compare actual photons per information bit to the required\n');
fprintf('      threshold to obtain photon-efficiency link margin.\n');

%% ========================================================================
%% 21. EQUATION SUMMARY
%% ========================================================================
fprintf('\n--------------------------------------------------------------------\n');
fprintf('3. EQUATION SUMMARY\n');
fprintf('--------------------------------------------------------------------\n');
fprintf('Photon energy:\n');
fprintf('  E_photon = h*c/lambda\n\n');

fprintf('Diffraction-limited divergence:\n');
fprintf('  theta_div = 1.22*lambda/D_tx\n\n');

fprintf('Far-field beam radius at Earth:\n');
fprintf('  r_beam = theta_div * range\n\n');

fprintf('Geometric interception efficiency:\n');
fprintf('  eta_geo = A_rx / A_beam\n\n');

fprintf('Pointing efficiency:\n');
fprintf('  eta_point = exp[-2*(sigma_point/theta_div)^2]\n\n');

fprintf('Received optical power:\n');
fprintf('  P_r = P_t * eta_tx * eta_geo * eta_point * eta_atm * eta_rx * eta_misc\n\n');

fprintf('Detected signal photon rate:\n');
fprintf('  N_sig = (P_r / E_photon) * eta_detector\n\n');

fprintf('PPM timing:\n');
fprintf('  bits/symbol(coded) = log2(M)\n');
fprintf('  bits/symbol(info)  = code_rate*log2(M)\n');
fprintf('  R_sym  = R_b / bits_per_symbol(info)\n');
fprintf('  R_slot = M * R_sym\n');
fprintf('  T_slot = 1 / R_slot\n\n');

fprintf('Slot-level statistics:\n');
fprintf('  n_sig_slot  = N_sig * T_slot\n');
fprintf('  n_bkg_slot  = N_bkg * T_slot\n');
fprintf('  n_dark_slot = N_dark * T_slot\n\n');

fprintf('Photon-efficiency margin:\n');
fprintf('  Margin = 10*log10[(photons/info bit)/(required photons/info bit)]\n');

%% ========================================================================
%% 22. RESULTS TABLE
%% ========================================================================
ReportTable = table( ...
    distance_AU, lambda*1e9, Pt_W, Pt_dBW, ...
    D_tx, D_rx, ...
    theta_div_rad*1e6, beam_diameter_m/1e6, ...
    eta_geo, eta_point, L_pointing_dB, ...
    L_optics_dB, L_atmosphere_dB, ...
    Pr_dBW, ...
    N_sig_det, N_bkg_det, N_dark, ...
    M_ppm, code_rate, R_b, R_sym, R_slot, T_slot, ...
    n_sig_slot, n_bkg_slot, n_dark_slot, ...
    photons_per_symbol, photons_per_info_bit, required_photons_per_info_bit, LinkMargin_dB, ...
    Gt_dBi_equiv, Gr_dBi_equiv, L_fs_equiv, EIRP_dBW_equiv, ...
    'VariableNames', { ...
    'Distance_AU', ...
    'Wavelength_nm', ...
    'TxPower_W', ...
    'TxPower_dBW', ...
    'TxAperture_m', ...
    'RxAperture_m', ...
    'BeamDivergence_urad', ...
    'BeamDiameterAtEarth_Mm', ...
    'GeometricCaptureEff', ...
    'PointingEfficiency', ...
    'PointingLoss_dB', ...
    'MiscOpticsLoss_dB', ...
    'AtmosphereLoss_dB', ...
    'ReceivedOpticalPower_dBW', ...
    'DetectedSignalPhotonsPerSec', ...
    'DetectedBackgroundPhotonsPerSec', ...
    'DarkCountsPerSec', ...
    'PPM_Order', ...
    'CodeRate', ...
    'InfoBitRate_bps', ...
    'SymbolRate_sym_s', ...
    'SlotRate_slot_s', ...
    'SlotDuration_s', ...
    'SignalPhotonsPerSlot', ...
    'BackgroundPhotonsPerSlot', ...
    'DarkCountsPerSlot', ...
    'SignalPhotonsPerSymbol', ...
    'SignalPhotonsPerInfoBit', ...
    'RequiredPhotonsPerInfoBit', ...
    'LinkMargin_dB', ...
    'TxGain_dBi_Equiv', ...
    'RxGain_dBi_Equiv', ...
    'FreeSpaceLoss_dB_Equiv', ...
    'EIRP_dBW_Equiv' ...
    });

fprintf('\n--------------------------------------------------------------------\n');
fprintf('4. NUMERICAL RESULTS\n');
fprintf('--------------------------------------------------------------------\n');
disp(ReportTable);

%% ========================================================================
%% 23. ENGINEERING INTERPRETATION
%% ========================================================================
fprintf('\n--------------------------------------------------------------------\n');
fprintf('5. ENGINEERING INTERPRETATION\n');
fprintf('--------------------------------------------------------------------\n');

fprintf('Beam divergence                     : %8.3f urad\n', theta_div_rad*1e6);
fprintf('Beam diameter at Earth              : %8.3f Mm\n', beam_diameter_m/1e6);
fprintf('Geometric capture efficiency        : %8.3e\n', eta_geo);
fprintf('Pointing efficiency                 : %8.6f\n', eta_point);
fprintf('Received optical power              : %8.3e W\n', Pr_W);
fprintf('Detected signal photon rate         : %8.3e photons/s\n', N_sig_det);
fprintf('Detected background photon rate     : %8.3e photons/s\n', N_bkg_det);
fprintf('Dark count rate                     : %8.3e counts/s\n', N_dark);
fprintf('Signal photons per slot             : %8.6f\n', n_sig_slot);
fprintf('Background photons per slot         : %8.6f\n', n_bkg_slot);
fprintf('Dark counts per slot                : %8.6f\n', n_dark_slot);
fprintf('Signal photons per information bit  : %8.3f\n', photons_per_info_bit);
fprintf('Required photons per information bit: %8.3f\n', required_photons_per_info_bit);
fprintf('Photon-efficiency link margin       : %8.3f dB\n', LinkMargin_dB);

fprintf('\nInterpretive remarks:\n');

if LinkMargin_dB > 3
    fprintf(['  The computed photon-efficiency link margin is positive and exceeds ', ...
             'a typical preliminary-design comfort threshold.\n']);
elseif LinkMargin_dB > 0
    fprintf(['  The computed photon-efficiency link margin is positive but modest; ', ...
             'the design would benefit from additional margin studies.\n']);
else
    fprintf(['  The computed photon-efficiency link margin is negative; the design ', ...
             'as currently parameterized is not supportable.\n']);
end

if n_sig_slot < 0.1
    fprintf(['  The detected signal photons per slot are very low, indicating a ', ...
             'strongly photon-starved regime and high sensitivity to\n']);
    fprintf(['  background counts, pointing excursions, and coding assumptions.\n']);
elseif n_sig_slot < 1
    fprintf(['  The detected signal photons per slot are below one, consistent with ', ...
             'a photon-limited deep-space optical channel.\n']);
else
    fprintf(['  The detected signal photons per slot exceed unity, which is more ', ...
             'favorable for symbol discrimination but still requires\n']);
    fprintf(['  careful control of background counts and timing performance.\n']);
end

if n_bkg_slot > 0.1*n_sig_slot
    fprintf(['  Background counts are non-negligible relative to the signal and ', ...
             'should be refined using site-specific sky/halo assumptions.\n']);
else
    fprintf(['  Background counts are small relative to the signal under the current ', ...
             'assumptions, but this depends strongly on site and filter design.\n']);
end

fprintf(['  The most important parameters for design sensitivity remain: ', ...
         'pointing stability, atmospheric transmission, receive-aperture\n']);
fprintf(['  diameter, optical throughput, detector efficiency, and the selected ', ...
         'PPM/FEC operating point.\n']);

%% ========================================================================
%% 24. TABLE VALUE DESCRIPTIONS
%% ========================================================================
fprintf('\n--------------------------------------------------------------------\n');
fprintf('6. TABLE VALUE DESCRIPTIONS\n');
fprintf('--------------------------------------------------------------------\n');
fprintf('Distance_AU                  : Earth-spacecraft range used in the analysis.\n');
fprintf('Wavelength_nm                : Optical downlink wavelength.\n');
fprintf('TxPower_W / TxPower_dBW      : Average transmitted optical power.\n');
fprintf('TxAperture_m                 : Spacecraft optical transmit aperture diameter.\n');
fprintf('RxAperture_m                 : Ground receive telescope diameter.\n');
fprintf('BeamDivergence_urad          : Diffraction-limited beam half-angle.\n');
fprintf('BeamDiameterAtEarth_Mm       : Beam footprint diameter at Earth.\n');
fprintf('GeometricCaptureEff          : Fraction of beam power intercepted by receiver.\n');
fprintf('PointingEfficiency           : Fraction of signal retained after pointing jitter.\n');
fprintf('PointingLoss_dB              : Pointing penalty expressed in dB.\n');
fprintf('MiscOpticsLoss_dB            : Internal optical/filter/coupling loss.\n');
fprintf('AtmosphereLoss_dB            : Atmospheric transmission loss at the receiver.\n');
fprintf('ReceivedOpticalPower_dBW     : Signal power arriving at detector input.\n');
fprintf('DetectedSignalPhotonsPerSec  : Usable signal count rate after detector QE.\n');
fprintf('DetectedBackgroundPhotonsPerSec : Background count rate after filtering and QE.\n');
fprintf('DarkCountsPerSec             : Detector false-count rate.\n');
fprintf('PPM_Order                    : Modulation order used for pulse-position modulation.\n');
fprintf('CodeRate                     : FEC code rate linking coded and information throughput.\n');
fprintf('InfoBitRate_bps              : Information-bearing downlink rate.\n');
fprintf('SymbolRate_sym_s             : PPM symbol rate required to support the bit rate.\n');
fprintf('SlotRate_slot_s              : Total slot decision rate.\n');
fprintf('SlotDuration_s               : Duration of an individual PPM slot.\n');
fprintf('SignalPhotonsPerSlot         : Signal counts available in one slot.\n');
fprintf('BackgroundPhotonsPerSlot     : Background counts accumulated in one slot.\n');
fprintf('DarkCountsPerSlot            : Dark counts accumulated in one slot.\n');
fprintf('SignalPhotonsPerSymbol       : Signal counts per PPM symbol.\n');
fprintf('SignalPhotonsPerInfoBit      : Primary photon-efficiency figure of merit.\n');
fprintf('RequiredPhotonsPerInfoBit    : Assumed threshold for successful communications.\n');
fprintf('LinkMargin_dB                : Margin between achieved and required photon efficiency.\n');
fprintf('TxGain_dBi_Equiv             : Equivalent transmit gain for comparison only.\n');
fprintf('RxGain_dBi_Equiv             : Equivalent receive gain for comparison only.\n');
fprintf('FreeSpaceLoss_dB_Equiv       : Equivalent free-space path loss for comparison only.\n');
fprintf('EIRP_dBW_Equiv               : Equivalent EIRP for comparison only.\n');

fprintf('\n====================================================================\n');
fprintf('End of report.\n');
fprintf('====================================================================\n\n');