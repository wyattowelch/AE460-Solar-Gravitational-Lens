### Overview

This MATLAB script evaluates a deep-space optical downlink for the SGL mission using a photon-counting, PPM-based link budget. The model computes received optical power, detected photon rates, and link margin based on system design parameters at a worst-case distance of 950 AU.

### Script Structure
The script is organized into clearly labeled sections:
- **Constants**: Physical constants and unit definitions

- **Mission Geometry**: Spacecraft range and configuration

- **Optical Parameters**: Wavelength and photon energy

- **4-6 Transmitter / Receiver / Losses**: System hardware and efficiency assumptions

- **Pointing Model**: RMS pointing error and associated loss

- **Modulation and Coding**: Bit rate, PPM order, and code rate

- **Background and Detector Model**: Background irradiance and dark counts

- **10-13 Optical Propagation**: Beam divergence, footprint, capture efficiency, received power

- **14-16 Photon Calculations**: Signal/background photons, photons per slot, photons per bit, link margin

- **17-24 Report Output**: Summary, results table, and interpretation

### Running the Simulation
1. Open the MATLAB .m file
2. Run the script
3. Review results in the Command Window
No additional toolboxes are required.

### Key Design Parameters
The following variables have the largest impact on link performance:

- **R_b** – Information bit rate
- **Pt_W** – Transmit optical power
- **D_tx** – Transmit aperture diameter
- **D_rx** – Receive aperture diameter
- **sigma_point_rad** – Pointing error
- **L_atmosphere_dB** – Atmospheric loss
- **background_irradiance_W_m2_nm** – Background light level
