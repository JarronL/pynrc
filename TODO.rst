Planned Updates
---------------

FoV aware positions
+++++++++++++++++++
    - Correct coronagraph field locations depending on Lyot optical wedge
    - Background roll off at grism edges
    - Filter location relative offsets
    - SIAF info
    - Subarray positions
    - SCA Gaps and Module gaps

Detector updates in ngNRC.py
++++++++++++++++++++++++++++
    - Pixel non-linearity
    - Intrapixel Capacitance (IPC)
    - Post-pixel Coupling (PPC) due to ADC "smearing"
    - Pixel glow based on subarray size
    - Charge diffusion (esp for saturated pixels)
    - Persistence/latent image
    - Optical distortions
    - QE variations across a pixel's surface
    - RTN Noise
    - Flat field variations

PSF Related
+++++++++++
    - Actual coronagraphic throughput from FITS files
    - More PSF Jitter options
    - PSF convolution based on geometric spot size
    
Observation Classes
+++++++++++++++++++
    - Grism time series
    - Photometric time series (incl. weak lens)
    - Wide-field grism
    - Wide field imaging (esp. SW modules)

Miscellaneous
+++++++++++++
    - Random cosmic ray hits in exposure simulator
    - Ramp optimizer warning for large number of group loops?
    - multi-thread ramp optimizer?
    - DHS mode

