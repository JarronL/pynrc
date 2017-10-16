Planned Updates
---------------

FoV aware positions
+++++++++++++++++++
    - Background roll off at grism edges
    - Filter location relative offsets
    - SIAF stuff
    - Subarray positions
    - SCA Gaps and Module gaps

Detector updates in ngNRC.py
++++++++++++++++++++++++++++
    - Pixel non-linearity
    - Intrapixel Capacitance (IPC)
    - Post-pixel Coupling (PPC) due to ADC "smearing"
    - Persistence/latent image
    - Optical distortions
    - QE variations across a pixel's surface
    - RTN Noise
    - Flat field variations

PSF Related
+++++++++++
    - Actual coronagraphic throughput from FITS files
    - Include dispersion of coronagraph PSF
    - WebbPSF needs to center PSF in pixel center
    - PSF Jitter options
    - Bar Mask offsets (using nrc.options['bar_offset'])
    - PSF updates (convolution) based on geometric spot size

Miscellaneous
+++++++++++++
    - Option to exclude flat field noise
    - Option to exclude cosmic ray noise
    - Ramp optimizer warning for large number of group loops?
    - multi-thread ramp optimizer?
    - Random cosmic ray hits in exposure simulator
