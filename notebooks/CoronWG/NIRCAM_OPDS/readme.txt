Description of files:

time_vector.fits
- Single extension fits file
- 1D array of time values (in minutes), of length nt

FRILLCO_***.fits
- Multi-extension fits file (3 extensions indexed 0-2)
- 0th fits extension is "Best case" scenario, 1st extension is "Nominal" scenario, 2nd extension is "Requirement" scenario
- Each extension contains a 3D data cube (dimensions = [100, 100, nt]) of Frill/Closeout dynamic WFEs (in m). First 2 dimensions are spatial coordinates, 3rd dimension is time coordinate. Initial WFE is zero, then asymptotically increases.

IEC_***.fits
- Multi-extension fits file (3 extensions indexed 0-2)
- 0th extension is "Best case" scenario, 1st extension is "Nominal" scenario, 2nd extension is "Requirement" scenario
- Each extension contains a 3D data cube (dimensions = [100, 100, nt]) of IEC dynamic WFEs (in m). First 2 dimensions are spatial coordinates, 3rd dimension is time coordinate. Varies as a cosine function, so 0th time index represents amplitude.

TD_***.fits
- Multi-extension fits file (3 extensions indexed 0-2)
- 0th extension is "Best case" scenario, 1st extension is "Nominal" scenario, 2nd extension is "Requirement" scenario
- Each extension contains a 3D data cube (dimensions = [100, 100, nt]) of Thermal Relaxation dynamic WFEs (in m). First 2 dimensions are spatial coordinates, 3rd dimension is time coordinate.  Initial WFE is zero, then asymptotically increases.

LOS_JITTER.fits
- Multi-extension fits file (3 extensions indexed 0-2)
- 0th extension is "Best case" scenario, 1st extension is "Nominal" scenario, 2nd extension is "Requirement" scenario
- Each extension contains a 2D array (dimensions = [1000, 2]) of (x,y) coordinates to sample for LOS jitter (in mas)

TARGET_ACQ.fits
- Multi-extension fits file (3 extensions indexed 0-2)
- 0th extension is "Best case" scenario, 1st extension is "Nominal" scenario, 2nd extension is "Requirement" scenario
- Each extension contains a 1D array of an (x,y) coordinate set to sample for LOS jitter (in mas)

STATIC_***_INPUT.fits
- Single extension fits file
- 2D array (dimensions = [1024,1024]) of the OTE exit pupil WFE (in m) measured at the instrument's central field point

STATIC_***_FULL_SYSTEM  
- *** Likely not used, provided for reference ***
- Single extension fits file
- 3D array (dimensions = [1024,1024,n]) of the full system WFE (in m) measured at several coronagraphic field points.


Notes:
 - slew values are 1.5 deg, 6.0 deg, and 50 deg for the best, nominal, and requirement scenarios
 - correspond to pitch values of ~1 deg, ~4 deg, and 50 deg, which is what is used to estimate the change in thermal state and the magnitudes of the frill/CO and thermal relaxation dynamic terms.
 - Simplified coronagraphic sequence 
    - initial slew, followed by a 1 hour science exposure
    - slew to the reference star, and then another 1 hour exposure of an identical star
 - We will start the simulation after the initial slew completes. The science exposure will last for 1 hour. There will then be a slew to the reference star. The magnitude and time of this slew will depend on the scenario being investigated. However, for simplification, I am currently assuming that the Frill/CO and thermal relaxation dynamic modes "kick in" once the slew is finished. This means that we don't really care about how long the slew to the reference star takes.