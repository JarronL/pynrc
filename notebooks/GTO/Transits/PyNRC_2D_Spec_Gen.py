import numpy, sys, math, batman
import matplotlib.pyplot as plt
from scipy import interpolate

file = numpy.load('GJ436b_Trans_SED.npz')
SEDarray = file['SEDarray']
print(SEDarray.shape)
plt.imshow(SEDarray)
plt.show()

stellarwave, stellarspec = numpy.loadtxt('ODFNEW_GJ436.spec', unpack=True, skiprows=800)
stellarwave /= 10000. # to um
relevant = numpy.where((stellarwave>1.5) & (stellarwave<5.5))
stellarwave = stellarwave[relevant]
stellarspec = stellarspec[relevant]
StellarInterp = interpolate.interp1d(stellarwave, stellarspec, kind='cubic')

planetwave, planetspec = numpy.loadtxt('../Transmission_Spec/GJ436b_trans_PyNRC_GRISMR.txt', unpack=True)
PlanetInterp = interpolate.interp1d(planetwave, planetspec, kind='cubic')

time = numpy.linspace(0.0,0.1,5000)

f = open('../BATMAN_Generation/Used/BatmanParams_PyNRC_GRISMR.txt', 'r')

params = batman.TransitParams
params.t0 = float(f.readline().split('=')[1]) # hardcoded readlines b/c the file I'm using has a fixed format
params.per = float(f.readline().split('=')[1])
params.inc = float(f.readline().split('=')[1])
params.rp = float(f.readline().split('=')[1])
params.a = float(f.readline().split('=')[1])
params.w = float(f.readline().split('=')[1])
params.ecc = float(f.readline().split('=')[1])
params.fp = float(f.readline().split('=')[1])
params.t_secondary = float(f.readline().split('=')[1])

limbdark = f.readline().split('=')[1] # ugh
u1 = float(limbdark.split(',')[0][2:])
u2 = float(limbdark.split(',')[1][1:-2])
params.u = [u1, u2]
params.limb_dark = "quadratic"

transitmodel = batman.TransitModel(params, time) # creates a transit model object using the time array; we can change the depth now by changing what's in params

SEDarray = numpy.zeros(time.shape[0]) # initialize so that we can vstack onto this

wave = numpy.linspace(1.75,5.25,3500)
for waveval in wave:
	params.rp = math.sqrt(PlanetInterp(waveval)) # sqrt b/c trans. spec is in depth, but batman wants rp/rs
	fluxtransit = transitmodel.light_curve(params)
	actualflux = fluxtransit * StellarInterp(waveval)
	SEDarray = numpy.vstack((SEDarray, actualflux))

SEDarray = numpy.delete(SEDarray, 0, 0) # trim that initial row with all zeroes

numpy.savez('GJ436b_Trans_SED', SEDarray=SEDarray, time=time, wave=wave)

plt.imshow(SEDarray)
plt.show()