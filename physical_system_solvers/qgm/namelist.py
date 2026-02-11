###########NAMELIST for NONDIMENSIONAL PARAMETERS##########
# J.Kang#
# model option
SEED = 1
import os
opt = 3 # 1 = just the linear parts, 2 = just the nonlinear parts, 3 = full model
model= 'moist'

# domain
N = 128 #zonal size of spectral decomposition
N2 = 128 #meridional size of spectral decomposition
Lx = 72. #size of x (in units of Rossby radius)
Ly = 96. #size of y (in units of Rossby radius)

# free parameters
nu = pow( 10., -6. ) #viscous dissipation
tau_d = 100. #Newtonian relaxation time-scale for interface
tau_f = 15. #surface friction
beta = 0.2 #beta
sigma = 3.5 # characteristic jet width
U_1 = 1. # maximum equilibrium wind

# moisture parameters, will be suppressed for dry run
if model == 'moist':
	C = 2. #linearized Clausius-Clapeyron parameter
	L = 0.2 #non-dimensional measure of the strength of latent heating
	Er = .1 #Evaporation rate
if model == 'dry':
	C = 0. #linearized Clausius-Clapeyron parameter
	L = 0. #non-dimensional measure of the strength of latent heating
	Er =0. #Evaporation rate

#running options
g = 0.04 #leapfrog filter coefficient
init = "cold" #cold = cold start, load = load data from res_filename

#time and save options
tot_time = 100 #Length of run (in model time-units)
dt = 0.025 #Timestep
ts = int( float(tot_time) / dt ) #Total timesteps
lim = 50  #Start saving after this time (model time-units), will be set to 0 if this is a restart
st = 1  #How often to record data (in model time-units)

dir_out = "/scratch/06675/tg859749/moist_gfd23/"
from pathlib import Path
Path(dir_out+str(SEED)).mkdir(parents=True, exist_ok=True)
#os.system('mkdir %s'%dir_out) # make output directory
