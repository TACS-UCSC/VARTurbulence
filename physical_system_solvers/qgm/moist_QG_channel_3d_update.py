"""
Moist QG channel model developed by N Lutsko at Scripps
See Lutsko and Hell (2021) for model description.

email for questions: nlutsko@ucsd.edu
"""
import numpy as np
import matplotlib.pylab as plt
from random import random
from numba import jit
import scipy.fft as sci_fft
import time as ti
import qg_io_3d
import sys,os
import pickle
import argparse
#######################################################
#  Declare some parameters, arrays, etc.

data_dir = "/hb/scratch/llupinji/qgm_sim"

parser = argparse.ArgumentParser(
                    prog='Quasi Geostrophic Turbulence and Moisture Numerical Simulation',
                    description='',
                    epilog='')

parser.add_argument("-s", "--seed", default = None, help="the starting seed used to generate starting conditions", type=int)

parser.add_argument("-l", "--loadstate", default = "cold", help="cold start or loading start from previous", type=str)

parser.add_argument("-n", "--noise", default = "none", help="adding random noise to initial conditions. ''none'', ''gauss''", type=str)

parser.add_argument("-d", "--dir_out", default = None, help="output directory", type=str)

parser.add_argument("-r", "--dir_resume", default = None, help="load initial condition directory", type=str)

parser.add_argument("-dd", "--data_dir", default = data_dir, help="root data dir", type=str)

parser.add_argument("-nf", "--noise_factor", default = 0.2, help="noise factor for gauss", type=float)

parser.add_argument("-tt", "--tot_time", default = 20000, help="noise factor for gauss", type=int)

random_noise_choices = {"none", "gauss"}
loadstate_choices = {"cold", "load"}

args = parser.parse_args()

SEED = args.seed
LOAD = args.loadstate
NOISE = args.noise
DIR_OUT = args.dir_out
DIR_RESUME = args.dir_resume
NOISE_FACTOR = 0.2
TOT_TIME = args.tot_time

dir_out = DIR_OUT
dir_resume = DIR_RESUME

print(args)

if SEED is None:
  SEED = int(np.random.uniform(0,100))

if LOAD not in loadstate_choices:
  raise ValueError(f"--loadstate flag not valid. Must be in {loadstate_choices}")
  
if NOISE not in random_noise_choices:
  raise ValueError(f"--noise flag not valid. Must be in {random_noise_choices}")

# SEED=int(sys.argv[1])
# LOAD=str(sys.argv[2])

print('Random SEED:', SEED)
np.random.seed(SEED)

if dir_out is None and LOAD == "cold":
    dir_out = data_dir+'/'+f"cold_seed-{str(SEED)}"+'/'
if dir_out is None and LOAD == "load":
    if not (os.path.exists(dir_resume) or dir_resume is None):
        raise ValueError("argument dir_resume must be defined and a valid directory")
    dir_out = data_dir+'/'+f"load_resDir-{os.path.basename(dir_resume)}_noise-{NOISE}_seed-{str(SEED)}"+'/'
print(f"Output directory: {dir_out}")
# dir_out = DIR_OUT

from pathlib import Path
Path(dir_out).mkdir(parents=True, exist_ok=True)

## creates file for storing the data. will load in and do small permutations with this
filename = dir_out+"output.2d.nc"
filename3 = dir_out+"output.3d.nc"
res_filename = dir_out+"res"
res_filename_init = dir_out+"res_init"

# SEED=int(42)

#namelist
#====================================================================
# Copied from namelist.py , change for conditions 
#====================================================================
###########NAMELIST for NONDIMENSIONAL PARAMETERS##########
# J.Kang#
# model option
import os
opt = 3 # 1 = just the linear parts, 2 = just the nonlinear parts, 3 = full model
model= 'moist'

# domain
N = 256 #zonal size of spectral decomposition
N2 = 256 #meridional size of spectral decomposition
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
init = LOAD #cold = cold start, load = load data from res_filename

tot_time = TOT_TIME #Length of run (in model time-units), 4 units ~ 1 day prediction
dt = 0.025 #Timestep
ts = int( float(tot_time) / dt ) #Total timesteps
#lim = 50  #Start saving after this time (model time-units), will be set to 0 if this is a restart
## will just keep this as 0 regardless
lim = 0  #Start saving after this time (model time-units), will be set to 0 if this is a restart
st = 1  #How often to record data (in model time-units)
# st = int(1.0/dt)  #How often to record data (in model time-units)

parameters = {
              "opt" : opt,
              "model" : model,
              "dir_resume" : dir_resume, # filename state is loaded from
              "seed" : SEED,
              "loadstate" : LOAD,
              "N" : N, #zonal size of spectral decomposition
              "N2" : N2, #meridional size of spectral decomposition
              "Lx" : Lx, #size of x (in units of Rossby radius)
              "Ly" : Ly, #size of y (in units of Rossby radius)

              # free parameters
              "nu" : nu, #viscous dissipation
              "tau_d" : tau_d, #Newtonian relaxation time-scale for interface
              "tau_f" : tau_f, #surface friction
              "beta" : beta, #beta
              "sigma" : sigma, # characteristic jet width
              "U_1" : U_1, # maximum equilibrium wind

              # moisture parameters, will be suppressed for dry run
              "C" : C, #linearized Clausius-Clapeyron parameter
              "L" : L, #non-dimensional measure of the strength of latent heating
              "Er" : Er, #Evaporation rate

              #running options
              "g" : g, #leapfrog filter coefficient
              "init" : init, #cold = cold start, load = load data from res_filename

              #time and save options
              "tot_time" : tot_time, #Length of run (in model time-units)
              "dt" : dt, #Timestep
              "ts" : ts, #Total timesteps
              "lim" : lim,  #Start saving after this time (model time-units), will be set to 0 if this is a restart
              "st" : st,  #How often to record data (in model time-units)..
             }

with open(dir_out+"parameters.pkl", "wb") as h:
    pickle.dump(parameters, h)
    
    
x = np.linspace( -Lx / 2, Lx / 2, N ,endpoint=False) 
y = np.linspace( -Ly / 2, Ly / 2, N2 ,endpoint=False) 

#Wavenumbers:
kk = np.fft.rfftfreq( N, Lx / float(N) / 2. / np.pi ) #zonal wavenumbers
ll = np.fft.fftfreq( N2, Ly / float(N2) / 2. / np.pi ) #meridional wavenumbers

Lapl = -(np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2)

fft = sci_fft
nworker = 4


#######################################################
#  Declare arrays

#Spectral arrays, only need 3 time-steps
psic_1 = np.zeros( ( ( 3 , N2 , int(N / 2 + 1) ) ) ).astype( complex )
psic_2 = np.zeros( ( ( 3 , N2 , int(N / 2 + 1) ) ) ).astype( complex )
qc_1 = np.zeros( ( ( 3 , N2, int(N / 2 + 1) ) ) ).astype( complex )
qc_2 = np.zeros( ( ( 3 , N2 , int(N / 2 + 1) ) ) ).astype( complex )
vorc_1 = np.zeros( ( ( 3 , N2, int(N / 2 + 1)  ) ) ).astype( complex )
vorc_2 = np.zeros( ( ( 3 , N2 , int(N / 2 + 1) ) ) ).astype( complex )

print('complex spectral array shapes: ' + str(psic_1.shape))

mc = np.zeros( ( ( 3, N2, N // 2 + 1 ) ) ).astype( complex ) #moisture

#Real arrays, only need 3 time-steps
psi_1 = np.zeros( ( ( 3 , N2 , N ) ) )
psi_2 = np.zeros( ( ( 3 , N2 , N ) ) )
q_1 = np.zeros( ( ( 3 , N2, N ) ) )
q_2 = np.zeros( ( ( 3 , N2 , N ) ) )

print('real array shapes: ' + str(psi_1.shape))

#Arrays for saving data
## this is wrong?
TN=int((tot_time - lim) // st) # J Kang editted here to deal with st < 1 cases.
zu1 = np.zeros( ( TN, N2 ) )
zu2 = np.zeros( ( TN, N2 ) )
ztau = np.zeros( ( TN, N2 ) )
zm = np.zeros( ( TN , N2 ) )
zP = np.zeros( ( TN , N2 ) )
zE = np.zeros( ( TN , N2 ) )
zw = np.zeros( ( TN , N2 ) )
zwskew = np.zeros( ( TN , N2 ) )
zeke1 = np.zeros( ( TN , N2 ) )
zeke2 = np.zeros( ( TN , N2 ) )
zemf1 = np.zeros( ( TN , N2 ) )
zemf2 = np.zeros( ( TN , N2 ) )
zehf1 = np.zeros( ( TN , N2 ) )
zehf2 = np.zeros( ( TN , N2 ) )

# saving 3d data (added by J.Kang)
tu1 = np.zeros( ( TN, N2 , N ) )
tu2 = np.zeros( ( TN, N2 , N ) )
tv1 = np.zeros( ( TN, N2 , N ) )
tv2 = np.zeros( ( TN, N2 , N ) )
ttau = np.zeros( ( TN, N2 , N ) )
tq1 = np.zeros( ( TN, N2 , N ) )
tq2 = np.zeros( ( TN, N2 , N ) )
tm = np.zeros( ( TN, N2 , N ) )
tP = np.zeros( ( TN, N2 , N ) )
tE = np.zeros( ( TN, N2 , N ) )
tw = np.zeros( ( TN, N2 , N ) )
tpsi_1 = np.zeros( ( TN, N2 , N ) )
tpsi_2 = np.zeros( ( TN, N2 , N ) )


#######################################################
#  Define equilibrium interface height + sponge

sponge = np.zeros( N2 )
u_eq = np.zeros( ( N2, N ) )

for i in range( N2 ):
	y1 = float( i - N2 /2) * (y[1] - y[0] )
	y2 = float(min(i, N2 -i - 1)) * (y[1] - y[0] )
	sponge[i] = U_1 / (np.cosh(abs(y2/sigma)))**2 
	u_eq[i, :] = U_1 * ( 1. / (np.cosh(abs(y1/sigma)))**2 - 1. / (np.cosh(abs(y2/sigma)))**2  )

psi_Rc = -np.fft.rfft2(  u_eq ) / 1.j / ll[:, np.newaxis]
psi_Rc[0] = 0.
psi_R = np.fft.irfft2( psi_Rc )


#######################################################
#  Spectral functions

@jit(nopython=True)
def ptq(kk, ll, ps1, ps2):
	"""
	Calculate PV
	in:
	meridional wavemnumber ll, zonal wavenumber kk, psi1(l,k), psi2(l,k)
	"""
	q1 = -(np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2 ) * ps1 - (ps1 - ps2) # -(k^2 + l^2) * psi_1 -0.5*(psi_1-psi_2)
	q2 = -(np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2 ) * ps2 + (ps1 - ps2) # -(k^2 + l^2) * psi_2 +0.5*(psi_1-psi_2)
	return q1, q2


@jit(nopython=True)
def qtp(kk, ll, q1_s, q2_s):
	"""Invert PV"""
	divider =  ( np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2)  # (psi_1 + psi_2)/2
	divider[0, 0] = np.nan
	psi_bt = -(q1_s + q2_s) / divider / 2.0  # (psi_1 + psi_2)/2
	psi_bt[0, 0] = 0.

	psi_bc = -(q1_s - q2_s) / (np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2 + 2. ) / 2.  # (psi_1 - psi_2)/2

	psi1 = psi_bt + psi_bc
	psi2 = psi_bt - psi_bc

	return psi1, psi2

@jit(nopython=True)
def qtp_3d(kk, ll, q1_s, q2_s):
	"""Invert PV"""

	divider =  ( np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2)  # (psi_1 + psi_2)/2
	divider[0, 0] = np.nan
	psi_bt = -(q1_s + q2_s) / divider  /2.0 # (psi_1 + psi_2)/2
	psi_bt[:, 0, 0] = 0.

	psi_bc = -(q1_s - q2_s) / (np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2 + 2. ) / 2.  # (psi_1 - psi_2)/2

	psi1 = psi_bt + psi_bc
	psi2 = psi_bt - psi_bc

	return psi1, psi2

@jit(nopython=True)
def grad( field, kk, ll ):

	d1, d2 = np.shape( field )
	grad = np.zeros( ( d1, d2 ) ) + 0.j
	grad[:, :] = 1.j * np.expand_dims(ll, 1) + 1.j * np.expand_dims(kk, 0)

	return grad * field

@jit(nopython=True)
def Laplace( field, kk, ll ):

	d1, d2 = np.shape( field )
	Lapl = np.zeros( ( d1, d2 ) )
	Lapl[:, :] = -(np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2)


	return Lapl * field

@jit(nopython=True, parallel = True)
def exponential_cutoff( data, a, s, kcut ):
    d1, d2 = np.shape( data )
    F = np.ones( ( d1, d2 ) )
    for i in range( d1 ):
        for j in range( d2 ):
            if i > 9 and i <= d1 / 2:
                F[i, j] *= np.exp( -a * ((float(i - kcut)/float(d1 / 2 - 1 - kcut) )) ** s )
            elif i > d1 / 2 and i < (d1 - 10 ):
                k = d1 - i
                F[i, j] *= np.exp( -a * ((float(k - kcut)/float(d1 / 2 - 1 - kcut) )) ** s )
            if j > 9:
                F[i, j] *= np.exp( -a * ((float(j - kcut)/float(d2 - 1 - kcut) )) ** s )
    return F



#######################################################
#  Initial conditions:

## to modify to include initial conditions and gaussian noise additions

if init == "cold":
    print(f"Cold start...")
    ds, zu1n, zu2n, ztaun, mn, Pn, En, wn, wskewn, eke1n, eke2n, emf1n, emf2n, ehf1n, ehf2n, time = qg_io_3d.create_file( filename, y, int(tot_time - lim))
    # J.Kang added this line
    ds3, u1n, u2n, v1n, v2n, taun, q1n, q2n, mn, Pn, En, wn, psi1n, psi2n, time = qg_io_3d.create_file_xyt( filename3, x, y, int(tot_time - lim))

    psic_1[0] = [ [ random() for i in range(N // 2 + 1 ) ] for j in range(N2) ]
    psic_2[0] = [ [ random() for i in range(N // 2 + 1 ) ] for j in range(N2) ]

    #Transfer values:
    psic_1[ 1 , : , : ] = psic_1[ 0 , : , : ]
    psic_2[ 1 , : , : ] = psic_2[ 0 , : , : ]

    #Calculate initial PV
    for i in range( 2 ):
        vorc_1[i], vorc_2[i] = ptq(kk, ll, psic_1[i], psic_2[i]) 
        q_1[i] = fft.irfft2( vorc_1[i], workers=nworker) + beta * y[:, np.newaxis]
        q_2[i] = fft.irfft2( vorc_2[i], workers=nworker) + beta * y[:, np.newaxis]
        qc_1[i] = fft.rfft2( q_1[i], workers=nworker )
        qc_2[i] = fft.rfft2( q_2[i], workers=nworker )

    #Start moisture at 50% saturation
    psi1 = fft.irfft2( psic_1[1], workers=nworker )
    psi2 = fft.irfft2( psic_2[1], workers=nworker )
 
    #Start at uniform 50% saturation
    m = C * (psi1 - psi2) / 2.

    mc[0] = fft.rfft2( m.real, workers=nworker )
    mc[1] = fft.rfft2( m.real, workers=nworker )

    t0 = 1

elif init == "load":
    # print(f"Resuming from {res_filename}...")
    # psic_1, psic_2, qc_1, qc_2, mc, t0 = qg_io_3d.load_res_file(  res_filename )
    # if model == "moist":
         # ds, zu1, zu2, ztau, zeke1, zeke2, zemf1, zemf2, zehf1, zehf2, zm, zP, zE, zw, zwskew, time = qg_io_3d.load_moist_data( filename ) 
    # else:
         # ds, zu1, zu2, ztau, zeke1, zeke2, zemf1, zemf2, zehf1, zehf2, time = qg_io_3d.load_dry_data( filename )
    print(f"Resuming from {dir_resume}...")
    
    filename_res = os.path.join(dir_resume, "output.2d.nc")
    filename3_res = os.path.join(dir_resume,"output.3d.nc")
    npz_res = os.path.join(dir_resume, "res")
    
    psic_1, psic_2, qc_1, qc_2, mc, t0_init = qg_io_3d.load_res_file(  npz_res )
    ## starting at the 0 step
    ## psis look ok from the load file
    t0 = 0
    
    if model == "moist":        
        ds, zu1n, zu2n, ztaun, mn, Pn, En, wn, wskewn, eke1n, eke2n, emf1n, emf2n, ehf1n, ehf2n, time = qg_io_3d.create_file( filename, y, int(tot_time - lim))
        # J.Kang added this line
        ds3, u1n, u2n, v1n, v2n, taun, q1n, q2n, mn, Pn, En, wn, psi1n, psi2n, time = qg_io_3d.create_file_xyt( filename3, x, y, int(tot_time - lim))        
    
    ## Permutations for the initial conditions of the loaded models
    nfact = NOISE_FACTOR
    if NOISE == "gauss":
      print(f"adding gauss noise, noise factor: {nfact}")
      psic_1_std = np.std(psic_1)
      psic_2_std = np.std(psic_2)
      gnoise1 = nfact*np.random.normal(0, psic_1_std, psic_1.shape)
      gnoise2 = nfact*np.random.normal(0, psic_2_std, psic_2.shape)
      psic_1 = psic_1+gnoise1
      psic_2 = psic_2+gnoise2
      
#######################################################
#  Time-stepping functions

def calc_nl( psi, qc ):
    """"Calculate non-linear terms, with Orszag 3/2 de-aliasing"""

    N2, N = np.shape( psi )
    ex = int(N *  3 / 2)# - 1
    ex2 = int(N2 * 3 / 2)# - 1
    temp1 = np.zeros( ( ex2, ex ) ).astype( complex )
    temp2 = np.zeros( ( ex2, ex ) ).astype( complex )
    temp4 = np.zeros( ( N2, N ) ).astype( complex )	#Final array

    #Pad values:
    temp1[:N2//2, :N] = psi[:N2//2, :N]
    temp1[ex2-N2//2:, :N] = psi[N2//2:, :N]

    temp2[:N2//2, :N] = qc[:N2//2, :N]
    temp2[ex2-N2//2:, :N] = qc[N2//2:, :N]

    #Fourier transform product, normalize, and filter:
    temp3 = np.fft.rfft2( np.fft.irfft2( temp1 ) * np.fft.irfft2( temp2 ) ) * 9. / 4.
    temp4[:N2//2, :N] = temp3[:N2//2, :N]
    temp4[N2//2:, :N] = temp3[ex2-N2//2:, :N]

    return temp4

@jit(nopython=True)
def jacobian_prep(kk, ll, psi, qc):
	#kk, ll, psi, qc = kk, ll, psic_1[1, :, :], vorc_1[1, :, :]

	kk2, ll2 =np.expand_dims(kk, 0), np.expand_dims(ll, 1)
	dpsi_dx = 1.j * kk2 * psi
	dpsi_dy = 1.j * ll2 * psi

	dq_dx = 1.j * kk2 * qc
	dq_dy = 1.j * ll2 * qc
	return dpsi_dx, dq_dy, dpsi_dy ,dq_dx

def nlterm(kk, ll, psi, qc):
    """"Calculate Jacobian"""

    dpsi_dx, dq_dy, dpsi_dy ,dq_dx = jacobian_prep(kk, ll, psi, qc) 

    return  calc_nl( dpsi_dx, dq_dy ) - calc_nl( dpsi_dy, dq_dx )

@jit(nopython=True)
def fs(ovar, rhs, det, nu, kk, ll):
    """Forward Step: q^t-1 / ( 1 + 2. dt * nu * (k^4 + l^4 ) ) + RHS"""
    mult = det / ( 1. + det * nu * (np.expand_dims(kk, 0) ** 4 + np.expand_dims(ll, 1) ** 4) )

    return mult * (ovar / det + rhs)

@jit(nopython=True)
def lf(oovar, rhs, det, nu, kk, ll):
    """Leap frog timestepping: q^t-2 / ( 1 + 2. * dt * nu * (k^4 + l^4 ) ) + RHS"""
    mult = 2. * det / ( 1. + 2. * det * nu * (np.expand_dims(kk, 0) ** 4 + np.expand_dims(ll, 1) ** 4) )
    return mult * (oovar / det / 2. + rhs)

@jit(nopython=True)
def filt(var, ovar, nvar, g):
	"""Leapfrog filtering"""
	return var + g * (ovar - 2. * var + nvar )


#######################################################
#  Main time-stepping loop

forc1 = np.zeros( ( N2, N ) )
forc2 = np.zeros( ( N2, N ) )
cforc1 = np.zeros( ( N2, N // 2 + 1 ) ).astype(complex)
cforc2 = np.zeros( ( N2, N // 2 + 1  ) ).astype(complex)

nl1 = np.zeros( ( N2, N // 2 + 1  ) ).astype(complex)
nl2 = np.zeros( ( N2, N // 2 + 1 ) ).astype(complex)

mnl = np.zeros( ( N2, N // 2 + 1 ) ).astype(complex)

F = exponential_cutoff( psic_1[0], np.log(1. + 400. * np.pi / float(N) ), 6, 7 )

## saving initial conditions for reset if needed
qg_io_3d.write_res_files( res_filename_init, psic_1, psic_2, qc_1, qc_2, mc, i )

#Timestepping:
for i in range( t0, ts ):
    start = ti.time()
    if i % 1000 == 0:
        print("Timestep:", i, " / ", ts)

    if opt > 1:
	#NL terms -J(psi, qc) - beta * v
        nl1[:, :] = -nlterm( kk, ll, psic_1[1, :, :], vorc_1[1, :, :]) - beta * 1.j * kk[np.newaxis, :] * psic_1[1, :, :]
        nl2[:, :] = -nlterm( kk, ll, psic_2[1, :, :], vorc_2[1, :, :]) - beta * 1.j * kk[np.newaxis, :] * psic_2[1, :, :]
        mnl[:, :] = -nlterm( kk, ll, psic_2[1, :, :], mc[1, :, :])

    if opt != 2:
	#Linear terms
	#Relax interface
        forc1[:, :] = (psi_1[1] - psi_2[1] - psi_R) / tau_d 
        forc2[:, :] = -(psi_1[1] - psi_2[1] - psi_R) / tau_d

	#Sponge
  ## mimicking periodicity in the y direction
  ## alternative is chebyshev, but expensive 
        forc1[:, :] -= sponge[:, np.newaxis] * (q_1[1] - np.mean( q_1[1], axis = 1)[:, np.newaxis] )
        forc2[:, :] -= sponge[:, np.newaxis] * (q_2[1] - np.mean( q_2[1], axis = 1)[:, np.newaxis] )

        #Convert to spectral space + add friction
        cforc1 = fft.rfft2( forc1, workers =nworker)
        cforc2 = fft.rfft2( forc2, workers =nworker) - Laplace( psic_2[1], kk, ll ) / tau_f

    rhs1 = nl1[:] + cforc1[:]
    rhs2 = nl2[:] + cforc2[:]
    mrhs = mnl[:]
	
    if i == 1 and init == 'cold':
	#Forward step
        qc_1[2, :] = fs(qc_1[1, :, :], rhs1[:], dt, nu, kk, ll)
        qc_2[2, :] = fs(qc_2[1, :, :], rhs2[:], dt, nu, kk, ll)
        mc[2, :] = fs(mc[1, :, :], mrhs[:], dt, nu, kk, ll)
    else:
	#Leapfrog step
        qc_1[2, :, :] = lf(qc_1[0, :, :], rhs1[:], dt, nu, kk, ll)
        qc_2[2, :, :] = lf(qc_2[0, :, :], rhs2[:], dt, nu, kk, ll)
        mc[2, :] = lf(mc[0, :, :], mrhs[:], dt, nu, kk, ll)


    if model == "moist":
       if opt > 1:
	    #NL terms -J(psi, m)
            mnl[:, :] = -nlterm( kk, ll, psic_2[1, :, :], mc[1, :, :])
        
       mrhs = mnl[:]
       if i == 1 and init == 'cold':
	    #Forward step
            mc[2, :] = fs(mc[1, :, :], mrhs[:], dt, nu, kk, ll)
       else:
	    #Leapfrog step
            mc[2, :] = lf(mc[0, :, :], mrhs[:], dt, nu, kk, ll)


       #############################################
       #Calculate precip, then adjust fields
       #x 1.5 ms
       #timeit

       #Convert to real space
       m = fft.irfft2(mc[2], workers =nworker)
       ## pass this psi_1 and psi_2 to the save
       ## save current psis to tensor, then pass at each of the write steps 
       psi_1[1] = fft.irfft2( psic_1[1], workers =nworker)
       psi_2[1] = fft.irfft2( psic_2[1], workers =nworker)
       u2 = fft.irfft2( -1.j * np.expand_dims(ll, 1) * psic_2[1], workers =nworker) # -i * k * psic2
       v2 = fft.irfft2( 1.j * np.expand_dims(kk, 0) * psic_2[1], workers =nworker)

       #Calculate precip
       sat_def = (C * (psi_1[1, :,:] - psi_2[1, :, :]) ) - m

       sat_def_mask=sat_def < 0.

       P = np.where(sat_def_mask, -sat_def , 0)
       E = np.where(~sat_def_mask, Er * np.sqrt(u2 ** 2 + v2 ** 2) * sat_def, 0)

       Pc = fft.rfft2( P, workers =nworker) * F #F is exponential cut-off filter
       Ec = fft.rfft2( E, workers =nworker) * F #F is exponential cut-off filter

       #Adjust fields, time-step
       rhs1 -= L * Pc
       rhs2 += L * Pc
       mrhs -= Pc - Ec
	
       if i == 1 and init == 'cold':
          #Forward step
          qc_1[2, :] = fs(qc_1[1, :, :], rhs1[:], dt, nu, kk, ll)
          qc_2[2, :] = fs(qc_2[1, :, :], rhs2[:], dt, nu, kk, ll)
       else:
	  #Leapfrog step
          qc_1[2, :] = lf(qc_1[0, :, :], rhs1[:], dt, nu, kk, ll)
          qc_2[2, :] = lf(qc_2[0, :, :], rhs2[:], dt, nu, kk, ll)

       vort1 = Laplace( psic_1[1], kk, ll )
       term1 = nlterm( kk, ll, psic_1[1], vort1)
       vort2 = Laplace( psic_2[1], kk, ll )
       term2 = nlterm( kk, ll, psic_2[1], vort2)
       tauc = psic_1[1] - psic_2[1]
       term3 = Laplace( nlterm( kk, ll, psic_2[1], tauc), kk, ll )
       v1 = 1.j * np.expand_dims(kk, 0) * psic_1[1, :, :]
       v2 = 1.j * np.expand_dims(kk, 0) * psic_2[1, :, :]
       term4 = beta * (v1 - v2) 
       term5 = Laplace( L * Pc, kk, ll )
       term6 = Laplace( tauc - psi_Rc, kk, ll ) / tau_d
       div_ageo = term1 - term2 - term3 + term4 + term5 - term6
       div_ageo /= (np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2 + 2)

       mforc = sponge[:, np.newaxis] * (m - np.mean( m, axis = 1)[:, np.newaxis] )
       mrhs -= div_ageo + np.fft.rfft2(mforc)
    #endif
    
    #x 3ms
    #timeit
    if i == 1 and init == 'cold':
        #Forward step
        mc[2, :] = fs(mc[1, :, :], mrhs[:], dt, nu, kk, ll)
    else:
        #Leapfrog step
        mc[2, :] = fs(mc[0, :, :], mrhs[:], dt, nu, kk, ll)

    if i > 1:
	#Leapfrog filter
        qc_1[1, :] = filt( qc_1[1, :], qc_1[0, :], qc_1[2, :], g)
        qc_2[1, :] = filt( qc_2[1, :], qc_2[0, :], qc_2[2, :], g)
        mc[1, :] = filt( mc[1, :], mc[0, :], mc[2, :], g)

    q_1[0:2] = fft.irfft2( qc_1[1:], workers =nworker)
    q_2[0:2] = fft.irfft2( qc_2[1:], workers =nworker)

    #Subtract off beta and invert
    vorc_1[0:2] = fft.rfft2( q_1[0:2] - beta * y[:, np.newaxis], workers =nworker)
    vorc_2[0:2] = fft.rfft2( q_2[0:2] - beta * y[:, np.newaxis], workers =nworker)
    psic_1[0:2], psic_2[0:2] = qtp_3d( kk, ll, vorc_1[0:2], vorc_2[0:2] )

    psi_1[0:2] = fft.irfft2( psic_1[0:2] , workers =nworker)
    psi_2[0:2] = fft.irfft2( psic_2[0:2] , workers =nworker)

    #Transfer values:
    qc_1[0:2, :, :] = qc_1[1:, :, :]
    qc_2[0:2, :, :] = qc_2[1:, :, :]
    mc[0:2, :, :] = mc[1:, :, :]

    if int(float(i) * dt) >= lim: 
        if i % int(float(st) / dt) == 0: 
            time[int(i * dt) - lim] = float(i) * dt

            u1 = fft.irfft2( -1.j * ll[:, np.newaxis] * psic_1[1], workers =nworker)
            u2 = fft.irfft2( -1.j * ll[:, np.newaxis] * psic_2[1], workers =nworker)
            v1 = fft.irfft2( 1.j * kk[np.newaxis, :] * psic_1[1], workers =nworker)
            v2 = fft.irfft2( 1.j * kk[np.newaxis, :] * psic_2[1], workers =nworker)
            tau = fft.irfft2( psic_1[1] - psic_2[1], workers =nworker)

            zmu1 = np.mean( u1, axis = 1 )
            zmu2 = np.mean( u2, axis = 1 )
            zmtau = np.mean( tau, axis = 1 )

            eddy_u1 = u1 - zmu1[:, np.newaxis]
            eddy_u2 = u2 - zmu2[:, np.newaxis]
            eddy_v1 = v1 - np.mean( v1, axis = 1)[:, np.newaxis]
            eddy_v2 = v2 - np.mean( v2, axis = 1)[:, np.newaxis]
            eddy_tau = tau - zmtau[:, np.newaxis]
            
            ind=int((i * dt - lim) // st)     #J. Kang added 'ind' and used it below.

            zu1[ind] = zmu1[:]
            zu2[ind] = zmu2[:]
            ztau[ind] = zmtau[:]

            zeke1[ind] = np.mean( eddy_u1 ** 1 + eddy_v1 ** 2, axis = 1) / 2.
            zeke2[ind] = np.mean( eddy_u2 ** 1 + eddy_v2 ** 2, axis = 1) / 2.
            zemf1[ind] = np.mean( eddy_u1 * eddy_v1, axis = 1)
            zemf2[ind] = np.mean( eddy_u2 * eddy_v2, axis = 1)
            zehf1[ind] = np.mean( eddy_v1 * eddy_tau, axis = 1)
            zehf2[ind] = np.mean( eddy_v2 * eddy_tau, axis = 1)
            
            #total data (added by J.Kang)
            tu1[ind] = u1[:]
            tu2[ind] = u2[:]
            tv1[ind] = v1[:]
            tv2[ind] = v2[:]
            ttau[ind] = tau[:]
            tq1[ind] = q_1[1,:]
            tq2[ind] = q_2[1,:]
            tpsi_1[ind] = psi_1[1,:]
            tpsi_2[ind] = psi_2[1,:]
            
            
            if model == "moist":
                m = np.fft.irfft2( mc[1] )
                w = np.fft.irfft2( div_ageo )
                zmw = np.mean(w, axis = 1)	
                zm[ind] = np.mean( m[:, :], axis = 1 )
                zP[ind] = np.mean( P[:, :], axis = 1 )
                zE[ind] = np.mean( E[:, :], axis = 1 )
                zw[ind] = np.mean( w[:, :], axis = 1 )
                eddy_w = w - zmw[:, np.newaxis] 
                zwskew[ind] = -np.mean( eddy_w ** 3, axis = 1 ) / np.mean( eddy_w ** 2, axis = 1 ) ** (3. / 2.)
                tm[ind] = m[:]
                tP[ind] = P[:]
                tE[ind] = E[:]
                tw[ind] = w[:]
        
        ## check to make sure it is saving the right timesteps
        ## if this isn't working correctly, save directly using tensor with timesteps x channels x lat x lon (psi1, psi2, m)
        if (i+1) % int(100/dt) == 0:
        # if (i+1) % int(st) == 0:
            #Sync + save stuff to make it easier to restart
            qg_io_3d.write_res_files( res_filename, psic_1, psic_2, qc_1, qc_2, mc, i )
            if model == "moist":
                print(f'writing-moist: i = {i}')
                qg_io_3d.write_data_moist( ds, zu1, zu2, ztau, zeke1, zeke2, zemf1, zemf2, zehf1, zehf2, zm, zP, zE, zw, zwskew )       
                qg_io_3d.write_data_moist_xyt( ds3, tu1, tu2,tv1, tv2, ttau, tq1, tq2, tm, tP, tE, tw, tpsi_1, tpsi_2)       
            else:
                print(f'writing-dry: i = {i}')
                qg_io_3d.write_data_dry( ds, zu1, zu2, ztau, zeke1, zeke2, zemf1, zemf2, zehf1, zehf2 )
                qg_io_3d.write_data_dry_xyt( ds3, tu1, tu2, tv1, tv2, ttau, tq1, tq2, tpsi_1, tpsi_2)         
        
    end = ti.time()
    if i % 1000 == 0:
        delt = (end - start)
        time_left = delt * (float(ts) - float(i))
        print("1 iteration = %s" % delt)
        print("Estimated time left: %0.1f" % time_left)

print(f"Tot time: {end-start}")
## final update for resume, if planned to continue in another run
qg_io_3d.write_res_files( res_filename, psic_1, psic_2, qc_1, qc_2, mc, i )
if model == "moist":
    print(f'writing-moist: i = {i}')
    qg_io_3d.write_data_moist( ds, zu1, zu2, ztau, zeke1, zeke2, zemf1, zemf2, zehf1, zehf2, zm, zP, zE, zw, zwskew )       
    qg_io_3d.write_data_moist_xyt( ds3, tu1, tu2,tv1, tv2, ttau, tq1, tq2, tm, tP, tE, tw, tpsi_1, tpsi_2)       
else:
    print(f'writing-dry: i = {i}')
    qg_io_3d.write_data_dry( ds, zu1, zu2, ztau, zeke1, zeke2, zemf1, zemf2, zehf1, zehf2 )
    qg_io_3d.write_data_dry_xyt( ds3, tu1, tu2, tv1, tv2, ttau, tq1, tq2, tpsi_1, tpsi_2)         

ds.close()
ds3.close()
