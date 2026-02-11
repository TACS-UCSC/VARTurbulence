#io for moist QG model
from netCDF4 import Dataset
import numpy as np

def create_file( filename, y, t ):
	ds = Dataset(filename, mode='w')

	ds.createDimension('time', size=t)
	ds.createDimension('y', size=y.shape[0])
	time = ds.createVariable('time', 'f4', dimensions=['time'])
	yn = ds.createVariable('y', 'f4', dimensions=['y'])
	yn.setncatts({'standard_name': 'y', 'units': 'degrees_east'})

	yn[:] = y
	time = np.linspace(0., float(t) - 1., t)

	zu1n = ds.createVariable(
		    'zu1',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	zu2n = ds.createVariable(
		    'zu2',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	ztaun = ds.createVariable(
		    'ztau',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	mn = ds.createVariable(
		    'zm',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	Pn = ds.createVariable(
		    'zP',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	En = ds.createVariable(
		    'zE',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	wn = ds.createVariable(
		    'zw',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	wskewn = ds.createVariable(
		    'zwskew',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	eke1n = ds.createVariable(
		    'zeke1',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	eke2n = ds.createVariable(
		    'zeke2',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	emf1n = ds.createVariable(
		    'zemf1',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	emf2n = ds.createVariable(
		    'zemf2',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	ehf1n = ds.createVariable(
		    'zehf1',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	ehf2n = ds.createVariable(
		    'zehf2',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)

	zu1n.setncatts({'standard_name': 'zonal-mean u1',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	zu2n.setncatts({'standard_name': 'zonal-mean u2',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	ztaun.setncatts({'standard_name': 'zonal-mean tau',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	mn.setncatts({'standard_name': 'moisture',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	Pn.setncatts({'standard_name': 'precip',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	En.setncatts({'standard_name': 'evap',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	wn.setncatts({'standard_name': 'w',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	wskewn.setncatts({'standard_name': 'w skew',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	eke1n.setncatts({'standard_name': 'eke 1',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	eke2n.setncatts({'standard_name': 'eke 2',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	emf1n.setncatts({'standard_name': 'emf 1',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	emf2n.setncatts({'standard_name': 'emf 2',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	ehf1n.setncatts({'standard_name': 'ehf 1',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	ehf2n.setncatts({'standard_name': 'ehf 2',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})

	return ds, zu1n, zu2n, ztaun, mn, Pn, En, wn, wskewn, eke1n, eke2n, emf1n, emf2n, ehf1n, ehf2n, time
###################################################
#Added by J.Kang to produce (x,y,t) outputs
def create_file_xyt( filename, x, y, t ):
	ds3 = Dataset(filename, mode='w')

	ds3.createDimension('time', size=t)
	ds3.createDimension('y', size=y.shape[0])
	ds3.createDimension('x', size=x.shape[0])
	time = ds3.createVariable('time', 'f4', dimensions=['time'])
	yn = ds3.createVariable('y', 'f4', dimensions=['y'])
	xn = ds3.createVariable('x', 'f4', dimensions=['x'])
	yn.setncatts({'standard_name': 'y', 'units': 'degrees_north'})
	xn.setncatts({'standard_name': 'x', 'units': 'degrees_east'})

	yn[:] = y
	xn[:] = x
	time = np.linspace(0., float(t) - 1., t)

	u1n = ds3.createVariable(
		    'u1',
		    'f4',
		    dimensions=['time', 'y', 'x'],
		    zlib=True)
	u2n = ds3.createVariable(
		    'u2',
		    'f4',
		    dimensions=['time', 'y', 'x'],
		    zlib=True)
	v1n = ds3.createVariable(
		    'v1',
		    'f4',
		    dimensions=['time', 'y', 'x'],
		    zlib=True)
	v2n = ds3.createVariable(
		    'v2',
		    'f4',
		    dimensions=['time', 'y', 'x'],
		    zlib=True)
	taun = ds3.createVariable(
		    'tau',
		    'f4',
		    dimensions=['time', 'y', 'x'],
		    zlib=True)
	q1n = ds3.createVariable(
		    'q1',
		    'f4',
		    dimensions=['time', 'y', 'x'],
		    zlib=True)
	q2n = ds3.createVariable(
		    'q2',
		    'f4',
		    dimensions=['time', 'y', 'x'],
		    zlib=True)
	mn = ds3.createVariable(
		    'm',
		    'f4',
		    dimensions=['time', 'y', 'x'],
		    zlib=True)
	Pn = ds3.createVariable(
		    'P',
		    'f4',
		    dimensions=['time', 'y', 'x'],
		    zlib=True)
	En = ds3.createVariable(
		    'E',
		    'f4',
		    dimensions=['time', 'y', 'x'],
		    zlib=True)
	wn = ds3.createVariable(
		    'w',
		    'f4',
		    dimensions=['time', 'y', 'x'],
		    zlib=True)
	psi1n = ds3.createVariable(
		    'psi1',
		    'f4',
		    dimensions=['time', 'y', 'x'],
		    zlib=True)
	psi2n = ds3.createVariable(
		    'psi2',
		    'f4',
		    dimensions=['time', 'y', 'x'],
		    zlib=True)
  
	u1n.setncatts({'standard_name': 'u1',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	u2n.setncatts({'standard_name': 'u2',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	v1n.setncatts({'standard_name': 'v1',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	v2n.setncatts({'standard_name': 'v2',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	taun.setncatts({'standard_name': 'tau',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	q1n.setncatts({'standard_name': 'q1',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	q2n.setncatts({'standard_name': 'q2',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	mn.setncatts({'standard_name': 'm',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	Pn.setncatts({'standard_name': 'P',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	En.setncatts({'standard_name': 'E',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	wn.setncatts({'standard_name': 'w',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	psi1n.setncatts({'standard_name': 'psi1',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	psi2n.setncatts({'standard_name': 'psi2',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	return ds3, u1n, u2n, v1n, v2n, taun, q1n, q2n, mn, Pn, En, wn, psi1n, psi2n, time

###################################################
def sdat(c, F):
    print("Saving in:", F)
    np.savez(F, u = c)
    return 0

def write_res_files( filename, psic1, psic2, qc1, qc2, mc, t0 ):
	
	sdat( psic1, filename + "_psic1.npz")
	sdat( psic2, filename + "_psic2.npz")
	sdat( qc1, filename + "_qc1.npz")
	sdat( qc2, filename + "_qc2.npz")
	sdat( mc, filename + "_mc.npz")
	sdat( t0, filename + "_t0.npz")

	return 0

def write_data_dry( ds, zu1, zu2, ztau, zeke1, zeke2, zemf1, zemf2, zehf1, zehf2):
	
	ds.variables['zu1'][:] = zu1[:]
	ds.variables['zu2'][:] = zu2[:]
	ds.variables['ztau'][:] = ztau[:]
	ds.variables['zeke1'][:] = zeke1[:]
	ds.variables['zeke2'][:] = zeke2[:]
	ds.variables['zemf1'][:] = zemf1[:]
	ds.variables['zemf2'][:] = zemf2[:]
	ds.variables['zehf1'][:] = zehf1[:]
	ds.variables['zehf2'][:] = zehf2[:]
	ds.sync()

	return 0
  
def write_data_moist( ds, zu1, zu2, ztau, zeke1, zeke2, zemf1, zemf2, zehf1, zehf2, zm, zP, zE, zw, zwskew):
	ds.variables['zu1'][:] = zu1[:]
	ds.variables['zu2'][:] = zu2[:]
	ds.variables['ztau'][:] = ztau[:]
	ds.variables['zeke1'][:] = zeke1[:]
	ds.variables['zeke2'][:] = zeke2[:]
	ds.variables['zemf1'][:] = zemf1[:]
	ds.variables['zemf2'][:] = zemf2[:]
	ds.variables['zehf1'][:] = zehf1[:]
	ds.variables['zehf2'][:] = zehf2[:]
	ds.variables['zm'][:] = zm[:]
	ds.variables['zP'][:] = zP[:]
	ds.variables['zE'][:] = zE[:]
	ds.variables['zw'][:] = zw[:]
	ds.variables['zwskew'][:] = zwskew[:]
	ds.sync()
	return 0

# ###################################################
# #Added by J.Kang to produce (x,y,t) outputs
# def write_data_dry_xyt( ds3, tu1, tu2, tv1, tv2, ttau, tq1, tq2):
	# ds3.variables['u1'][:] = tu1[:]
	# ds3.variables['u2'][:] = tu2[:]
	# ds3.variables['v1'][:] = tv1[:]
	# ds3.variables['v2'][:] = tv2[:]
	# ds3.variables['tau'][:] = ttau[:]
	# ds3.variables['q1'][:] = tq1[:]
	# ds3.variables['q2'][:] = tq2[:]
	# ds3.sync()
	# return 0

def write_data_dry_xyt( ds3, tu1, tu2, tv1, tv2, ttau, tq1, tq2, tpsi_1, tpsi_2):
	ds3.variables['u1'][:] = tu1[:]
	ds3.variables['u2'][:] = tu2[:]
	ds3.variables['v1'][:] = tv1[:]
	ds3.variables['v2'][:] = tv2[:]
	ds3.variables['tau'][:] = ttau[:]
	ds3.variables['q1'][:] = tq1[:]
	ds3.variables['q2'][:] = tq2[:]
	ds3.variables['psi1'][:] = tpsi_1[:]
	ds3.variables['psi2'][:] = tpsi_2[:]
  
	ds3.sync()
	return 0

# def write_data_moist_xyt( ds3, tu1, tu2, tv1, tv2, ttau, tq1, tq2, tm, tP, tE, tw, psi1, psi2):
	# ds3.variables['u1'][:] = tu1[:]
	# ds3.variables['u2'][:] = tu2[:]
	# ds3.variables['v1'][:] = tv1[:]
	# ds3.variables['v2'][:] = tv2[:]
	# ds3.variables['tau'][:] = ttau[:]
	# ds3.variables['q1'][:] = tq1[:]
	# ds3.variables['q2'][:] = tq2[:]
	# ds3.variables['m'][:] = tm[:]
	# ds3.variables['P'][:] = tP[:]
	# ds3.variables['E'][:] = tE[:]
	# ds3.variables['w'][:] = tw[:]
	# ds3.sync()
	# return 0
  
def write_data_moist_xyt( ds3, tu1, tu2, tv1, tv2, ttau, tq1, tq2, tm, tP, tE, tw, tpsi_1, tpsi_2):
	ds3.variables['u1'][:] = tu1[:]
	ds3.variables['u2'][:] = tu2[:]
	ds3.variables['v1'][:] = tv1[:]
	ds3.variables['v2'][:] = tv2[:]
	ds3.variables['tau'][:] = ttau[:]
	ds3.variables['q1'][:] = tq1[:]
	ds3.variables['q2'][:] = tq2[:]
	ds3.variables['m'][:] = tm[:]
	ds3.variables['P'][:] = tP[:]
	ds3.variables['E'][:] = tE[:]
	ds3.variables['w'][:] = tw[:]
  # ## added psi1 and psi2
  
	ds3.variables['psi1'][:] = tpsi_1[:]
	ds3.variables['psi2'][:] = tpsi_2[:]
  
	ds3.sync()
	return 0
##################################################
def load_res_file( filename ):

    fpsic1 = np.load( filename + "_psic1.npz")
    psic1 = fpsic1['u'][:]
    fpsic2 = np.load( filename + "_psic2.npz")
    psic2 = fpsic2['u'][:]
    fqc1 = np.load( filename + "_qc1.npz")
    qc1 = fqc1['u'][:]
    fqc2 = np.load( filename + "_qc2.npz")
    qc2 = fqc2['u'][:]
    fmc = np.load( filename + "_mc.npz")
    mc = fmc['u'][:]
    ft0 = np.load( filename + "_t0.npz")
    t0 = ft0['u']

    return psic1, psic2, qc1, qc2, mc, t0

def load_dry_data( filename ):
    ds = Dataset(filename, mode='a')

    zu1 = ds.variables['zu1'][:]
    zu2 = ds.variables['zu2'][:]
    ztau = ds.variables['ztau'][:]
    zeke1 = ds.variables['zeke1'][:]
    zeke2 = ds.variables['zeke2'][:]
    zemf1 = ds.variables['zemf1'][:]
    zemf2 = ds.variables['zemf2'][:]
    zehf1 = ds.variables['zehf1'][:]
    zehf2 = ds.variables['zehf2'][:]
    time = ds.variables['time'][:]

    return ds, zu1, zu2, ztau, zeke1, zeke2, zemf1, zemf2, zehf1, zehf2, time

def load_moist_data( filename ):
    ds = Dataset(filename, mode='a')

    zu1 = ds.variables['zu1'][:]
    zu2 = ds.variables['zu2'][:]
    ztau = ds.variables['ztau'][:]
    zeke1 = ds.variables['zeke1'][:]
    zeke2 = ds.variables['zeke2'][:]
    zemf1 = ds.variables['zemf1'][:]
    zemf2 = ds.variables['zemf2'][:]
    zehf1 = ds.variables['zehf1'][:]
    zehf2 = ds.variables['zehf2'][:]
    zm = ds.variables['zm'][:]
    zP = ds.variables['zP'][:]
    zE = ds.variables['zE'][:]
    zw = ds.variables['zw'][:]
    zwskew = ds.variables['zwskew'][:]
    time = ds.variables['time'][:]
    
    return ds, zu1, zu2, ztau, zeke1, zeke2, zemf1, zemf2, zehf1, zehf2, zm, zP, zE, zw, zwskew, time

# def write_data_dry_xyt( ds3, tu1, tu2, tv1, tv2, ttau, tq1, tq2, tpsi_1, tpsi_2):
	# ds3.variables['u1'][:] = tu1[:]
	# ds3.variables['u2'][:] = tu2[:]
	# ds3.variables['v1'][:] = tv1[:]
	# ds3.variables['v2'][:] = tv2[:]
	# ds3.variables['tau'][:] = ttau[:]
	# ds3.variables['q1'][:] = tq1[:]
	# ds3.variables['q2'][:] = tq2[:]
	# ds3.variables['psi1'][:] = tpsi_1[:]
	# ds3.variables['psi2'][:] = tpsi_2[:]
  
	# ds3.sync()
	# return 0
  
def load_moist_data_xyt(filename3):
  
    ds3 = Dataset(filename3, mode='a')
    
    # print(ds3.variables)
    u1n = ds3.variables['u1'][:]
    u2n = ds3.variables['u2'][:]
    v1n = ds3.variables['v1'][:]
    v2n = ds3.variables['v2'][:]
    taun = ds3.variables['tau'][:]
    q1n = ds3.variables['q1'][:]
    q2n = ds3.variables['q2'][:]
    mn = ds3.variables['m'][:]
    Pn = ds3.variables['P'][:]
    En = ds3.variables['E'][:]
    wn = ds3.variables['w'][:]
    # ## added psi1 and psi2
    psi1n = ds3.variables['psi1'][:]
    psi2n = ds3.variables['psi2'][:]
  
    return ds3, u1n, u2n, v1n, v2n, taun, q1n, q2n, mn, Pn, En, wn, psi1n, psi2n