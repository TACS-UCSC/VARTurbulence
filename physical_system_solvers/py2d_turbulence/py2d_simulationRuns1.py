import sys
sys.path.append("/glade/derecho/scratch/llupinji/scripts/numerical_simulations/py2d")
from py2d.Py2D_solver import Py2D_solver
import os
import scipy.io as sio
import netCDF4 as nc
from datetime import datetime
import numpy as np
import itertools
import matplotlib.pyplot as plt

def num2str(num):
    return str(num).replace("e", "E").replace("+", "").replace("-", "n").replace(".", "p")

# for some reason combined path creation and path name together?
def get_path_names(NX, dt, ICnum, Re, 
             fkx, fky, alpha, beta, SGSModel_string, dealias):
    # Snapshots of data save at the following directory
    if dealias:
        dataType_DIR = 'Re' + str(int(Re)) + '_fkx' + str(fkx) + 'fky' + str(fky) + '_r' + str(alpha) + '_b' + str(beta) + '/'
    else:
        dataType_DIR = 'Re' + str(int(Re)) + '_fkx' + str(fkx) + 'fky' + str(fky) + '_r' + str(alpha) + '_b' + str(beta) + '_alias/'
    SAVE_DIR = 'results/' + dataType_DIR + SGSModel_string + '/NX' + str(NX) + '/dt' + str(dt) + '_IC' + str(ICnum) + '/'
    SAVE_DIR_DATA = SAVE_DIR + 'data/'
    SAVE_DIR_IC = SAVE_DIR + 'IC/'
        
    return SAVE_DIR, SAVE_DIR_DATA, SAVE_DIR_IC

## this needs to change to be able to run on multiple nodes concurrently
Re = [1000, 10000]
forcing_wavenumbers = [(4,4)]
NX = [512]
dt = [1e-4]
tTotal = [20.0]
tSave = [1e-2]
beta = [20] # 100

# standard parameters not predefined
Lx = 2*np.pi

# Makes combinations of all the parameters
combinations = list(itertools.product(Re, forcing_wavenumbers, NX, dt, tTotal, tSave, beta))

# Additional stability conditions for 2D turbulence simulations
for Re_val, (fkx_val, fky_val), NX_val, dt_val, tTotal_val, tSave_val, beta_val in combinations:
    Lx_val = 2 * np.pi
    dx = Lx_val / NX_val
    U_max = 1.0  # conservative estimate

    # CFL condition
    cfl = U_max * dt_val / dx

    # Viscous stability condition: dt < dx^2 / (4 * nu)
    # nu = 1/Re
    nu = 1.0 / Re_val
    viscous_stability = dx**2 / (4 * nu)
    viscous_ok = dt_val < viscous_stability

    # Forcing scale stability (if relevant): dt < 1/(forcing_wavenumber * U_max)
    kf = np.sqrt(fkx_val**2 + fky_val**2)
    if kf > 0:
        forcing_stability = 1.0 / (kf * U_max)
        forcing_ok = dt_val < forcing_stability
    else:
        forcing_stability = None
        forcing_ok = True

    # Print all stability checks
    print(f"Stability checks: Re={Re_val}, fkx={fkx_val}, fky={fky_val}, NX={NX_val}, dt={dt_val}")
    print(f"  CFL: {cfl:.3e} (OK if < 1.0)")
    print(f"  Viscous: dt={dt_val:.2e} < {viscous_stability:.2e} (OK: {viscous_ok})")
    if forcing_stability is not None:
        print(f"  Forcing: dt={dt_val:.2e} < {forcing_stability:.2e} (OK: {forcing_ok})")
    else:
        print(f"  Forcing: Not applicable (kf=0)")

    if not (cfl < 1.0 and viscous_ok and forcing_ok):
        print("  WARNING: One or more stability conditions violated!")
    else:
        print("  All stability conditions satisfied.")

for Re, (fkx, fky), NX, dt, tTotal, tSave, beta in combinations:
    # Script to call the function with the given parameters
    print(f"Running simulation: Re = {Re}, fkx = {fkx}, fky = {fky}, NX = {NX}, dt = {dt}, tTotal = {tTotal}, tSave = {tSave}")
    
    # to check existence of the directory
    SAVE_DIR, SAVE_DIR_DATA, SAVE_DIR_IC = get_path_names(NX=NX, dt=dt, ICnum=1, Re=Re, fkx=fkx, fky=fky, alpha=0.1, beta=0, SGSModel_string='NoSGS', dealias=True)
    if os.path.exists(SAVE_DIR_DATA):
        print(f" Directory {SAVE_DIR_DATA} already exists! Continuing...")
        continue
    else:
        print(f" Directory {SAVE_DIR_DATA} does not exist. Running simulation...")

        Py2D_solver(Re=Re, # Reynolds number
                    fkx=fkx, # Forcing wavenumber in x dimension
                    fky=fky, # Forcing wavenumber in y dimension
                    alpha=0.1, # Rayleigh drag coefficient
                    beta=beta, # Coriolis parameter (Beta-plane turbulence)
                    NX=NX, # Number of grid points in x and y (Presuming a square domain) '32', '64', '128', '256', '512'
                    forcing_filter = None, # None, "gaussian", "box"
                    SGSModel_string='NoSGS', # SGS closure model/parametrization to use. 'NoSGS' (no closure) for DNS simulations. Available SGS models: 'SMAG', 'DSMAG', 'LEITH', 'DLEITH', 'PiOmegaGM2', 'PiOmegaGM4', 'PiOmegaGM6'
                    eddyViscosityCoeff=0, # Coefficient for eddy viscosity models: Only used for SMAG and LEITH SGS Models
                    dt=dt, # Time step
                    dealias=True, # Dealiasing
                    saveData=True, # Save data: The saved data directory would be printed.
                    tSAVE=tSave, # Time interval to save data
                    tTotal=tTotal, # Length (total time) of simulation
                    readTrue=False, 
                    ICnum=1, # Initial condition number: Choose between 1 to 20
                    resumeSim=False, # start new simulation (False) or resume simulation (True) 
                    )

## because the data is already in netcdf format, we can load it from the netcdf file. otherwise, we would have to load it from the mat files.
py2d_sims_dir = "/glade/derecho/scratch/llupinji/data/py2d_sims5_beta1"
if not os.path.exists(py2d_sims_dir):
    os.makedirs(py2d_sims_dir)

for Re, (fkx, fky), NX, dt, tTotal, tSave, beta in combinations:
    mat_dir = f"./results/Re{int(Re)}_fkx{fkx}fky{fky}_r0.1_b{beta}/NoSGS/NX{NX}/dt{dt}_IC1/data"

    # identifying the matrix names and their numbers
    step_names = os.listdir(mat_dir)
    step_names = [name for name in step_names if name.endswith('.mat')]
    step_nums = [int(name.split('.')[0]) for name in step_names]
    step_locs = [os.path.join(mat_dir, name) for name in step_names]
    step_locs_nums = list(zip(step_locs, step_nums))
    step_locs_nums.sort(key=lambda x: x[1])

    sample_data = sio.loadmat(step_locs_nums[0][0])
    sample_data_omega = sample_data['Omega']

    Nx, Ny = sample_data_omega.shape
    dx = Lx/Nx
    dy = Lx/Ny

    numchannels = 1
    lead = 0 # same time predictions

    nc_output_path = os.path.join(py2d_sims_dir, f"turb2d_data_combined_Re-{int(Re)}_force_fkx-{fkx}_fky-{fky}_NX-{NX}_dt-{num2str(dt)}_tTotal-{num2str(tTotal)}_beta-{beta}.nc")

    if not os.path.exists(nc_output_path):
    # if True:
        print(f"Loading data from {mat_dir}")
        data_all = np.zeros((len(step_locs_nums), 1, Nx, Ny))
        timesteps = []
        # Define data paths
        # data_dir = setup["data_dir"]
        for i, (step_loc, step_num) in enumerate(step_locs_nums):
            train_data = sio.loadmat(step_loc)
            data_step = np.array(train_data['Omega'], dtype=np.float32)
            data_all[i,0] = data_step
            timesteps.append(step_num*tSave)
            if i % 500 == 0:
                print(f"Loaded {i} of {len(step_locs_nums)} steps")

        data_all = data_all

        print(f"Creating NetCDF file at: {nc_output_path}")

        # Create the NetCDF file
        with nc.Dataset(nc_output_path, 'w', format='NETCDF4') as ncfile:
            # Define dimensions
            ncfile.createDimension('time', len(timesteps))
            ncfile.createDimension('x', Nx)
            ncfile.createDimension('y', Ny)
            
            # Create variables
            time_var = ncfile.createVariable('time', 'f4', ('time',))
            omega_var = ncfile.createVariable('omega', 'f4', ('time', 'x', 'y'))
            x_var = ncfile.createVariable('x', 'f4', ('x',))
            y_var = ncfile.createVariable('y', 'f4', ('y',))
            
            # Add attributes for coordinates
            x_var.units = 'grid points'
            x_var.long_name = 'x-coordinate'
            y_var.units = 'grid points'
            y_var.long_name = 'y-coordinate'
            
            # Set coordinate values
            x_var[:] = np.arange(Nx)*dx
            y_var[:] = np.arange(Ny)*dy

            # Add attributes
            time_var.units = 'timestep'
            time_var.long_name = 'Simulation timestep'
            
            omega_var.units = 'vorticity'
            omega_var.long_name = 'Vorticity field'
            
            # Add data
            time_var[:] = timesteps
            # Move data to CPU and reshape to remove channel dimension
            omega_data = data_all[:, 0, :, :]
            omega_var[:] = omega_data
            
            # Add global attributes
            ncfile.reynolds_number = Re
            ncfile.description = 'Turbulence 2D simulation data'
            ncfile.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            ncfile.forcing_wavenumbers = (fkx, fky)

        print(f"NetCDF file created successfully with {len(timesteps)} timesteps")

save_dir = f"{py2d_sims_dir}/snapshots"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for Re, (fkx, fky), NX, dt, tTotal, tSave, beta in combinations:
    nc_name = f"turb2d_data_combined_Re-{int(Re)}_force_fkx-{fkx}_fky-{fky}_NX-{NX}_dt-{num2str(dt)}_tTotal-{num2str(tTotal)}_beta-{beta}.nc"
    nc_path = f"{py2d_sims_dir}/{nc_name}"

    nc_file = nc.Dataset(nc_path, 'r')

    fig, axs = plt.subplots(1, 4, figsize=(10, 3))

    ts = [0,1,-2,-1]
    for i, t in enumerate(ts):
        im = axs[i].imshow(nc_file["omega"][t], cmap='coolwarm', vmin=-10, vmax=10)
        axs[i].set_title(f"iter={t}, t={nc_file['time'][t]:.4f}")
    
    cbar = plt.colorbar(im, ax=axs[len(ts)-1], fraction=0.046, pad=0.04)
    cbar.set_label('Vorticity')

    plt.suptitle(f"{nc_name}\nfkx={fkx}, fky={fky}, NX={NX}, dt={dt}, tTotal={tTotal}, tSave={tSave}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{nc_name.replace('.nc', '.png')}", dpi=300)
    plt.close()


# mat_file_test = "/glade/derecho/scratch/asheshc/beta-channel-turbulence/data_lowres/16190.mat"

# # Load the mat file
# test_data = sio.loadmat(mat_file_test)
# test_omega = test_data['Omega']

# # Plot the test data
# plt.figure(figsize=(8, 6))
# plt.imshow(test_omega, cmap='coolwarm')
# plt.colorbar()
# plt.title('Test Data from MAT File')
# plt.savefig(f'{save_dir}/test_data_snapshot.png')
# plt.close()


for Re, (fkx, fky), NX, dt, tTotal, tSave, beta in combinations:
    nc_name = f"turb2d_data_combined_Re-{int(Re)}_force_fkx-{fkx}_fky-{fky}_NX-{NX}_dt-{num2str(dt)}_tTotal-{num2str(tTotal)}_beta-{beta}.nc"
    nc_path = f"{py2d_sims_dir}/{nc_name}"
    # Create a directory to store the animation frames
    import os

    anim_dir = f"{save_dir}/anim_Re-{Re}_force_fkx-{fkx}_fky-{fky}_NX-{NX}_dt-{num2str(dt)}_tTotal-{num2str(tTotal)}_beta-{beta}"
    os.makedirs(anim_dir, exist_ok=True)

    # Open the NetCDF file
    nc_file = nc.Dataset(nc_path, 'r')
    omega = nc_file["omega"][:]
    times = nc_file["time"][:]

    tindices = np.arange(0, omega.shape[0], step=40)
    # Save each frame as an image
    for it, t in enumerate(tindices):
        plt.figure(figsize=(6, 5))
        plt.imshow(omega[t], cmap='coolwarm', vmin=-10, vmax=10)
        plt.title(f"{nc_name}\ntimestep={t:06d}, t={times[t]:.2f}")
        plt.colorbar(label='Vorticity')
        plt.axis('off')
        frame_path = os.path.join(anim_dir, f"frame_{it:06d}.png")
        plt.savefig(frame_path, bbox_inches='tight', dpi=300)
        plt.close()

    # Use ffmpeg to create an animation from the frames
    # The output video will be saved in the save_dir
    video_path = f"{save_dir}/{nc_name.replace('.nc', '.mp4')}"
    # ffmpeg_cmd = f"ffmpeg -y -framerate 10 -i {anim_dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p -vf scale=trunc(iw/2)*2:trunc(ih/2)*2 {video_path}"
    # Create ffmpeg command to generate animation from frames
    # Use high quality settings with smooth playback
    ffmpeg_cmd = f"""ffmpeg -y -framerate 15 -i {anim_dir}/frame_%06d.png \
        -c:v libx264 -preset slow -crf 18 \
        -pix_fmt yuv420p \
        -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=15" \
        -movflags +faststart \
        {video_path}"""
    
    os.system(ffmpeg_cmd)