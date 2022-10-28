"""
Dedalus script for Rainy Benard Aggregation study

This script is currently parallelized using a 1D mesh decomposition, so that the 
maximum number of processors is set by the direction with the lowest resolution.
A 2D mesh decomposition can be used to run the problem with more processors.  To
do this, you must specify mesh=[N1,N2] in the domain creation, where N1*N2 is the
total number of processors.  In grid space, each processor has all the x data,
and a fraction of the y data (N1/Ny), and a fraction of the z data (N2/Nz).

Default paramters from Vallis, Parker, and Tobias (2018)
http://empslocal.ex.ac.uk/people/staff/gv219/papers/VPT_convection18.pdf

Usage:
    rot_rainy_benard.py [--beta=<beta> --Rayleigh=<Rayleigh> --Prandtl=<Prandtl> --Prandtlm=<Prandtlm>  --Taylor=<Taylor> --theta=<theta> --F=<F> --alpha=<alpha> --gamma=<gamma> --DeltaT=<DeltaT> --sigma2=<sigma2> --q0=<q0> --nx=<nx> --ny=<ny> --nz=<nz> --Lx=<Lx> --Ly=<Ly> --Lz=<Lz> --restart=<restart_file> --filter=<filter> --mesh=<mesh> --nondim=<nondim> --wall_time=<wall_time> --label=<label>] 

Options:
    --Rayleigh=<Rayleigh>    Rayleigh number [default: 1e6]
    --Prandtl=<Prandtl>      Prandtl number [default: 1]
    --Prandtlm=<Prandtlm>    moist Prandtl number [default: 1]
    --F=<F>                  basic state buoyancy difference [default: 0]
    --alpha=<alpha>          Clausius Clapeyron parameter [default: 3.0]
    --beta=<beta>            beta parameter [default: 1.20]
    --gamma=<gamma>          condensational heating parameter [default: 0.19]
    --DeltaT=<DeltaT>        Temperature at top [default: -1.0]
    --sigma2=<sigma2>        Initial condition sigma2 [default: 0.05]
    --q0=<q0>                Initial condition q0 [default: 5.]
    --nx=<nx>                x (Fourier) resolution [default: 256]
    --ny=<ny>                y (Fourier) resolution [default: 256]
    --nz=<nz>                vertical z (Chebyshev) resolution [default: 64]
    --Lx=<Lx>                x length  [default: 10.]
    --Ly=<Ly>                y length [default: 10.]
    --Lz=<Lz>                vertical z length [default: 1.]
    --restart=<restart_file> Restart from checkpoint
    --nondim=<nondim>        nondimensionalization (buoyancy or RB) [default: buoyancy]
    --filter=<filter>        fraction of modes to keep in ICs [default: 0.5]
    --mesh=<mesh>            processor mesh (you're in charge of making this consistent with nproc) [default: None]
    --Taylor=<Taylor>        Taylor number [default: 1e2]
    --theta=<theta>          angle between gravity and rotation vectors [default: 0]
    --wall_time=<wall_time>  wall time (in hours) [default: 23.5]
    --label=<label>          optional output directrory label [default: None]

"""
from docopt import docopt
import os
import sys
import numpy as np
import time

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post

import logging
logger = logging.getLogger(__name__)

args = docopt(__doc__)
# Parameters
Lx = float(args['--Lx'])
Ly = float(args['--Ly'])
Lz = float(args['--Lz'])
nx = int(args['--nx'])
ny = int(args['--ny'])
nz = int(args['--nz'])
mesh = args['--mesh']
wall_time = float(args['--wall_time'])
if mesh == 'None':
    mesh = None
else:
    mesh = [int(i) for i in mesh.split(',')]
if ny == 0:
    threeD = False
    Ly = 0. # override Ly if 2D
else:
    threeD = True

betaval = float(args['--beta'])
Rayleigh = float(args['--Rayleigh'])    # The real Rayleigh number is this number times a buoyancy difference
Prandtl = float(args['--Prandtl'])
Prandtlm = float(args['--Prandtlm'])
Fval = float(args['--F'])
alphaval = float(args['--alpha'])
gammaval = float(args['--gamma'])
DeltaTval = float(args['--DeltaT'])
Taylor = float(args['--Taylor'])
theta = float(args['--theta'])


# initial conditions
#sigma2 = 0.005
sigma2 = float(args['--sigma2'])
q0_amplitude = float(args['--q0'])

# Nondimensionalization
nondim = args['--nondim']

if nondim == 'RB':
    #  RB diffusive scaling
    Pval = 1                      #  diffusion on buoyancy. Always = 1 in this scaling.
    Sval = 1                      #  diffusion on moisture  k_q / k_b
    PdRval = Prandtl              #  diffusion on momentum
    PtRval = Prandtl * Rayleigh   #  Prandtl times Rayleigh = buoyancy force
    t_therm = 1/Pval              #  thermal diffusion timescale, always = 1 in this scaling
    tauval   = 5e-5*t_therm     #  condensation time scale
    #tauval = 3/8*t_therm
    
    # Rotation Stuff
    omegaval = 1/2*Taylor**(1/2)*PdRval
    omega_xval = 0
    omega_yval = omegaval * np.sin(theta)
    omega_zval = omegaval * np.cos(theta)

    slices_dt = 0.01
    snap_dt = 1.0
    prof_dt = 0.01
    ts_dt = 0.001

elif nondim == 'buoyancy':                                           #  Buoyancy scaling
    Pval = (Rayleigh * Prandtl)**(-1/2)         #  diffusion on buoyancy
    Sval = (Rayleigh * Prandtlm)**(-1/2)        #  diffusion on moisture
    PdRval = (Prandtl / Rayleigh)**(1/2)        #  diffusion on momentum
    PtRval = 1                                  #  buoyancy force  = 1 always
    t_therm = 1/Pval                            #  thermal diffusion timescale
    tauval   = 5e-5*t_therm                     #  condensation timescale
    #tauval = 3/8 *t_therm
    
    #Rotation stuff
    omegaval = 1/2*Taylor**(1/2)*PdRval
    omega_xval = 0
    omega_yval = omegaval * np.sin(theta)
    omega_zval = omegaval * np.cos(theta)
    
    
    slices_dt = 0.001*(Rayleigh* Prandtl)**(1/2)
    snap_dt = 1.0*(Rayleigh* Prandtl)**(1/2)
    prof_dt = 0.01*(Rayleigh* Prandtl)**(1/2)
    ts_dt = 0.001*(Rayleigh* Prandtl)**(1/2)
else:
    raise ValueError("Nondimensionalization {} not supported.".format(nondim))
logger.info("Output timescales (in sim time): slices = {}, snapshots = {}, profiles ={}, timeseries = {}".format(slices_dt, snap_dt, prof_dt, ts_dt))
logger.info("Rotation rate (in 1/sim time), and angle of rotation: omega={}, theta={}".format(omegaval, theta))
# Create bases and domain
bases = []
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
bases.append(x_basis)
if threeD:
    y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
    bases.append(y_basis)
z_basis = de.Chebyshev('z', nz, interval=(0, Lz), dealias=3/2)
bases.append(z_basis)
domain = de.Domain(bases, grid_dtype=np.float64, mesh=mesh)
# 3D Boussinesq hydrodynamics
variables = ['p','b','u','w','bz','uz','wz','temp','q','qz']
if threeD:
    variables += ['v', 'vz']

problem = de.IVP(domain,
                 variables=variables)

# save data in directory named after script
logger.info(Rayleigh, betaval, Prandtl, Prandtlm, Taylor, theta, Fval, alphaval, gammaval, DeltaTval, sigma2, q0_amplitude, nondim, nx, ny, nz, Lx, Ly, Lz)
data_dir = "scratch/" + sys.argv[0].split('.py')[0]
data_dir +="_Ra{:.2e}_gamma{:.2f}_beta{:.2f}_Ta{:.2e}_theta{:.2f}_{}_{}x{}x{}".format(Rayleigh, gammaval, betaval, Taylor, theta, nondim, nx, ny, nz)
#data_dir +="_Ra{0:5.02e}_beta{1:5.02e}_Pr{2:5.02e}_Prm{3:5.02e}_Ta{0:5.02e}_theta{4:5.02e}_F{4:5.02e}_alpha{5:5.02e}_gamma{6:5.02e}_DeltaT{7:5.02e}_sigma2{8:5.02e}_q0{9:5.02e}_nondim:{10:s}_nx{11:d}_ny{12:d}_nz{13:d}_Lx{14:5.02e}_Ly{15:5.02e}_Lz{16:5.02e}".format(Rayleigh, betaval, Prandtl, Prandtlm, Taylor, theta, Fval, alphaval, gammaval, DeltaTval, sigma2, q0_amplitude, nondim, nx, ny, nz, Lx, Ly, Lz)
if args['--label'] is not None:
    data_dir += "_{}".format(args['--label'])
data_dir += '/'
logger.info("saving run in: {}".format(data_dir))

if domain.distributor.rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.makedirs('{:s}/'.format(data_dir))

problem.parameters['P'] = Pval
problem.parameters['PdR'] = PdRval
problem.parameters['PtR'] = PtRval
problem.parameters['gamma'] = gammaval
problem.parameters['S'] = Sval
problem.parameters['beta'] = betaval
problem.parameters['tau'] = tauval
problem.parameters['alpha'] = alphaval
problem.parameters['DeltaT'] = DeltaTval
problem.parameters['omega_x'] = omega_xval
problem.parameters['omega_y'] = omega_yval
problem.parameters['omega_z'] = omega_zval
problem.parameters['omega'] = omegaval


# numerics parameters
problem.parameters['k'] = 1e5 # cutoff for tanh
problem.parameters['Lx'] = Lx
problem.parameters['Lz'] = Lz
if threeD:
    problem.parameters['Ly'] = Ly

if threeD:
    problem.substitutions['plane_avg(A)'] = 'integ(A, "x", "y")/Lx/Ly'
    problem.substitutions['vol_avg(A)'] = 'integ(A)/Lx/Ly/Lz'
    problem.substitutions['KE'] = '0.5*(u*u + v*v + w*w)'
else:
    problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
    problem.substitutions['vol_avg(A)'] = 'integ(A)/Lx/Lz'
    problem.substitutions['KE'] = '0.5*(u*u + w*w)'

if threeD:
    problem.substitutions['Coriolis_x'] = '(2*omega_y*w - 2*omega_z*v)'
    problem.substitutions['Coriolis_y'] = '(2*omega_z*u - 2*omega_x*w)'
    problem.substitutions['Coriolis_z'] = '(2*omega_x*v - 2*omega_y*u)'

problem.substitutions['vorticity_y'] = '( uz  - dx(w))'        
if threeD:
    problem.substitutions['vorticity_x'] = '(dy(w) - vz)'        
    problem.substitutions['vorticity_z'] = '(dx(v) - dy(u))'
    problem.substitutions['enstrophy']   = '(vorticity_x**2 + vorticity_y**2 + vorticity_z**2)'
    problem.substitutions['Rossby'] = '(sqrt(enstrophy)/(2*omega))'



problem.substitutions['H(A)'] = '0.5*(1. + tanh(k*A))'
problem.substitutions['qs'] = 'exp(alpha*temp)'
problem.substitutions['rh'] = 'q/exp(alpha*temp)'


if threeD:
    problem.add_equation('dx(u) + dy(v) + wz = 0')

    problem.add_equation('dt(b) - P*(dx(dx(b)) + dy(dy(b)) + dz(bz)) = - u*dx(b) - v*dy(b) - w*bz + gamma*H(q - qs)*(q - qs)/tau')
    problem.add_equation('dt(q) - S*(dx(dx(q)) + dy(dy(q)) + dz(qz)) = - u*dx(q) - v*dy(q) - w*qz +   H(q - qs)*(qs - q)/tau')

    problem.add_equation('dt(u) - PdR*(dx(dx(u)) + dy(dy(u)) + dz(uz)) + dx(p) + Coriolis_x         = - u*dx(u) - v*dy(u) - w*uz')
    problem.add_equation('dt(v) - PdR*(dx(dx(v)) + dy(dy(v)) + dz(vz)) + dy(p) + Coriolis_y         = - u*dx(v) - v*dy(v) - w*vz')
    problem.add_equation('dt(w) - PdR*(dx(dx(w)) + dy(dy(w)) + dz(wz)) + dz(p) - PtR*b + Coriolis_z = - u*dx(w) - v*dy(w) - w*wz')
else:
    problem.add_equation('dx(u) + wz = 0')

    problem.add_equation('dt(b) - P*(dx(dx(b)) + dz(bz)) = - u*dx(b) - w*bz + gamma*H(q - qs)*(q - qs)/tau')
    problem.add_equation('dt(q) - S*(dx(dx(q)) + dz(qz)) = - u*dx(q) - w*qz +   H(q - qs)*(qs - q)/tau')

    problem.add_equation('dt(u) - PdR*(dx(dx(u)) + dz(uz)) + dx(p)                = - u*dx(u) - w*uz')
    problem.add_equation('dt(w) - PdR*(dx(dx(w)) + dz(wz)) + dz(p) - PtR*b        = - u*dx(w) - w*wz')
    

problem.add_equation('bz - dz(b) = 0')
problem.add_equation('qz - dz(q) = 0')
problem.add_equation('uz - dz(u) = 0')
if threeD:
    problem.add_equation('vz - dz(v) = 0')
problem.add_equation('wz - dz(w) = 0')
problem.add_equation('dz(temp) - bz = -beta')

problem.add_bc('left(b) = 0')
problem.add_bc('right(b) = beta + DeltaT')
problem.add_bc('left(q) = 1')
problem.add_bc('right(q) = exp(alpha*DeltaT)')
problem.add_bc('left(u) = 0')
problem.add_bc('right(u) = 0')
if threeD:
    problem.add_bc('left(v) = 0')
    problem.add_bc('right(v) = 0')
    
problem.add_bc('left(w) = 0')
problem.add_bc('left(temp) = 0')

if threeD:
    cond1 = 'nx != 0 or ny != 0'
    cond2 = 'nx == 0 and ny == 0'
else:
    cond1 = 'nx != 0'
    cond2 = 'nx == 0'
problem.add_bc('right(w) = 0', condition=cond1)
problem.add_bc('right(p) = 0', condition=cond2)

# Build solver
ts = de.timesteppers.SBDF3
#ts = de.timesteppers.RK443
solver = problem.build_solver(ts)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
if threeD:
    y = domain.grid(1)
    z = domain.grid(2)
else:
    z = domain.grid(1)
b = solver.state['b']
bz = solver.state['bz']
q = solver.state['q']
qz = solver.state['qz']

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval

gshape = problem.domain.dist.grid_layout.global_shape(scales=problem.domain.dealias)
slices = problem.domain.dist.grid_layout.slices(scales=problem.domain.dealias)
rand = np.random.RandomState(seed=42)
pert = rand.standard_normal(gshape)[slices]

#b['g'] = -0.0*(z - pert)
#b['g'] = 0#T1ovDTval-(1.00-betaval)*z
b['g'] = (betaval + DeltaTval) * z / Lz
b.differentiate('z', out=bz)
#q['g'] = q0val*np.exp(-betaval*z/T0val)+1e-2*np.exp(-((x-1.0)/0.01)^2)*np.exp(-((z-0.5)/0.01)^2)
#q['g'] = q0val*np.exp(-betaval*z/T0val)+(1e-2)*np.exp(-((z-0.5)*(z-0.5)/0.02))*np.exp(-((x-1.0)*(x-1.0)/0.02))

#q = qs = np.exp(alpha*T) = np.exp(alpha * (b - beta z))
#q += q_pert
#q *= envelope = np.sin(pi*z/Lz)
q['g'] = 1.0*np.exp(alphaval * (DeltaTval * z))
#q['g'] = q0_amplitude*np.exp(-((z-0.1)*(z-0.1)/sigma2))*np.exp(-((x-1.0)*(x-1.0)/sigma2))
if threeD:
    q['g'] += q0_amplitude*np.exp(-((z-0.1)*(z-0.1)/sigma2))*np.exp(-((x-1.0)*(x-1.0)/sigma2))*np.exp(-((y-1.0)*(y-1.0)/sigma2))*np.sin(np.pi * z/Lz)
else:
     q['g'] += q0_amplitude*np.exp(-((z-0.1)*(z-0.1)/sigma2))*np.exp(-((x-1.0)*(x-1.0)/sigma2))*np.sin(np.pi * z/Lz)

q.differentiate('z', out=qz)

# Integration parameters
dt = 1e-7

#solver.stop_sim_time = 2000
solver.stop_sim_time = np.inf
solver.stop_wall_time = wall_time * 3600.
solver.stop_iteration = np.inf

hermitian_cadence = 10

# CFL routines
logger.info("Starting CFL")
tausafety= 0.1

CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=5, safety=0.30,
                     max_change=1.5, max_dt=tauval*tausafety)#, min_change=0.5)
cfl_vels = ['u','w']
if threeD:
    cfl_vels.append('v')
CFL.add_velocities(cfl_vels)

# Analysis
analysis_tasks = []
slices = solver.evaluator.add_file_handler(os.path.join(data_dir, 'slices'), sim_dt=slices_dt, max_writes=50)
if threeD:
    slices.add_task('interp(b, z = 0.5)', name='b midplane')
    slices.add_task('interp(u, z = 0.5)', name='u midplane')
    slices.add_task('interp(v, z = 0.5)', name='v midplane')
    slices.add_task('interp(w, z = 0.5)', name='w midplane')
    slices.add_task('interp(temp, z = 0.5)', name='temp midplane')
    slices.add_task('interp(q, z = 0.5)', name='q midplane')
    slices.add_task('interp(rh, z = 0.5)', name='rh midplane')
    slices.add_task('interp(vorticity_x, z=0.5)', name='x vorticity midplane')
    slices.add_task('interp(vorticity_y, z=0.5)', name='y vorticity midplane')
    slices.add_task('interp(vorticity_z, z=0.5)', name='z vorticity midplane')

    slices.add_task('interp(b, z = 0.8)', name='b z.8')
    slices.add_task('interp(u, z = 0.8)', name='u z.8')
    slices.add_task('interp(v, z = 0.8)', name='v z.8')
    slices.add_task('interp(w, z = 0.8)', name='w z.8')
    slices.add_task('interp(temp, z = 0.8)', name='temp z.8')
    slices.add_task('interp(q, z = 0.8)', name='q z.8')
    slices.add_task('interp(rh, z = 0.8)', name='rh z.8')
    slices.add_task('interp(vorticity_x, z=0.8)', name='x vorticity z.8')
    slices.add_task('interp(vorticity_y, z=0.8)', name='y vorticity z.8')
    slices.add_task('interp(vorticity_z, z=0.8)', name='z vorticity z.8')

    
    slices.add_task('interp(b, z = 0.9)', name='b z.9')
    slices.add_task('interp(u, z = 0.9)', name='u z.9')
    slices.add_task('interp(v, z = 0.9)', name='v z.9')
    slices.add_task('interp(w, z = 0.9)', name='w z.9')
    slices.add_task('interp(temp, z = 0.9)', name='temp z.9')
    slices.add_task('interp(q, z = 0.9)', name='q z.9')
    slices.add_task('interp(rh, z = 0.9)', name='rh z.9')
    slices.add_task('interp(vorticity_x, z=0.9)', name='x vorticity z.9')
    slices.add_task('interp(vorticity_y, z=0.9)', name='y vorticity z.9')
    slices.add_task('interp(vorticity_z, z=0.9)', name='z vorticity z.9')

    
    
    slices.add_task('interp(b, x = 0)', name='b vertical')
    slices.add_task('interp(u, x = 0)', name='u vertical')
    slices.add_task('interp(v, x = 0)', name='v vertical')
    slices.add_task('interp(w, x = 0)', name='w vertical')
    slices.add_task('interp(temp, x = 0)', name='temp vertical')
    slices.add_task('interp(q, x = 0)', name='q vertical')
    slices.add_task('interp(rh, x = 0)', name='rh vertical')
    slices.add_task('interp(vorticity_x, x=0)', name='x vorticity vertical')
    slices.add_task('interp(vorticity_y, x=0)', name='y vorticity vertical')
    slices.add_task('interp(vorticity_z, x=0)', name='z vorticity vertical')

else:
    slices.add_task('b', name='b vertical')
    slices.add_task('u', name='u vertical')
    slices.add_task('w', name='w vertical')
    slices.add_task('temp', name='temp vertical')
    slices.add_task('q', name='q vertical')
    slices.add_task('rh', name='rh vertical')
analysis_tasks.append(slices)

snapshots = solver.evaluator.add_file_handler(os.path.join(data_dir, 'snapshots'), sim_dt=snap_dt, max_writes=10)
snapshots.add_system(solver.state)
analysis_tasks.append(snapshots)
profiles = solver.evaluator.add_file_handler(os.path.join(data_dir, 'profiles'), sim_dt=prof_dt)
profiles.add_task('plane_avg(b)', name='b')
profiles.add_task('plane_avg(u)', name='u')
if threeD:
    profiles.add_task('plane_avg(v)', name='v')
profiles.add_task('plane_avg(w)', name='w')
profiles.add_task('plane_avg(q)', name='q')
profiles.add_task('plane_avg(rh)', name='rh')
profiles.add_task('plane_avg(temp)', name='temp')
analysis_tasks.append(profiles)
timeseries = solver.evaluator.add_file_handler(os.path.join(data_dir, 'timeseries'), sim_dt=ts_dt)
timeseries.add_task('vol_avg(KE)', name='KE')
if threeD and Taylor > 0:
    timeseries.add_task('vol_avg(Rossby)', name='Rossby')
analysis_tasks.append(timeseries)

mode='append'
checkpoint_min = 60
checkpoint = solver.evaluator.add_file_handler(data_dir+'checkpoint', wall_dt=checkpoint_min*60, sim_dt=np.inf, iter=np.inf, max_writes=1, mode=mode)
checkpoint.add_system(solver.state, layout = 'c')
analysis_tasks.append(checkpoint)

# Main loop
dt = CFL.compute_dt()

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("KE", name='KE')

try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        solver.step(dt)
        if (solver.iteration - 1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e, max E_kin: %e' %(solver.iteration, solver.sim_time, dt, flow.max('KE')))

        if (solver.iteration - 1) % hermitian_cadence == 0:
            for field in solver.state.fields:
                field.require_grid_space()
        dt = CFL.compute_dt()

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.evaluate_handlers_now(dt)

    end_time = time.time()

    # Print statistics
    logger.info('Run time: %f' %(end_time-start_time))
    logger.info('Iterations: %i' %solver.iteration)

    for task in analysis_tasks:
        logger.info(task.base_path)
        post.merge_analysis(task.base_path)


