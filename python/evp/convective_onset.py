"""
Dedalus script for determining instability of static drizzle solutions to the Rainy-Benard system of equations.  This script computes curves of growth at discrete kx, scanning a variety of Rayleigh numbers.

Read more about these equations in:

Vallis, Parker & Tobias, 2019, JFM,
``A simple system for moist convection: the Rainy–Bénard model''

This script solves EVPs for an existing atmospheres, solved for by scripts in the nlbvp section.

Roberts, G.O., 1972,
``Dynamo action of fluid motions with two-dimensional periodicity''

Usage:
    convective_onset.py <cases>... [options]

Options:
    <cases>           Case (or cases) to plot results from

    --method=<m>      Method of onset searching [default: Rayleigh]
    --Rayleigh=<Ra>   Rayleigh number to test [default: 1e5]

    --nondim=<n>      Non-Nondimensionalization [default: buoyancy]

    --top-stress-free     Stress-free upper boundary
    --stress-free         Stress-free both boundaries

    --nz=<nz>         Number of coeffs to use in eigenvalue search; if not set, uses resolution of background
    --target=<targ>   Target value for sparse eigenvalue search [default: 0]
    --eigs=<eigs>     Target number of eigenvalues to search for [default: 20]

    --dense           Solve densely for all eigenvalues (slow)

    --verbose         Show plots on screen
"""
import logging
logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
     logging.getLogger(system).setLevel(logging.WARNING)

import numpy as np
import dedalus.public as de
import h5py

from docopt import docopt
args = docopt(__doc__)

method = args['--method']
N_evals = int(float(args['--eigs']))
target = float(args['--target'])

for case in args['<cases>']:
    f = h5py.File(case+'/drizzle_sol/drizzle_sol_s1.h5', 'r')
    sol = {}
    for task in f['tasks']:
        sol[task] = f['tasks'][task][0,0,0][:]
    sol['z'] = f['tasks']['b'].dims[3][0][:]
    tau_in = sol['tau'][0]
    k = sol['k'][0]
    α = sol['α'][0]
    β = sol['β'][0]
    γ = sol['γ'][0]
logger.info('α={:}, β={:}, γ={:}, tau={:}, k={:}'.format(α,β,γ,tau_in, k))
nz_sol = sol['z'].shape[0]
if args['--nz']:
    nz = int(float(args['--nz']))
else:
    nz = nz_sol

dealias = 3/2
dtype = np.complex128

Prandtlm = 1
Prandtl = 1
Rayleigh = float(args['--Rayleigh'])


Lz = 1
coords = de.CartesianCoordinates('x', 'y', 'z')
dist = de.Distributor(coords, dtype=dtype)
dealias = 2
zb = de.ChebyshevT(coords.coords[2], size=nz, bounds=(0, Lz), dealias=dealias)
z = zb.local_grid(1)

b0 = dist.Field(name='b0', bases=zb)
q0 = dist.Field(name='q0', bases=zb)

zb_sol = de.ChebyshevT(coords.coords[2], size=nz_sol, bounds=(0, Lz), dealias=dealias)
b0_sol = dist.Field(name='b0_sol', bases=zb_sol)
q0_sol = dist.Field(name='q0_sol', bases=zb_sol)

b0_sol['g'] = sol['b']
q0_sol['g'] = sol['q']

scale_ratio = nz/nz_sol
b0_sol.change_scales(scale_ratio)
q0_sol.change_scales(scale_ratio)

logger.info('rescaling background from {:} to {:} coeffs (ratio: {:})'.format(nz_sol, nz, scale_ratio))
b0['g'] = b0_sol['g']
q0['g'] = q0_sol['g']

p = dist.Field(name='p', bases=zb)
u = dist.VectorField(coords, name='u', bases=zb)
b = dist.Field(name='b', bases=zb)
q = dist.Field(name='q', bases=zb)

τp = dist.Field(name='τp')
τu1 = dist.VectorField(coords, name='τu1')
τu2 = dist.VectorField(coords, name='τu2')
τb1 = dist.Field(name='τb1')
τb2 = dist.Field(name='τb2')
τq1 = dist.Field(name='τq1')
τq2 = dist.Field(name='τq2')

zb1 = zb.clone_with(a=zb.a+1, b=zb.b+1)
zb2 = zb.clone_with(a=zb.a+2, b=zb.b+2)
lift1 = lambda A, n: de.Lift(A, zb1, n)
lift = lambda A, n: de.Lift(A, zb2, n)

ex, ey, ez = coords.unit_vector_fields(dist)

H = lambda A: 0.5*(1+np.tanh(k*A))

z_grid = dist.Field(name='z_grid', bases=zb)
z_grid['g'] = z

T0 = b0 - β*z_grid
qs0 = np.exp(α*T0)

tau = dist.Field(name='tau')
kx = dist.Field(name='kx')
Rayleigh = dist.Field(name='Ra_c')

# follows Roberts 1972 convention, eq 1.1, 2.8
dx = lambda A: 1j*kx*A
dy = lambda A: 0*A #1j*kx*A # try 2-d mode onset
dz = lambda A: de.Differentiate(A, coords['z'])

grad = lambda A: de.Gradient(A, coords)
div = lambda A:  dx(A@ex) + dy(A@ey) + dz(A@ez)
grad = lambda A: dx(A)*ex + dy(A)*ey + dz(A)*ez
lap = lambda A: dx(dx(A)) + dy(dy(A)) + dz(dz(A))
trans = lambda A: de.TransposeComponents(A)

e = grad(u) + trans(grad(u))
vars = [p, u, b, q, τp, τu1, τu2, τb1, τb2, τq1, τq2]
# fix Ra, find omega
dt = lambda A: ω*A
ω = dist.Field(name='ω')
problem = de.EVP(vars, eigenvalue=ω, namespace=locals())
#Ras = np.logspace(4,5,num=5)

nondim = args['--nondim']
if nondim == 'diffusion':
    P = 1                      #  diffusion on buoyancy. Always = 1 in this scaling.
    S = Prandtlm               #  diffusion on moisture  k_q / k_b
    PdR = Prandtl              #  diffusion on momentum
    PtR = Prandtl*Rayleigh     #  Prandtl times Rayleigh = buoyancy force
elif nondim == 'buoyancy':
    P = (Rayleigh * Prandtl)**(-1/2)         #  diffusion on buoyancy
    S = (Rayleigh * Prandtlm)**(-1/2)        #  diffusion on moisture
    PdR = (Prandtl / Rayleigh)**(1/2)        #  diffusion on momentum
    PtR = 1
    #tau_in /=                     # think through what this should be
else:
    raise ValueError('nondim {:} not in valid set [diffusion, buoyancy]'.format(nondim))

tau['g'] = tau_in

sech = lambda A: 1/np.cosh(A)
scrN = (H(q0 - qs0) + (q0 - qs0)*k/2*sech(k*(q0 - qs0))**2).evaluate()
scrN.name='scrN'

problem.add_equation('div(u) + τp + 1/PdR*dot(lift(τu2,-1),ez) = 0')
problem.add_equation('dt(u) - PdR*lap(u) + grad(p) - PtR*b*ez + lift(τu1, -1) + lift(τu2, -2) = 0')
problem.add_equation('dt(b) - P*lap(b) + dot(u, grad(b0)) - γ/tau*(q-α*qs0*b)*scrN + lift(τb1, -1) + lift(τb2, -2) = 0')
problem.add_equation('dt(q) - S*lap(q) + dot(u, grad(q0)) + 1/tau*(q-α*qs0*b)*scrN + lift(τq1, -1) + lift(τq2, -2) = 0')
problem.add_equation('b(z=0) = 0')
problem.add_equation('b(z=Lz) = 0')
problem.add_equation('q(z=0) = 0')
problem.add_equation('q(z=Lz) = 0')
if args['--stress-free']:
    problem.add_equation('ez@u(z=0) = 0')
    problem.add_equation('ez@(ex@e(z=0)) = 0')
    problem.add_equation('ez@(ey@e(z=0)) = 0')
else:
    problem.add_equation('u(z=0) = 0')
if args['--top-stress-free'] or args['--stress-free']:
    problem.add_equation('ez@u(z=Lz) = 0')
    problem.add_equation('ez@(ex@e(z=Lz)) = 0')
    problem.add_equation('ez@(ey@e(z=Lz)) = 0')
else:
    problem.add_equation('u(z=Lz) = 0')
problem.add_equation('integ(p) = 0')
solver = problem.build_solver()

dlog = logging.getLogger('subsystems')
dlog.setLevel(logging.WARNING)

# fix Ra, find omega
def compute_growth_rate(kx_i, Ra_i):
    kx['g'] = kx_i
    Rayleigh['g'] = Ra_i

    if args['--dense']:
        solver.solve_dense(solver.subproblems[0], rebuild_matrices=True)
    else:
        solver.solve_sparse(solver.subproblems[0], N=N_evals, target=target, rebuild_matrices=True)
    i_evals = np.argsort(solver.eigenvalues.real)
    evals = solver.eigenvalues[i_evals]
    peak_eval = evals[-1]
    # choose convention: return the positive complex mode of the pair
    if peak_eval.imag < 0:
        peak_eval = np.conj(peak_eval)
    return peak_eval

def peak_growth_rate(*args):
    rate = compute_growth_rate(*args)
    # flip sign so minimize finds maximum
    return -1*rate.real


growth_rates = {}
Ras = np.geomspace(1e4,1e5,num=10)
Ras = np.geomspace(2.5e4,1e5,num=10)
kxs = np.logspace(-1, 1, num=40)
print(Ras)
for Ra_i in Ras:
    σ = []
    for kx_i in kxs:
        σ_i = compute_growth_rate(kx_i, Ra_i)
        σ.append(σ_i)
        logger.info('Ra = {:.2g}, kx = {:.2g}, σ = {:.2g}'.format(Ra_i, kx_i, σ_i))
    growth_rates[Ra_i] = np.array(σ)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
peak_σ = -np.inf

if nondim == 'diffusion':
    ax2 = ax.twinx()
    ax.set_ylim(-15, 25)
    ax2.set_ylim(1e-1, 1e3)
    ax2.set_yscale('log')
    ax.set_ylabel(r'$\omega_R$ (solid)')
    ax2.set_ylabel(r'$\omega_I$ (dashed)')
elif nondim == 'buoyancy':
    ax2 = ax
    ax.set_ylim(-0.1, 0.5)
    ax.set_ylabel(r'$\omega_R$ (solid), $\omega_I$ (dashed)')

for Ra in growth_rates:
    σ = growth_rates[Ra]
    peak_σ = max(peak_σ, np.max(σ))
    p = ax.plot(kxs, σ.real, label='Ra = {:.2g}'.format(Ra))
    ax2.plot(kxs, σ.imag, linestyle='dashed', color=p[0].get_color())
ax.set_xscale('log')

fig_filename = 'growth_curves_{:}'.format(nondim)
if args['--stress-free']:
    fig_filename += '_SF'
if args['--top-stress-free']:
    fig_filename += '_TSF'
ax.legend()
ax.axhline(y=0, linestyle='dashed', color='xkcd:grey', alpha=0.5)
ax.set_title(r'$\gamma$ = {:}, $\beta$ = {:}, $\tau$ = {:}'.format(γ,β,tau['g'][0,0,0]))
ax.set_xlabel('$k_x$')
ax.set_title('{:} timescales'.format(nondim))
fig.savefig(case+'/'+fig_filename+'.png', dpi=300)



import scipy.optimize as sciop
bounds = sciop.Bounds(lb=1, ub=10)

peaks = {'σ':[], 'k':[], 'Ra':[]}
for Ra in growth_rates:
    σ = growth_rates[Ra]
    peak_i = np.argmax(σ)
    kx_i = kxs[peak_i]
    result = sciop.minimize(peak_growth_rate, kx_i, args=(Ra), bounds=bounds, method='Nelder-Mead', tol=1e-5)
    # obtain full complex rate
    σ = compute_growth_rate(result.x[0], Ra)
    logger.info('peak search: start at Ra = {:.4g}, kx = {:.4g}, found σ_max = {:.2g},{:.2g}i, kx = {:.4g}'.format(Ra, kx_i, σ.real, σ.imag, result.x[0]))
    peaks['σ'].append(σ)
    peaks['k'].append(result.x[0])
    peaks['Ra'].append(Ra)

peaks['σ'] = np.array(peaks['σ'])
peaks['k'] = np.array(peaks['k'])
peaks['Ra'] = np.array(peaks['Ra'])

from scipy.interpolate import interp1d
f_σR_i = interp1d(peaks['σ'].real, peaks['k']) #inverse
f_σR = interp1d(peaks['k'], peaks['σ'].real)
f_σI_i = interp1d(peaks['σ'].imag, peaks['k']) #inverse
f_σI = interp1d(peaks['k'], peaks['σ'].imag)

# to find critical Ra
f_σR_Ra_i = interp1d(peaks['σ'].real, peaks['Ra'])
f_σR_Ra = interp1d(peaks['Ra'], peaks['σ'].real)
f_σI_Ra = interp1d(peaks['Ra'], peaks['σ'].imag)
f_k_Ra = interp1d(peaks['Ra'], peaks['k'])

peak_ks = np.geomspace(np.min(peaks['k']), np.max(peaks['k']))

crit_Ra = f_σR_Ra_i(0)
crit_k = f_k_Ra(crit_Ra)
crit_σ_R = f_σR_Ra(crit_Ra)
crit_σ_I = f_σI_Ra(crit_Ra)
logger.info('Critical point, based on interpolation:')
logger.info('Ra = {:}, k = {:}'.format(crit_Ra, crit_k))
logger.info('σ = {:}, {:}i'.format(crit_σ_R, crit_σ_I))

p = ax.plot(peak_ks, f_σR(peak_ks), linestyle='dotted', color='xkcd:grey')

fig_filename = 'growth_curves_peaks_{:}'.format(nondim)
if args['--stress-free']:
    fig_filename += '_SF'
if args['--top-stress-free']:
    fig_filename += '_TSF'
fig.savefig(case+'/'+fig_filename+'.png', dpi=300)
