clc; clear; close all;

%% Parameters
epsilon = 0.05; % used for scaling mu
blowup_time = epsilon^(-2/3);

t0 = -2.0;
T  = blowup_time + 200.0;
dt = 0.01; 

scale = 100;
Lx = scale*pi;
Nx = 2^11;
dx = Lx/Nx;

plot_dt = 0.5; 
plot_every = round(plot_dt / dt);

% Initial condition
sigma = 0.01;
u0 = sigma*randn(1,Nx);
u0_hat = fft(u0);

% discretization
x = (-Nx/2:Nx/2-1)*dx;

% Fourier wavenumbers
kx = 2*pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
Laplacian_k = -(kx.^2);

% dealiasing mask
kx_max = max(abs(kx));
dealias_mask = abs(kx) <= (2/3)*kx_max;

%% 1. Define the Evolution Problem
mu = @(t) t*epsilon;

% Dummy linear operator, solver does the stabilized IMEX division internally
L_operator = Laplacian_k.^2 - 2.0*Laplacian_k; % (using S=2.0)

% The nonlinear operator for GradStableIMEXSolver returns dealiased u^3 in Fourier space
nonlin_op = @(u_hat, t) dealias_mask .* fft( real(ifft(u_hat)).^3 );

problem = EvolutionProblem(L_operator, nonlin_op, u0_hat, [t0, T]);

%% 2. Setup the Solver
S = 2.0;
solver = GradStableIMEXSolver(S, Laplacian_k, mu);

%% 3. Setup the Monitor
params.epsilon = epsilon;
params.blowup_time = blowup_time;
params.Lx = Lx;
params.Nx = Nx;
params.dx = dx;
params.Laplacian_hat = Laplacian_k;
params.plot_every = plot_every;
params.x = x;
params.kx = kx;
params.mu = mu;
params.save_video = true;
params.video_filename = ['cahn_hilliard_1D_mu(1)=' num2str(mu(1),'%.2f') '.avi'];

monitor = DensityL2FreeEnergyMonitor(params);

%% 4. Execute the Solve
disp('Starting integration (CH1D Grad Stable)...');
sol = evolution_solve(problem, solver, dt, save_every=plot_every, monitors=monitor);
disp(['Integration complete. Video saved to: ', params.video_filename]);