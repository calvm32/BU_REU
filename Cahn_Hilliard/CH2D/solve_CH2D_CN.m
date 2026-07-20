clc; clear; close all;

%% 2D Cahn-Hilliard with varying mu using Crank-Nicolson


%% Parameters
epsilon = 0.2; % used for scaling mu
blowup_time = epsilon^(-2/3);

t0 = 0.0;
T  = 5.0;

scale = 20;
Lx = scale*pi;
Ly = scale*pi;

Nx = 2^8;
Ny = 2^8;

dx = Lx/Nx;
dy = Ly/Ny;

plot_dt = 0.05; 

% Initial condition
sigma = 0.01;
u0 = sigma*randn(Nx,Ny);
u0_hat = fft2(u0);

% Discretization
x = (-Nx/2:Nx/2-1)*dx;
y = (-Ny/2:Ny/2-1)*dy;

dt = 0.1 * min(dx,dy)^2;
plot_every = round(plot_dt / dt);

% Fourier wavenumbers
kx = 2*pi * [0:Nx/2-1 -Nx/2:-1] / Lx;
ky = 2*pi * [0:Ny/2-1 -Ny/2:-1] / Ly;
[KX,KY] = ndgrid(kx,ky);
Laplacian_k = -(KX.^2 + KY.^2);

% Dealiasing mask
kx_max = max(abs(kx));
ky_max = max(abs(ky));
dealias_mask = (abs(KX) <= (2/3)*kx_max) & (abs(KY) <= (2/3)*ky_max);

%% Define the Evolution Problem
mu = @(t) t*epsilon;

% Linear operator is time-dependent
L_operator = @(t) -Laplacian_k.^2 - mu(t)*Laplacian_k;

% Nonlinear operator: N = Laplacian * dealias * fft(u^3)
nonlin_op = @(u_hat, t) Laplacian_k .* dealias_mask .* fft2( real(ifft2(u_hat)).^3 );

problem = EvolutionProblem(L_operator, nonlin_op, u0_hat, [t0, T]);

%% Setup the Solver
solver = CrankNicolsonSolver();

%% Setup the Monitor
params.mode = 'domain_wall_density';
params.x = x;
params.y = y;
params.scale = scale;
params.Lx = Lx;
params.Ly = Ly;
params.dx = dx;
params.dy = dy;
params.epsilon = epsilon;
params.blowup_time = blowup_time;
params.Laplacian_k = Laplacian_k;
params.plot_every = plot_every;
params.save_video = true;
params.video_filename = ['cahn_hilliard_2D_mu(1)=' num2str(mu(1),'%.2f') '.avi'];

monitor = CH2DMonitor(params);

%% Execute the Solve
disp('Starting integration (CH2D CN)...');
sol = evolution_solve(problem, solver, dt, save_every=plot_every, monitors=monitor);
disp(['Integration complete. Video saved to: ', params.video_filename]);
