clc; clear; close all;
image_mode = "count_droplets"; % "count_droplets" or "count_minmax"

%% Parameters
mu = @(t) 3;
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

% discretization
x = (-Nx/2:Nx/2-1)*dx;
y = (-Ny/2:Ny/2-1)*dy;

dt = 0.1 * min(dx,dy)^2;
plot_every = round(plot_dt / dt);

% Fourier wavenumbers
kx = 2*pi * [0:Nx/2-1 -Nx/2:-1] / Lx;
ky = 2*pi * [0:Ny/2-1 -Ny/2:-1] / Ly;
[KX,KY] = ndgrid(kx,ky);
Laplacian_k = -(KX.^2 + KY.^2);

% dealiasing mask
kx_max = max(abs(kx));
ky_max = max(abs(ky));
dealias_mask = (abs(KX) <= (2/3)*kx_max) & (abs(KY) <= (2/3)*ky_max);

%% 1. Define the Evolution Problem
% Linear operator: L = -Laplacian^2 - mu*Laplacian
% Since mu is constant, L is constant
L_operator = -Laplacian_k.^2 - mu(0)*Laplacian_k;

% Nonlinear operator: N = Laplacian * dealias * fft(u^3)
nonlin_op = @(u_hat, t) Laplacian_k .* dealias_mask .* fft2( real(ifft2(u_hat)).^3 );

problem = EvolutionProblem(L_operator, nonlin_op, u0_hat, [t0, T]);

%% 2. Setup the Solver
solver = CrankNicolsonSolver();

%% 3. Setup the Monitor
params.mode = image_mode;
params.x = x;
params.y = y;
params.scale = scale;
params.Lx = Lx;
params.Ly = Ly;
params.dx = dx;
params.dy = dy;
params.plot_every = plot_every;
params.save_video = true;
params.video_filename = ['cahn_hilliard_2D_muIC=' num2str(mu(0),'%.2f') '.mp4'];

monitor = CH2DMonitor(params);

%% 4. Execute the Solve
disp('Starting integration (CH2D Phase Separation)...');
sol = evolution_solve(problem, solver, dt, save_every=plot_every, monitors=monitor);
disp(['Integration complete. Video saved to: ', params.video_filename]);
