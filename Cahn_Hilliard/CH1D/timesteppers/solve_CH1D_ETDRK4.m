clc; clear; close all;

%% 1D Cahn-Hilliard with varying mu using ETDRK4 time-stepping

%% Parameters & Setup
epsilon = 0.05;
blowup_time = epsilon^(-2/3);
t0 = -2.0;
T  = blowup_time + 100.0;
dt = 0.1;

scale = 100;
Lx = scale*pi;
Nx = 2^11;
dx = Lx/Nx;

plot_dt = 0.5; 
plot_every = round(plot_dt / dt);

% Domain and Fourier Setup
x = (-Nx/2:Nx/2-1)*dx;
kx = 2*pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
Laplacian_hat = -kx.^2;
L_operator = -(Laplacian_hat.^2);

kx_max = max(abs(kx));
dealias_mask = abs(kx) <= (2/3)*kx_max;

% Initial Conditions
sigma = 0.01;
u0_real = sigma*randn(1,Nx);
u0_hat = fft(u0_real);

%% Define the Evolution Problem
mu = @(t) t*epsilon;

% The nonlinear operator: Takes u_hat, returns Nu_hat
nonlin_op = @(u_hat, t) dealias_mask .* Laplacian_hat .* ...
                        fft( real(ifft(u_hat)).^3 - mu(t)*real(ifft(u_hat)) );
                    
problem = EvolutionProblem(L_operator, nonlin_op, u0_hat, [t0, T]);

%% Setup the Solver
solver = ETDRK4Solver(16); 

%% Execute the Solve with Monitor
params.epsilon = epsilon;
params.blowup_time = blowup_time;
params.Lx = Lx;
params.Nx = Nx;
params.dx = dx;
params.Laplacian_hat = Laplacian_hat;
params.plot_every = plot_every;
params.x = x;
params.kx = kx;
params.mu = mu;
params.save_video = true;
params.video_filename = sprintf('T=%.0f_epsilon=%.4f.avi', T, epsilon);

monitor = DensityL2FreeEnergyMonitor(params);

disp('Starting integration...');
sol = evolution_solve(problem, solver, dt, save_every=plot_every, monitors=monitor);
disp('Integration complete.');
