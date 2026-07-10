clc; clear; close all;

%% 
% This experiment runs through different cosine initial conditions 
% varying in mode. Some white noise is added.

%% Parameters
t0 = 0.0;
T  = 300;
dt = 0.1;

mass = 0;

xscale = 6;
Lx = xscale*pi;
Nx = 2^12;

[mu, ~] = critical_bifurcation(5, 2 * Lx, mass);

plot_dt = 4.0; 
plot_every = round(plot_dt / dt);
save_video = true;

% Find k_j's for initial data
k_j = pi * [0:7] / Lx;
k_i = k_j(2:8);

% discretization
dx = 2*Lx/Nx;
x = (-Nx/2:Nx/2-1)*dx;

% Fourier wavenumbers
kx = pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
Laplacian_hat = -kx.^2;
L_operator = -(Laplacian_hat.^2);

kx_max = max(abs(kx));
dealias_mask = abs(kx) <= (2/3)*kx_max;

%% Loop over all k_i
for m = 1:length(k_i)
    k = k_i(m);
    fprintf('Running k = %.4f\n', k);

    % Zero mean white noise
    noise = 0.0002 * (rand(1, Nx) - 0.5);
    noise = noise - mean(noise);
    
    % Start at k_bif
    u0 = 0.001*cos(k * x) + mass + noise;
    u0_hat = fft(u0);

    % Define problem
    nonlin_op = @(u_hat, t) dealias_mask .* Laplacian_hat .* fft( real(ifft(u_hat)).^3 - mu*real(ifft(u_hat)) );

    problem = EvolutionProblem(L_operator, nonlin_op, u0_hat, [t0, T]);

    % Solver
    solver = ETDRK4Solver(16);

    monitor = BifurcationMonitor(mu, kx, Lx, Nx, x, k_j, ...
        plot_every=plot_every, ...
        save_video=save_video, ...
        video_filename=sprintf('k=%.3f_T=%.0f.mp4', k, T), ...
        video_framerate=60);

    % Execute
    sol = evolution_solve(problem, solver, dt, save_every=plot_every, monitors=monitor);
    disp(['Integration complete for k = ', num2str(k)]);
end

function [mu_j, k_j] = critical_bifurcation(j, L, mass)
    % Computes the j'th frequencies critical mu and associated eigenvalue
    k_j = 2 * pi * j / L;
    mu_j = 3 * mass^2 + k_j^2;    
end