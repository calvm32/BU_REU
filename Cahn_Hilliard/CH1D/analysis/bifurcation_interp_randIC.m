clc; clear; close all;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Description
% This experiment runs through different constant mu 
% between two bifurcation nodes. There was found to be 
% little qualitative difference, with initial conditions 
% more of an effect on the end dominant mode.

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Parameters
t0 = 0.0;
T  = 400;
dt = 0.1;

mass = 0;

xscale = 6;
Lx = xscale*pi;
Nx = 2^12;

[mu4, ~] = critical_bifurcation(4, 2 * Lx, mass);
[mu5, ~] = critical_bifurcation(5, 2 * Lx, mass);

alpha = [0.001, 0.01, 0.1, 0.5, .9, .99, .999];

% Find k_j for graphing purposes
k_j = pi * [0:6] / Lx;

plot_dt = 0.5; 
plot_every = round(plot_dt / dt);
save_video = false;

% spatial grid
dx = 2*Lx/Nx;
x = (-Nx/2:Nx/2-1)*dx;

% Zero mean white noise
noise = 0.0002 * (rand(1, Nx) - 0.5);
noise = noise - mean(noise);
u0 = mass + noise;
u0_hat = fft(u0);

% Fourier wavenumbers
kx = pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
Laplacian_hat = -kx.^2;
L_operator = -(Laplacian_hat.^2);

kx_max = max(abs(kx));
dealias_mask = abs(kx) <= (2/3)*kx_max;

%% Loop over all alphas
for m = 1:length(alpha)
    mu_val = (1 - alpha(m)) * mu4 + alpha(m)*mu5;
    fprintf('Running alpha = %.3f, mu = %.4f\n', alpha(m), mu_val);

    % Define problem
    % Nonlinear operator: dealias_mask .* Laplacian_hat .* fft( u^3 - mu*u )
    nonlin_op = @(u_hat, t) dealias_mask .* Laplacian_hat .* fft( real(ifft(u_hat)).^3 - mu_val*real(ifft(u_hat)) );

    problem = EvolutionProblem(L_operator, nonlin_op, u0_hat, [t0, T]);

    % Solver
    solver = ETDRK4Solver(16);

    monitor = BifurcationMonitor(mu_val, kx, Lx, Nx, x, k_j, ...
        plot_every=plot_every, ...
        save_video=save_video, ...
        video_filename=sprintf('long_alpha=%.3f_T=%.0f.mp4', alpha(m), T), ...
        video_framerate=30);

    % Execute
    sol = evolution_solve(problem, solver, dt, save_every=plot_every, monitors=monitor);
    disp(['Integration complete for alpha = ', num2str(alpha(m))]);
end


function [mu_j, k_j] = critical_bifurcation(j, L, mass)
    % Computes the j'th frequencies critical mu and associated eigenvalue
    k_j = 2 * pi * j / L;
    mu_j = 3 * mass^2 + k_j^2;    
end