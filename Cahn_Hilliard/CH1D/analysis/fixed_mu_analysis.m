clc; clear; close all;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Description
% This experiment runs through different cosine initial conditions 
% varying in mode. Some white noise is added.

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Parameters
t0 = 0;
T  = 8000;
dt = 0.5;
max_mode_num = 12;
L = 10;

start_time = 0;

% choose max subcritical, get n
n_subc = 2; % 1,...,n modes = subcritical, n+1,... modes = supercritical
mass = (2*n_subc+1)*sqrt(2)/20; 

n_check = 2;
mu_n = 3*mass^2 + (n_check/L)^2;
mu_next = 3*mass^2 + ((n_check + 1)/L)^2;
mu = (mu_n + mu_next)/2;

k_hat = @(k) (k^2)/(L^2);
lambda_k = @(k) k_hat(k) * ( -k_hat(k) + mu - k_hat(k)*mass^2 );
disp(lambda_k(1))

disp(sprintf('using mass = %.2f', mass))
disp(sprintf('using m    = %.2f', mu))

Lx = L*pi;
Nx = 2^12;

plot_dt = 12.0; 
plot_every = round(plot_dt / dt);
save_video = true;

save_every = 8.0;

% Find k_j's for initial data
k_j = pi * [0:max_mode_num+1] / Lx;
k_i = k_j(2:max_mode_num+2);

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
    fprintf('Running k  = %.4f\n', k);
    fprintf('transcritical value  = %.4f\n', lambda_k(k)/mass);

    % Zero mean white noise
    % noise = 0.0002 * (rand(1, Nx) - 0.5);
    % noise = noise - mean(noise);
    
    % Start at k_bif
    u0 = 0.01*cos(0 * x) + mass;
    u0_hat = 10^(-5)*ones(1,Nx)*Nx; %fft(u0);

    u0_hat(1) = mass*Nx;

    % Define problem
    nonlin_op = @(u_hat, t) dealias_mask .* Laplacian_hat .* fft( real(ifft(u_hat)).^3 - mu*real(ifft(u_hat)) );

    problem = EvolutionProblem(L_operator, nonlin_op, u0_hat, [t0, T]);

    % Solver
    solver = ETDRK4Solver(16);

    monitor = BifurcationMonitor(mu, kx, Lx, Nx, x, k_j, ...
        plot_every=plot_every, ...
        save_video=save_video, ...
        video_filename=sprintf('k=%.3f_T=%.0f.avi', k, T), ...
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