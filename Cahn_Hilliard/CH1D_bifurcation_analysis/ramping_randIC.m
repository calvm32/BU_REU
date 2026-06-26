clc; clear; close all;

function [mu_j, k_j] = critical_bifurcation(j, L, mass)
    % Computes the j'th frequencies critical mu and associated eigenvalue
    k_j = 2 * pi * j / L;
    mu_j = 3 * mass^2 + k_j^2;    
end

%% Parameters
t0 = 0.0;
T  = 500;
dt = 0.1;

mass = 0;

xscale = 6;
Lx = xscale*pi;
Nx = 2^12;

[mu4, ~] = critical_bifurcation(4, 2 * Lx, mass);
[mu5, ~] = critical_bifurcation(5, 2 * Lx, mass);

alpha = [ 0.1 ];

%% Find k_j for graphing purposes
k_j = pi * [0:6] / Lx;

%% Video paramaters
plot_dt = 1; 
plot_every = round(plot_dt / dt); % make multiple of dt
save_video = true;

%% Initial condition
% spatial grid (needed for patterned base state)
dx = 2*Lx/Nx;
x = (-Nx/2:Nx/2-1)*dx;

% Zero mean white noise
noise = 0.0002 * (rand(1, Nx) - 0.5);
noise = noise - mean(noise);

% Zero mean white noise
u0 = mass + noise;

% Start at k_bif
% u0 = 0.1*cos(k_bif * x) + 0.01 * cos(k_pert * x) + noise + mass;

%% discretization
% time discretization
t = t0:dt:T;
num_time_steps = length(t);

% Fourier wavenumbers
kx = pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
Laplacian_hat = -kx.^2;

%% Linear operator:
L_operator = -(Laplacian_hat.^2);

%% 2/3 dealiasing mask
kx_max = max(abs(kx));
dealias_mask = abs(kx) <= (2/3)*kx_max;

%% ETDRK4 setup
E  = exp(dt*L_operator);
E2 = exp(dt*L_operator/2);
M = 16; % no. of points for complex means
r = exp(1i*pi*((1:M)-0.5)/M); % roots of unity
Lvec = L_operator(:);
LR = dt*Lvec(:,ones(M,1)) + r(ones(numel(L_operator),1),:);

Q  = dt*real(mean((exp(LR/2)-1)./LR,2)).';
f1 = dt*real(mean((-4-LR + exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2)).';
f2 = dt*real(mean((2+LR + exp(LR).*(-2+LR))./LR.^3 ,2)).';
f3 = dt*real(mean((-4-3*LR-LR.^2 + exp(LR).*(4-LR))./LR.^3 ,2)).';

clear LR Lvec

%% Loop over all alphas
for m = 1:length(alpha)
    mu = (1 - alpha(m)) * mu4 + alpha(m)*mu5
    
    % Restart initial conditions
    u_hat = fft(u0);
    u = u0;
    
    %% Time loop
    for n = 2:num_time_steps
        % Stage 1
        u3_nonlinear = u.^3 - mu*u;
        Nu_hat = dealias_mask .* Laplacian_hat .* fft(u3_nonlinear);
        
        % Stage 2
        a_hat = E2.*u_hat + Q.*Nu_hat;
        a = real(ifft(a_hat));
        Na_hat = dealias_mask .* Laplacian_hat .* fft(a.^3 - mu*a);
        
        % Stage 3
        b_hat = E2.*u_hat + Q.*Na_hat;
        b = real(ifft(b_hat));
        Nb_hat = dealias_mask .* Laplacian_hat .* fft(b.^3 - mu*b);
        
        % Stage 4
        c_hat = E2.*a_hat + Q.*(2*Nb_hat - Nu_hat);
        c = real(ifft(c_hat));
        Nc_hat = dealias_mask .* Laplacian_hat .* fft(c.^3 - mu*c);
        
        % Final Time Step Combination
        u_hat = E.*u_hat + f1.*Nu_hat + 2*f2.*(Na_hat + Nb_hat) + f3.*Nc_hat;
        u = real(ifft(u_hat));
    end
    
    disp('Done')
end