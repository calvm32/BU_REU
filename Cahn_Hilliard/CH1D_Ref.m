clc; clear; close all;

%% Parameters
eps = 0.05; % used for scaling mu
blowup_time = eps^(-2/3);

tspan = [-2, blowup_time + 100.0];
dt = 0.1;

scale = 100;
Lx = scale*pi;

Nx = 2^11;
dx = Lx/Nx;

plot_every = 5;


%% functions
function mu_val = mu(t)
    mu_val = t*eps;
end

function P = power_spec(t, Laplacian_k, u_hat)
    ksq = -Laplacian_k;
    P = abs(u_hat).^2 .* exp(t^2*ksq*eps - 2*t*ksq.^2);
end

function n = domainwall_density_theory(t, Laplacian_k, u_hat)
    P = power_spec(t, Laplacian_k, u_hat);
    ksq = -Laplacian_k;

    num = sum(ksq.*P,'all');
    den = sum(P,'all');

    n = (1/pi)*sqrt(num/den);
end

function density = domainwall_density_computed(u, domain_length)
    walls = sum(u .* circshift(u,-1) < 0);
    density = walls / domain_length;
end

function F = free_energy(u, mu, dx)
    ux = gradient(u, dx);
    density = 0.25*u.^4 -0.5*mu*u.^2 +0.5*ux.^2;
    F = sum(density)*dx;
end

%% Initial condition
sigma = 0.01;
u0 = sigma*randn(1,Nx);

u_hat = fft(u0);
u = u0;

% spatial grid
x = (-Nx/2:Nx/2-1)*dx;

%% Fourier wavenumbers
kx = 2*pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
Laplacian_hat = - kx.^2;

%% Set-up problem
global Laplacian_hat dealias_mask

% 2/3 dealiasing mask
kx_max = max(abs(kx));
dealias_mask = abs(kx) <= (2/3)*kx_max;
Laplacian_hat = -kx.^2;

% Linear operator:
L_operator = -(Laplacian_hat.^2);

L_op = -kx.^2;

function [Nu_hat] = N_op(u_hat, t)
    global Laplacian_hat dealias_mask

    u = real(ifft(u_hat));
    Nu_hat = Laplacian_hat .* fft(u.^3 - mu(t) .* u);
    Nu_hat = Nu_hat .* dealias_mask;
end

prob = EvolutionProblem(L_op, @N_op, tspan, dt);

%% Plot setup
fig = figure('Position',[100 100 1400 450]);

% Pre-allocate tracking metrics
l2_hist = zeros(1, length(prob.t));
l2_hist(1) = l2_norm_periodic_1D(u_hat, Lx);

computed_density = zeros(1, length(prob.t));
computed_density(1) = domainwall_density_computed(u, Lx);

theory_density = zeros(1, length(prob.t));
theory_density(1) = domainwall_density_theory(tspan(1), Laplacian_hat, u_hat);

energy_hist = zeros(1, length(prob.t));
energy_hist(1) = free_energy(u0, mu(tspan(1)), dx);

% Subplot 1: Solution
ax1 = subplot(2,2,1);
u_line = plot(ax1, x, u, 'LineWidth', 1.5);

ylim(ax1, [-1.2 1.2]); 
xlabel(ax1, 'x'); 
ylabel(ax1, 'u');

hTitle1 = title(ax1, sprintf('t = %.3f, \\mu = %.3f', tspan(1), mu(tspan(1))));
hTitle1.Interpreter = 'tex';

grid(ax1, 'on');

%% Subplot 2: Mean value
ax2 = subplot(2,2,2);
l2_line = semilogy(ax2, prob.t(1), l2_hist(1), 'LineWidth', 2);

xlabel(ax2, 't'); 
ylabel(ax2, 'log(||u||_{L^2})');
title(ax2, sprintf('L2 Norm of u'));

grid(ax2, 'on');

%% Subplot 3: Domain Wall Density
ax3 = subplot(2,2,3);

theory_density_line = plot(ax3, prob.t(1), theory_density(1), ...
    'LineWidth',2,'DisplayName','theoretical');

hold(ax3, 'on');

computed_density_line = plot(ax3, prob.t(1), computed_density(1), ...
    'LineWidth',2,'DisplayName','computed');

title(ax3, sprintf('Domain Wall Densities'));
grid(ax3, 'on');
legend(ax3, 'show');

%% Subplot 4: Free Energy
ax4 = subplot(2,2,4);

energy_line = plot(ax4, prob.t(1), energy_hist(1), 'LineWidth',2);

title(ax3, sprintf('Free Energy'));
grid(ax4, 'on');

drawnow limitrate

%% Setup callback functions
function callback(u_hat, tcurr, n)
    u = real(ifft(u_hat));

    % Update data arrays
    computed_density(n) = domainwall_density_computed(u, Lx);
    theory_density(n) = domainwall_density_theory(tcurr, Laplacian_hat, u_hat);
    l2_hist(n) = l2_norm_periodic_1D(u_hat, Lx);
end

function save_callback(u_hat, tcurr, n)
    u = real(ifft(u_hat));

    u_line.YData = u;

    hTitle1.String = sprintf('t = %.3f, \\mu = %.3f', t, mu(tcurr));
    hTitle1.Interpreter = 'tex';

    l2_line.XData = prob.t(1:n);
    l2_line.YData = l2_hist(1:n);
    axis(ax2, 'tight');

    blowup_mask = prob.t(1:n) >= blowup_time;

    theory_density_line.XData = prob.t(blowup_mask);
    theory_density_line.YData = theory_density(blowup_mask);

    computed_density_line.XData = prob.t(blowup_mask);
    computed_density_line.YData = computed_density(blowup_mask);

    energy_hist(n) = free_energy(u, mu(tcurr), dx);
    energy_line.XData = prob.t(1:n);
    energy_line.YData = energy_hist(1:n);

    axis(ax3, 'tight');

    drawnow limitrate 
end



%% L2 norm
function val = l2_norm_periodic_1D(u_hat,Lx)
    Nx = length(u_hat);
    val = sqrt(Lx)* sqrt(sum(abs(u_hat).^2))/Nx;
end

%% Solve
sol = evolution_solve(prob, ETDRK4Solver(), u0, ...
    save_every=plot_every, callback=@callback, ...
    save_callback=@save_callback);
