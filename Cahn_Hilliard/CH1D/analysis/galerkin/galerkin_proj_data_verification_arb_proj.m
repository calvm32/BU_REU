clc; clear; close all;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Description
% Galerkin projection of 1D Cahn-Hilliard compared against Full PDE
% Initializes the ODE using the exact PDE state at mu = target_mu

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Load Full PDE Data
saved = load('dom_mode_ep=0.00010_mu0=-0.20.mat');
sol = saved.sol;
epsilon = saved.EP(1);
mu0 = saved.mu0;

% Squeeze the solution array to ensure it is cleanly 2D: [modes x time]
sol_data = squeeze(sol.solution);

% Process PDE data for comparison
mu_pde = mu0 + epsilon * sol.time;
amp_pde = abs(sol_data) / saved.Nx; 

%% Find the exact index where mu = -0.2 (or any target start time)
target_mu = -0.2;
[~, idx_start] = min(abs(mu_pde - target_mu));
mu_start = mu_pde(idx_start);

fprintf('Initializing Galerkin ODE at mu = %.6f (PDE index %d)\n', mu_start, idx_start);

%% Parameters
mu_final = max(mu_pde); 

% CHANGE THIS TO PLOT MORE MODES
K = -6:6; 
K_max = max(K);
% --------------------------------------

Nmodes = length(K);
index_of = @(k) find(K==k);

L = 10; % half-domain size

%% Time span for Galerkin ODE 
% In the RHS function, mu = epsilon * t. 
% Therefore, t_start = mu_start / epsilon
t_start = mu_start / epsilon;
t_end   = mu_final / epsilon;

Npts  = 500; 
tspan = linspace(t_start, t_end, Npts);   

%% Build cubic interaction tensor
C = zeros(Nmodes,Nmodes,Nmodes,Nmodes);
for kk = 1:Nmodes
    k = K(kk);
    for mm = 1:Nmodes
        m = K(mm);
        for nn = 1:Nmodes
            n = K(nn);
            for pp = 1:Nmodes
                p = K(pp);
                if m+n+p==k
                    C(kk,mm,nn,pp)=1;
                end
            end
        end
    end
end

% ------------------------------------------------------------------------------------------
%% Initial condition: Extract complex state from PDE at mu_start
% ------------------------------------------------------------------------------------------
u0 = zeros(Nmodes, 1);
num_pde_modes = size(sol_data, 1) - 1; % Total rows minus 1 (for k=0)

% Extract mean mode (k=0) AND DIVIDE BY Nx
u0(index_of(0)) = real(sol_data(1, idx_start)) / saved.Nx;

% Extract available higher modes
for k = 1:min(K_max, num_pde_modes)
    % Extract the exact complex coefficient from the PDE data at mu_start AND DIVIDE BY Nx
    val = sol_data(k+1, idx_start) / saved.Nx; 
    
    u0(index_of(k))  = val;
    u0(index_of(-k)) = conj(val);
end

%% Solve the Galerkin ODE
opts = odeset('RelTol', 1e-14, 'AbsTol', 1e-16);
[t, U] = ode89(@(t,u) rhs(t,u,epsilon,K,C,L), tspan, u0, opts);
mu_vec = epsilon * t;   

%% ------------------------------------------------------------------------------------------
%% Plot Predicted vs True Amplitudes
%% ------------------------------------------------------------------------------------------
figure('Position', [200 200 800 500]);
hold on;

K_plot = min(K_max, num_pde_modes);
colors = lines(K_plot); 
legend_entries = cell(1, 2 * K_plot);

% 1. Plot Full PDE Data (Thick, lighter lines)
for k = 1:K_plot
    light_color = colors(k, :) + (1 - colors(k, :)) * 0.6;
    plot(mu_pde, amp_pde(k+1, :), '-', 'Color', light_color, 'LineWidth', 4);
    legend_entries{k} = sprintf('PDE k=%d', k);
end

% 2. Plot Galerkin Data starting from mu_start (Dashed lines)
for k = 1:K_plot
    plot(mu_vec, abs(U(:, index_of(k))), '--', 'Color', colors(k, :), 'LineWidth', 2);
    legend_entries{K_plot + k} = sprintf('Galerkin k=%d', k);
end

% Draw a vertical line showing where the ODE was initialized
xline(mu_start, 'k:', 'LineWidth', 1.5, 'Label', sprintf('Initialization (\\mu=%.2f)', target_mu));

% Formatting
set(gca, 'YScale', 'log');
xlabel('\mu (Slow Time Scale)', 'FontSize', 12);
ylabel('Mode Amplitudes |U_k|', 'FontSize', 12);
title(sprintf('PDE Data vs Galerkin Approximation (\\epsilon = %.4f)', epsilon), 'FontSize', 14);

legend(legend_entries, 'Location', 'SouthEast', 'FontSize', 9, 'NumColumns', 2);
grid on;
set(gca, 'FontSize', 11);
xlim([mu0, mu_final]);
hold off;

%% Functions
function dudt = rhs(t, u, epsilon, K, C, L)
    Nmodes = length(K);
    mu = epsilon * t;
    dudt = zeros(Nmodes, 1);
    for kk = 1:Nmodes
        nonlinear = 0;
        for mm = 1:Nmodes
            for nn = 1:Nmodes
                for pp = 1:Nmodes
                    nonlinear = nonlinear + C(kk,mm,nn,pp)*u(mm)*u(nn)*u(pp);
                end
            end
        end
        lambda = -(K(kk)^4)/(L^4) + mu*(K(kk)^2)/(L^2);
        dudt(kk) = lambda*u(kk) - (K(kk)^2)/(L^2)*nonlinear;
    end
end