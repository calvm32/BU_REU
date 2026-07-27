clc; clear; close all;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Description
% Find how domain wall density corresponds with different values (freezeout time, largest modes, etc.)
% and how theoretical values compare with computed values

% note: able to compute with mu = tanh = bounded for all time

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Parameters & Setup
epsilon = 0.05;
alpha = 1.0; 
beta = 0.0; % beta=0 so bifurcation occurs exactly at t=0

% KZ time-scales
freeze_out_time = epsilon^(-2/3);
t0 = -2.0 * freeze_out_time; % Allow for adiabatic tracking
T = freeze_out_time + 100.0;
dt = 0.05;

% Domain setup [-L*pi, L*pi]
L = 100;
Lx = 2 * L * pi;
Nx = 2^11;
dx = Lx/Nx;
plot_dt = 0.5; 
plot_every = round(plot_dt / dt);

x = (-Nx/2:Nx/2-1)*dx;
kx = 2*pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
Laplacian_hat = -kx.^2;
L_operator = -(Laplacian_hat.^2);
kx_max = max(abs(kx));
dealias_mask = abs(kx) <= (2/3)*kx_max;

% ICs
sigma = 0.01;
u0_real = sigma*randn(1,Nx);
u0_real = u0_real - mean(u0_real); % enforce zero mass for H^{-1} bounds
u0_hat = fft(u0_real);

%% Define the Evolution Problem (Bounded mu(t))
mu = @(t) alpha * tanh(epsilon * t) + beta;

nonlin_op = @(u_hat, t) dealias_mask .* Laplacian_hat .* ...
    fft( real(ifft(u_hat)).^3 - mu(t)*real(ifft(u_hat)) ); 

problem = EvolutionProblem(L_operator, nonlin_op, u0_hat, [t0, T]);

%% Setup the Solver
solver = ETDRK4Solver(16); 

%% Execute the Solve with Monitor
params.epsilon = epsilon;
params.alpha = alpha;
params.beta = beta;
params.blowup_time = freeze_out_time;
params.Lx = Lx;
params.Nx = Nx;
params.dx = dx;
params.Laplacian_hat = Laplacian_hat;
params.plot_every = plot_every;
params.x = x;
params.kx = kx;
params.mu = mu;
params.save_video = false;

monitor = DensityL2FreeEnergyMonitor(params); 
disp('Starting integration...');
sol = evolution_solve(problem, solver, dt, save_every=plot_every, monitors=monitor);
disp('Integration complete.');

%% Post-Processing
sol_data = squeeze(sol.solution);
t_data = sol.time(:)'; 
num_steps = length(t_data);

u_data = sol.solution;
if size(u_data, 1) == num_steps && size(u_data, 2) == Nx
    u_data = u_data';
end

% Initialize arrays
H1_norm = zeros(1, num_steps);
Hmin1_norm = zeros(1, num_steps);
dominant_k = zeros(1, num_steps);

% Precompute Fourier weights for Sobolev norms
w_H1 = (1 + kx.^2);
w_Hmin1 = zeros(size(kx));
w_Hmin1(2:end) = 1 ./ (kx(2:end).^2); % avoid division by zero at k=0

for i = 1:num_steps
    u_current_hat = u_data(:, i).';
    power_spectrum = abs(u_current_hat / Nx).^2;
    
    % Pullback attractor norm metrics
    H1_norm(i) = sqrt(sum(w_H1 .* power_spectrum));
    Hmin1_norm(i) = sqrt(sum(w_Hmin1 .* power_spectrum));
    
    % Track dominant wave mode
    [~, max_idx] = max(power_spectrum(1:floor(Nx/2)));
    dominant_k(i) = kx(max_idx);
end

%% KZ Theoretical vs Empirical Integrated Eigenvalue
[~, freeze_out_idx] = min(abs(t_data - freeze_out_time));
t_hat = t_data(freeze_out_idx); 

u_hat_freezeout = u_data(:, freeze_out_idx).';
empirical_power = abs(u_hat_freezeout(1:floor(Nx/2))).^2;
k_pos = kx(1:floor(Nx/2));

% Analytical integrated eigenvalue calculation from t=0 to t_hat for tanh
% \int_0^{\hat{t}} -k^4 + k^2(alpha*tanh(eps*s) + beta) ds
int_eval = -k_pos.^4 .* t_hat + k_pos.^2 .* ( (alpha/epsilon)*log(cosh(epsilon*t_hat)) + beta*t_hat );
theoretical_amplification = exp(2 * int_eval); 

%% Plotting
figure('Position', [100, 100, 1200, 800]);

% Plot 1: Phase Space tracking of the Pullback Attractor bounds
subplot(2,2,1);
plot(Hmin1_norm, H1_norm, 'k-', 'LineWidth', 1.5); hold on;
% Plot theoretical limit bounds
kappa_1 = 2 / (L^4); 
kappa_2 = Lx * (alpha+beta)^2;
rho_0 = sqrt(kappa_2 / kappa_1);
xline(rho_0, 'r--', 'LineWidth', 1);
xlabel('$\|v\|_{H^{-1}}$', 'Interpreter', 'latex');
ylabel('$\|v\|_{H^{1}}$', 'Interpreter', 'latex');
title('Pullback Attractor Trajectory vs $\rho_0$ Bound', 'Interpreter', 'latex');
legend('Trajectory', 'Theoretical $\rho_0$', 'Location', 'Best', 'Interpreter', 'latex');
grid on;

% Plot 2: KZ Structure Factor Peak Match
subplot(2,2,2);
plot(k_pos, empirical_power / max(empirical_power), 'b-', 'LineWidth', 1.5); hold on;
plot(k_pos, theoretical_amplification / max(theoretical_amplification), 'r--', 'LineWidth', 1.5);
xline(epsilon^(1/6), 'k:', 'LineWidth', 2); 
xlim([0, 1.5]);
xlabel('Wavenumber $k$', 'Interpreter', 'latex');
ylabel('Normalized Power Spectrum', 'Interpreter', 'latex');
legend('Empirical FFT', 'Integrated $\lambda_k$', 'KZ $\hat{k} \sim \epsilon^{1/6}$', 'Interpreter', 'latex');
title(sprintf('Structure Factor Profile at $\\hat{t} = %.2f$', t_hat), 'Interpreter', 'latex');
grid on;

% Plot 3: Time Evolution of the Scale Freeze-out
subplot(2,2,[3 4]);
plot(t_data, dominant_k, 'b-', 'LineWidth', 1.5); hold on;
yline(epsilon^(1/6), 'r--', 'LineWidth', 1.5);
xline(freeze_out_time, 'k:', 'LineWidth', 1.5);
xline(0, 'g:', 'LineWidth', 1.5);
ylim([0, 1.2 * max(dominant_k)]);
xlabel('Time $t$', 'Interpreter', 'latex');
ylabel('Dominant $k$', 'Interpreter', 'latex');
title('Evolution of Characteristic Length Scale', 'Interpreter', 'latex');
legend('Empirical Dominant $k$', 'KZ Predicted $\hat{k}$', 'Freeze-out Time $\hat{t}$', 'Bifurcation ($t=0$)', 'Interpreter', 'latex');
grid on;