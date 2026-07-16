clear; close all; clc;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Description
% Computes steady-state bifurcation diagrams for 1D CH w/
% fixed L, varying mu, 2 different values of mass m

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

L = 10;
modes = 1:3; % First 3 spatial modes
m1 = 0;
m2 = 0.5;
N = 2^7;

fig = figure('Position', [50 50 1300 850]);

%% Plot 1: m = 0
subplot(2, 2, 1); hold on; box on; grid on;
plot([0, 1], [0, 0], 'k-', 'LineWidth', 2, 'DisplayName', 'Tivial mode');
colors = lines(length(modes));

for i = 1:length(modes)
    k = modes(i);
    [mu_v, norm_v, ~, ~] = trace_branch_spectral(k, m1, L, 1, N);
    plot(mu_v, norm_v, '-', 'Color', colors(i,:), 'LineWidth', 2, ...
        'DisplayName', sprintf('Mode %d', k));
end
xlabel('\mu', 'FontSize', 12); ylabel('||u - m||_{L_2}', 'FontSize', 12);
title(sprintf('Bifurcation Diagram (m = %g, Unstable)', m1), 'FontSize', 12);
xlim([0, 0.4]); ylim([0, 1.2]);
legend('Location', 'northwest', 'FontSize', 10);

%% Plot 2: m \neq 0
subplot(2, 2, 2); hold on; box on; grid on;
plot([0, 1], [0, 0], 'k-', 'LineWidth', 2, 'DisplayName', 'Trivial mode');

p_mu_p = {}; p_prof_p = {}; 
p_mu_n = {}; p_prof_n = {}; 

for i = 1:length(modes)
    k = modes(i);
    % Trace both peak (droplet) and valley (bubble) branches
    [mu_p, norm_p, prof_p, x_plot] = trace_branch_spectral(k, m2, L, 1);  
    [mu_n, norm_n, prof_n, ~]      = trace_branch_spectral(k, m2, L, -1); 
    
    p_mu_p{i} = mu_p; p_prof_p{i} = prof_p;
    p_mu_n{i} = mu_n; p_prof_n{i} = prof_n;
    
    plot(mu_p, norm_p, '-', 'Color', colors(i,:), 'LineWidth', 2, ...
        'DisplayName', sprintf('Mode %d', k));
    plot(mu_n, norm_n, '--', 'Color', colors(i,:), 'LineWidth', 2, ...
        'HandleVisibility', 'off');
end
xlabel('\mu', 'FontSize', 12); ylabel('||u - m||_{L_2}', 'FontSize', 12);
title(sprintf('Bifurcation Diagram (m = %g, Transitional)', m2), 'FontSize', 12);
xlim([0.4, 0.9]); ylim([0, 1.2]);
legend('Location', 'northwest', 'FontSize', 10);
text(0.42, 1.1, 'Solid: Droplet (Peak at x=0)', 'FontSize', 10, 'BackgroundColor', 'w');
text(0.42, 1.0, 'Dashed: Bubble (Valley at x=0)', 'FontSize', 10, 'BackgroundColor', 'w');

%% Plot 3: Profiles for m \neq 0 (Mode 1, Peak branch)
subplot(2, 2, 3); hold on; box on; grid on;
idx_list = round(linspace(1, size(p_prof_p{1}, 2), 6));
cc = parula(length(idx_list) + 1);

for i = 1:length(idx_list)
    idx = idx_list(i);
    plot(x_plot, p_prof_p{1}(:, idx), 'Color', cc(i,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('\\mu = %.3f', p_mu_p{1}(idx)));
end
yline(m2, 'k--', 'LineWidth', 1, 'DisplayName', 'm=0.5 (Mean)');
xlabel('x'); ylabel('u(x)');
title('Mode 1 Profiles: Dense Droplet Phase');
xlim([-pi*L, pi*L]);
legend('Location', 'northeast', 'NumColumns', 2);

%% Plot 4: Profiles for m \neq 0 (Mode 1, valley branch)
subplot(2, 2, 4); hold on; box on; grid on;
idx_list2 = round(linspace(1, size(p_prof_n{1}, 2), 6));

for i = 1:length(idx_list2)
    idx = idx_list2(i);
    plot(x_plot, p_prof_n{1}(:, idx), 'Color', cc(i,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('\\mu = %.3f', p_mu_n{1}(idx)));
end
yline(m2, 'k--', 'LineWidth', 1, 'DisplayName', 'm=0.5 (Mean)');
xlabel('x'); ylabel('u(x)');
title('Mode 1 Profiles: Dilute Bubble Phase');
xlim([-pi*L, pi*L]);
legend('Location', 'northeast', 'NumColumns', 2);

sgtitle('Cahn-Hilliard Steady-State Bifurcations (Spectral Collocation + fsolve)', ...
        'FontSize', 14, 'FontWeight', 'bold');

% =========================================================================
% FOURIER PSEUDOSPECTRAL AMPLITUDE CONTINUATION
% =========================================================================

function [mu_vec, norm_vec, prof_u, x_plot] = trace_branch_spectral(mode_n, m, L, signA, N)
    % Grid setup: N must be even. 
    % We solve on [-pi*L, pi*L) so x=0 is exactly in the middle.
    x = (2*pi*L/N) * (-N/2 : N/2-1)';
    idx_0 = N/2 + 1; % The exact index where x = 0
    
    % Build the explicit N x N Spectral Second Derivative Matrix (D2)
    n_vec = [0:N/2-1 0 -N/2+1:-1]';
    k_vec = n_vec / L;
    % D2 operates on a vector u equivalent to ifft(-k^2 .* fft(u))
    D2 = real(ifft(bsxfun(@times, -k_vec.^2, fft(eye(N))))); 
    
    % Analytical critical bifurcation point
    mu_crit = 3*m^2 + (mode_n/L)^2; 
    
    % Adaptive amplitude stepping
    A_max = 1.2 * signA;
    dA_base = 0.02 * signA;
    dA = dA_base;
    A_curr = dA;
    
    mu_vec = []; norm_vec = []; prof_u = [];
    
    % Initial Guess for fsolve
    u_guess = m + A_curr * cos(mode_n * x / L);
    C_guess = mu_crit * m - m^3;
    X0 = [u_guess; mu_crit; C_guess]; % Vector of unknowns: [u_1...u_N, mu, C]
    
    options = optimoptions('fsolve', 'Display', 'none', ...
                           'FunctionTolerance', 1e-6, ...
                           'StepTolerance', 1e-6);
    
    while abs(A_curr) <= abs(A_max)
        % Pin the amplitude at x=0
        U0 = m + A_curr; 
        
        [X_sol, fval, exitflag] = fsolve(@(X) spectral_sys(X, m, U0, D2, N, idx_0), X0, options);
        
        if exitflag > 0 && norm(fval) < 1e-4
            % Extract solution variables
            u_sol = X_sol(1:N);
            mu_sol = X_sol(N+1);
            
            % Save metrics
            mu_vec(end+1) = mu_sol;
            
            % L2 Norm on a periodic grid is simply the RMS of the variance
            norm_vec(end+1) = sqrt(mean((u_sol - m).^2));
            
            % Close the periodic boundary for clean plotting
            x_plot = [x; pi*L]; 
            prof_u(:, end+1) = [u_sol; u_sol(1)];
            
            % Update for next step
            X0 = X_sol; 
            A_curr = A_curr + dA;
            dA = sign(dA) * min(abs(dA)*1.2, abs(dA_base)); % Gently increase step
            
        else
            % Solver failed (step too large). Shrink step and retry.
            dA = dA / 2;
            if abs(dA) < 1e-4
                break; % Hit the limit of the branch
            end
        end
    end
end

% Discretized Nonlinear System (N+2 Equations)
function F = spectral_sys(X, m, U0, D2, N, idx_0)
    % Unpack
    u = X(1:N);
    mu = X(N+1);
    C = X(N+2);
    
    % 1. Pseudospectral PDE evaluation (D2*u is the spectral u_xx)
    F_pde = D2*u + mu*u - u.^3 - C;
    
    % 2. Mass conservation (mean of u must equal m)
    F_mass = mean(u) - m;
    
    % 3. Amplitude constraint: Pin the center point (x=0) to exactly U0
    F_amp = u(idx_0) - U0;
    
    % Combine residuals
    F = [F_pde; F_mass; F_amp];
end