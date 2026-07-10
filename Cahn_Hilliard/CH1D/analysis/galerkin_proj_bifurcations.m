clc; clear; close all;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Description
% Computes the stationary (\epsilon = 0) bifurcation structure of the 
% 1D Cahn-Hilliard equation using Galerkin projection.
% 
% Fixed L, varying \mu.
% Solves for equilibria using Amplitude Continuation, utilizing a custom 
% Newton-Raphson solver to entirely bypass the Optimization Toolbox.

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Parameters
L = 10;
K = -5:5;
Nmodes = length(K);
Kmax = max(K);
index_of = @(k) find(K==k);
m = 0.5;

% Amplitude stepping parameters
A_steps = 100;
A_max = 1.2;
A_vec = linspace(0.01, A_max, A_steps);

%% Build cubic interaction tensor
C = zeros(Nmodes,Nmodes,Nmodes,Nmodes);
for kk = 1:Nmodes
    for mm = 1:Nmodes
        for nn = 1:Nmodes
            for pp = 1:Nmodes
                if K(mm)+K(nn)+K(pp) == K(kk)
                    C(kk,mm,nn,pp) = 1;
                end
            end
        end
    end
end

%% Run Continuation for two masses
masses = [0, m];
titles = {'m = 0', sprintf('m = %.3f', m)};

fig1 = figure('Position', [100, 100, 1200, 500], 'Name', 'Bifurcation Diagram: L2 Norm');
fig2 = figure('Position', [100, 650, 1200, 500], 'Name', 'Bifurcation Diagram: Amplitudes');

colors = lines(Kmax);

for m_idx = 1:length(masses)
    m = masses(m_idx);
    
    % Prepare plots
    figure(fig1); subplot(1, 2, m_idx); hold on; grid on;
    xlabel('\mu'); ylabel('||u||_2');
    title(titles{m_idx});
    
    figure(fig2); subplot(1, 2, m_idx); hold on; grid on;
    xlabel('\mu'); ylabel('Primary Mode Amplitude |U_k|');
    title(titles{m_idx});
    
    % Plot trivial branch (u = m)
    mu_plot = linspace(0, 1.5, 100);
    L2_trivial = abs(m) * ones(size(mu_plot));
    
    figure(fig1); plot(mu_plot, L2_trivial, 'k-', 'LineWidth', 2, 'DisplayName', 'Trivial Branch');
    figure(fig2); plot(mu_plot, zeros(size(mu_plot)), 'k-', 'LineWidth', 2, 'DisplayName', 'Trivial Branch');
    
    % Trace branches for the first 4 modes
    for k_target = 1:4
        
        mu_branch = zeros(A_steps, 1);
        L2_branch = zeros(A_steps, 1);
        
        % Initial guess: mu = critical mu, all other mode amplitudes = 0
        mu_crit = 3*m^2 + (k_target/L)^2;
        vars0 = zeros(Kmax, 1);
        vars0(k_target) = mu_crit; 
        
        % Step through amplitudes
        for a_idx = 1:A_steps
            A = A_vec(a_idx);
            
            % Use our custom base-MATLAB Newton solver instead of fsolve
            target_fun = @(v) stationary_rhs(v, A, k_target, m, L, K, C);
            [vars, success] = newton_solve(target_fun, vars0);
            
            if ~success
                fprintf('Warning: Newton solver failed to converge at A = %.3f for mode %d\n', A, k_target);
            end
            
            % Extract results
            mu_branch(a_idx) = vars(k_target);
            
            % Reconstruct full U vector to compute L2 norm
            U = zeros(Nmodes, 1);
            U(index_of(0)) = m;
            for j = 1:Kmax
                if j == k_target
                    U(index_of(j)) = A; U(index_of(-j)) = A;
                else
                    U(index_of(j)) = vars(j); U(index_of(-j)) = vars(j);
                end
            end
            
            % L2 norm is exactly the norm of the Fourier coefficients
            L2_branch(a_idx) = sqrt(sum(abs(U).^2));
            
            % Update guess for next amplitude step to ensure smooth continuation
            vars0 = vars;
        end
        
        % Plot the branch
        figure(fig1);
        plot(mu_branch, L2_branch, 'Color', colors(k_target,:), 'LineWidth', 2, ...
             'DisplayName', sprintf('Mode k = %d', k_target));
         
        % Plot theoretical bifurcation point
        plot(mu_crit, abs(m), 'o', 'MarkerFaceColor', colors(k_target,:), ...
             'MarkerEdgeColor', 'k', 'MarkerSize', 8, 'HandleVisibility','off');
         
        figure(fig2);
        plot(mu_branch, A_vec, 'Color', colors(k_target,:), 'LineWidth', 2, ...
             'DisplayName', sprintf('Mode k = %d', k_target));
        plot(mu_crit, 0, 'o', 'MarkerFaceColor', colors(k_target,:), ...
             'MarkerEdgeColor', 'k', 'MarkerSize', 8, 'HandleVisibility','off');
    end
    
    figure(fig1); xlim([0, 1.5]); legend('Location', 'northwest');
    figure(fig2); xlim([0, 1.5]); legend('Location', 'northwest');
end

%% Helper Function: Evaluates the stationary Galerkin RHS
function F = stationary_rhs(vars, A, k_target, m, L, K, C)
    Kmax = max(K);
    Nmodes = length(K);
    
    U = zeros(Nmodes, 1);
    idx_zero = find(K==0);
    U(idx_zero) = m;
    
    mu = 0;
    for j = 1:Kmax
        idx_pos = find(K==j);
        idx_neg = find(K==-j);
        if j == k_target
            U(idx_pos) = A;
            U(idx_neg) = A;
            mu = vars(j);
        else
            U(idx_pos) = vars(j);
            U(idx_neg) = vars(j);
        end
    end
    
    F = zeros(Kmax, 1);
    for j = 1:Kmax
        idx_j = find(K==j);
        nonlinear = 0;
        for mm = 1:Nmodes
            for nn = 1:Nmodes
                for pp = 1:Nmodes
                    if C(idx_j, mm, nn, pp)
                        nonlinear = nonlinear + U(mm)*U(nn)*U(pp);
                    end
                end
            end
        end
        F(j) = (mu - j^2/L^2) * U(idx_j) - nonlinear;
    end
end

%% Custom Newton-Raphson Solver (Replaces fsolve)
function [x, success] = newton_solve(fun, x0)
    % A simple implementation of Newton's method with a numerical Jacobian
    max_iter = 50;
    tol = 1e-10;
    dx = 1e-6; % Step size for numerical derivative
    
    x = x0;
    n = length(x);
    success = false;
    
    for iter = 1:max_iter
        F = fun(x);
        
        % Check for convergence
        if norm(F) < tol
            success = true;
            break;
        end
        
        % Approximate Jacobian matrix numerically
        J = zeros(n, n);
        for j = 1:n
            x_step = x;
            x_step(j) = x_step(j) + dx;
            F_step = fun(x_step);
            J(:, j) = (F_step - F) / dx;
        end
        
        % Solve for the Newton step (J * step = F) and update
        % Using the pseudo-inverse/backslash operator
        step = J \ F;
        x = x - step;
    end
end