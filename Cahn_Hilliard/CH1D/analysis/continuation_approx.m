clear; close all; clc;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Description
% Computes steady-state bifurcation diagrams for 1D CH w/
% fixed L, varying mu, 2 different values of mass m
% Domain: x \in [-pi*L, pi*L)

%% NOTE: 
% in connection with the center manifold analysis, "A" below in the code is 
% equal to TWICE the "A" in the LaTeX. this is because below, we assume everything is real
% so the equation becomes A_text e^{} + A_text e^{} := A_code e^{}

%% Methods:
% uses prediction (secand) - correction (newton's) 
% w/ some flagging to ensure on the dominant mode

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

L = 10;
modes = 1:10;
N = 2^7;
m1 = 0;

% choose max subcritical, get n
n_subc = 3; % 1,...,n modes = subcritical, n+1,... modes = supercritical
m2 = (2*n_subc+1)*sqrt(2)/20; 

n_check = 2;
mu_n = 3*m2^2 + (n_check/L)^2;
mu_next = 3*m2^2 + ((n_check + 1)/L)^2;
mu_subc = (mu_n + mu_next)/2;

% choose mass, get max subcritical
% m2 = 0.6;
% n_subc = sqrt(2*(m2*L)^2);
% mu_subc = 3*m2^2 + (n_subc/L)^2;

disp(sprintf("Predicted largest subcritical mode is n = %d", floor(n_subc)));
fig = figure('Position',[50 50 1300 850]);

% Parent layout
outer = tiledlayout(fig, 2, 1, ...
    'TileSpacing','loose', ...
    'Padding','compact');

%% Top row: bifurcation diagrams
top = tiledlayout(outer, 1, 2, ...
    'TileSpacing','compact', ...
    'Padding','compact');
top.Layout.Tile = 1;

sgtitle('Cahn-Hilliard Steady-State Bifurcations', 'FontSize', 14, 'FontWeight', 'bold');

%% Plot 1: m = 0
ax1 = nexttile(top);
hold(ax1,'on'); box(ax1,'on'); grid(ax1,'on');
colors = turbo(length(modes)); %colors = lines(length(modes));
all_mu1 = [];

for i = 1:length(modes)
    k = modes(i);
    [mu_v, norm_v, ~, ~] = trace_branch_spectral(k, m1, L, 1, N);
    plot(mu_v, norm_v, '-', 'Color', colors(i,:), 'LineWidth', 2, ...
        'DisplayName', sprintf('Mode %d', k));

    all_mu1 = [all_mu1 mu_v];
end
xlabel('\mu', 'FontSize', 12); ylabel('||u - m||_{L_2}', 'FontSize', 12);
title(sprintf('Bifurcation Diagram (m = %g, Unstable)', m1), 'FontSize', 12);
padding = 0.02*(max(all_mu1)-min(all_mu1));

xlim([min(all_mu1)-padding, max(all_mu1)+padding]); %xlim([0, 0.4]); ylim([0, 1.2]);
plot([0, max(all_mu1)+padding], [0, 0], 'k-', 'LineWidth', 2, 'DisplayName', 'Trivial mode');

%% Plot 2: m \neq 0
ax2 = nexttile(top);
hold(ax2,'on'); box(ax2,'on'); grid(ax2,'on');

all_mu2 = [];

p_mu_p = {}; p_prof_p = {}; 
p_mu_n = {}; p_prof_n = {}; 

for i = 1:length(modes)
    k = modes(i);
    % Trace both peak/positive (droplet) and valley/negative (bubble) branches
    [mu_p, norm_p, prof_p, x_plot] = trace_branch_spectral(k, m2, L, 1, N);  
    [mu_n, norm_n, prof_n, ~]      = trace_branch_spectral(k, m2, L, -1, N); 
    
    p_mu_p{i} = mu_p; p_prof_p{i} = prof_p;
    p_mu_n{i} = mu_n; p_prof_n{i} = prof_n;
    
    plot(mu_p, norm_p, '-', 'Color', colors(i,:), 'LineWidth', 2, ...
        'DisplayName', sprintf('Mode %d', k));
    plot(mu_n, norm_n, '--', 'Color', colors(i,:), 'LineWidth', 2, ...
        'HandleVisibility', 'off');

    all_mu2 = [all_mu2 mu_p mu_n];
end
xlabel('\mu', 'FontSize', 12); ylabel('||u - m||_{L_2}', 'FontSize', 12);
title(sprintf('Bifurcation Diagram (m = %g, Transitional)', m2), 'FontSize', 12);

padding = 0.02*(max(all_mu2)-min(all_mu2));
xlim([min(all_mu2)-padding, max(all_mu2)+padding]); %xlim([0.4, 0.9]); ylim([0, 1.2]);
plot([0, max(all_mu2)+padding], [0, 0], 'k-', 'LineWidth', 2, 'DisplayName', 'Trivial mode');
xline(ax2, mu_subc, ':', 'LineWidth', 2, 'DisplayName', sprintf('Cutoff n=%.2f', n_subc));

text(0.1, 1.35, 'Solid: Droplet (Peak at x=0)', 'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'top', 'BackgroundColor', 'w');
text(0.1, 1.25, 'Dashed: Bubble (Valley at x=0)', 'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'top', 'BackgroundColor', 'w');

% --- ONE shared legend for ax1+ax2 ---
linkaxes([ax1 ax2],'xy'); % combine axes

h1 = findobj(ax1.Children, 'flat', '-property', 'DisplayName', '-and', '-not', {'DisplayName',''});
h2 = findobj(ax2.Children, 'flat', '-property', 'DisplayName', '-and', '-not', {'DisplayName',''});
lgd_top = legend(ax2, [h2], 'Orientation','horizontal', 'NumColumns', 6);
lgd_top.Layout.Tile = 'south';
lgd_top.Layout.TileSpan = [1 2];

%% Bottom row: solution profiles
bottom = tiledlayout(outer, 1, 2, ...
    'TileSpacing','compact', ...
    'Padding','compact');
bottom.Layout.Tile = 2;

%% Plot 3: Profiles for m \neq 0 (Mode 1, peak/droplet branch)
ax3 = nexttile(bottom);
hold(ax3,'on'); box(ax3,'on'); grid(ax3,'on');

idx_list = round(linspace(1, size(p_prof_p{1}, 2), 6));
cc = parula(length(idx_list) + 1);

for i = 1:length(idx_list)
    idx = idx_list(i);
    plot(x_plot, p_prof_p{1}(:, idx), 'Color', cc(i,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('\\mu = %.3f', p_mu_p{1}(idx)));
end
yline(m2, 'k--', 'LineWidth', 1, 'DisplayName', 'm=0.5 (Mean)');
xlabel('x'); ylabel('u(x)');
title('Mode 1 Profiles: Droplet Phase');
xlim([-pi*L, pi*L]);

%% Plot 4: Profiles for m \neq 0 (Mode 1, valley/bubble branch)
ax4 = nexttile(bottom);
hold(ax4,'on'); box(ax4,'on'); grid(ax4,'on');
idx_list2 = round(linspace(1, size(p_prof_n{1}, 2), 6));

for i = 1:length(idx_list2)
    idx = idx_list2(i);
    plot(x_plot, p_prof_n{1}(:, idx), 'Color', cc(i,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('\\mu = %.3f', p_mu_n{1}(idx)));
end
yline(m2, 'k--', 'LineWidth', 1, 'DisplayName', 'm=0.5 (Mean)');
xlabel('x'); ylabel('u(x)');
title('Mode 1 Profiles: Bubble Phase');
xlim([-pi*L, pi*L]);

% --- ONE shared legend for ax3+ax4 ---
linkaxes([ax3 ax4],'xy');
h3 = findobj(ax3.Children, 'flat', '-property', 'DisplayName', '-and', '-not', {'DisplayName',''});
h4 = findobj(ax4.Children, 'flat', '-property', 'DisplayName', '-and', '-not', {'DisplayName',''});
lgd_bot = legend(ax4, [h3], 'Orientation','horizontal', 'NumColumns', 4);
lgd_bot.Layout.Tile = 'south';
lgd_bot.Layout.TileSpan = [1 2];

% FOURIER PSEUDOSPECTRAL AMPLITUDE CONTINUATION (SECANT PREDICTOR)
function [mu_vec, norm_vec, prof_u, x_plot] = trace_branch_spectral(mode_n, m, L, signA, N)
    % Grid setup: Domain is [-pi*L, pi*L)
    x = (2*pi*L/N) * (-N/2 : N/2-1)';
    idx_0 = N/2 + 1; % The exact index where x = 0
    
    % Build the explicit N x N Spectral Second Derivative Matrix (D2)
    n_vec = [0:N/2-1 0 -N/2+1:-1]';
    k_vec = n_vec / L; % Wavenumbers for domain of length 2*pi*L
    D2 = real(ifft(bsxfun(@times, -k_vec.^2, fft(eye(N))))); 
    
    % Analytical critical bifurcation point
    mu_crit = 3*m^2 + (mode_n / L)^2; 
    
    % Adaptive amplitude stepping
    A_max = 1.2 * signA;        % stops when amplitude exceeds \pm 1.2
    dA_base = 0.02 * signA;     % starting value of A
    dA = dA_base;
    
    % Initialize outputs explicitly with the analytical bifurcation point
    mu_vec = mu_crit; 
    norm_vec = 0; 
    prof_u = [m*ones(N,1); m]; % profile for the periodic closure padding
    
    % Tracking variables for Secant Predictor
    A_last = 0;
    X_last = [m*ones(N,1); mu_crit; mu_crit*m - m^3];   % equation w/ uknowns [u, \mu, C]
    dX_dA  = [cos(mode_n * x / L); 0; 0];               % linear initial direction
    
    % Solver Options: EXACT Jacobian passed to massively improve reliability
    options = optimoptions('fsolve', 'Display', 'none', ...
                           'SpecifyObjectiveGradient', true, ...    % THIS explicitly provides Jacobian
                           'FunctionTolerance', 1e-8, ...
                           'StepTolerance', 1e-8);
    
    while abs(A_last) < abs(A_max)
        % Compute next target amplitude
        A_curr = A_last + dA;
        if abs(A_curr) > abs(A_max) % if that step overshoots the limit, reduce it
            dA = sign(dA) * (abs(A_max) - abs(A_last));
            A_curr = A_last + dA;
        end
        
        U0 = m + A_curr; % amplitude constraint, using A = u(0) - m
        
        % First-Order Secant Predictor for the initial guess
        % i.e. continue in the same direction as the last step
        X0 = X_last + dX_dA * dA;
        
        [X_sol, fval, exitflag] = fsolve(@(X) spectral_sys(X, m, U0, D2, N, idx_0), X0, options);
        
        % Safeguard: Prevent mode hopping
        % this is because, even if Newton converges, it may go to wrong branch
        % i.e. if we're looking at mode 1 but mode 2 dominates, it may go to mode 2. 
        % so we reject that solution
        if exitflag > 0 && norm(fval) < 1e-5
            u_sol = X_sol(1:N);
            % ensure the targeted mode remains dominant
            u_hat = fft(u_sol - mean(u_sol));
            [~, max_mode_idx] = max(abs(u_hat(2:floor(N/2))));
            if max_mode_idx ~= mode_n
                exitflag = -1; % branch hopped = reject this step.
            end
        end
        
        if exitflag > 0 && norm(fval) < 1e-5
            % Extract and save successful metrics
            mu_vec(end+1) = X_sol(N+1);
            norm_vec(end+1) = sqrt(mean((u_sol - m).^2));
            prof_u(:, end+1) = [u_sol; u_sol(1)];
            
            % Update secant derivative for the next step
            % tangent vector = direction to trace down next
            dX_dA = (X_sol - X_last) / dA;
            
            % Save state
            X_last = X_sol; 
            A_last = A_curr;
            
            % Gently increase step size
            dA = sign(dA) * min(abs(dA)*1.2, abs(dA_base)); 
        else
            % Solver failed or hopped branch. Shrink step and retry.
            dA = dA / 2;
            if abs(dA) < 1e-5
                break; % Reached a turning point or stability limit
            end
        end
    end
    x_plot = [x; pi*L]; % Array for plotting perfectly closed loops
end

% Discretized Nonlinear System with Exact Analytical Jacobian (N+2 Equations)
function [F, J] = spectral_sys(X, m, U0, D2, N, idx_0)
    % Unpack
    u  = X(1:N);
    mu = X(N+1);
    C  = X(N+2);
    
    % 1. Pseudospectral PDE evaluation
    F_pde = D2*u + mu*u - u.^3 - C; % equivalent to solving steady-state equation for CH
    
    % 2. Mass conservation (mean of u must equal m)
    F_mass = mean(u) - m;
    
    % 3. Amplitude constraint (Pin the center point)
    F_amp = u(idx_0) - U0;
    
    F = [F_pde; F_mass; F_amp];
    
    % Analytical Jacobian computation
    if nargout > 1
        % d(F_pde) / d[u, mu, C]
        J_pde_u  = D2 + diag(mu - 3*u.^2);
        J_pde_mu = u;
        J_pde_C  = -ones(N, 1);
        
        % d(F_mass) / d[u, mu, C]
        J_mass_u  = ones(1, N) / N;
        J_mass_mu = 0;
        J_mass_C  = 0;
        
        % d(F_amp) / d[u, mu, C]
        J_amp_u = zeros(1, N);
        J_amp_u(idx_0) = 1;
        J_amp_mu = 0;
        J_amp_C  = 0;
        
        J = [J_pde_u,  J_pde_mu,  J_pde_C;
             J_mass_u, J_mass_mu, J_mass_C;
             J_amp_u,  J_amp_mu,  J_amp_C];
    end
end