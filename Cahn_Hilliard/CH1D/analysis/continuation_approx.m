clear; close all; clc;

%---------------------------------------------------------------------------------------
%---------------------------------------------------------------------------------------

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

%---------------------------------------------------------------------------------------
%---------------------------------------------------------------------------------------

L = 10;
modes = 1:10;
N = 2^7;
m1 = 0;

% choose max subcritical, get n
n_subc = 7; % 1,...,n modes = subcritical, n+1,... modes = supercritical
m2 = (2*n_subc+1)*sqrt(2)/20; 

n_check = 2;
mu_n = 3*m2^2 + (n_check/L)^2;
mu_next = 3*m2^2 + ((n_check + 1)/L)^2;
mu_subc = (mu_n + mu_next)/2;

% choose mass, get max subcritical
% m2 = 0.6;
% n_subc = sqrt(2*(m2*L)^2);
% mu_subc = 3*m2^2 + (n_subc/L)^2;

max_mu_c = 3*m2^2 + (max(modes)/L)^2;
mu_target = max_mu_c + 0.5; 

fprintf('Predicted largest subcritical mode is n = %d\n', floor(n_subc));
fprintf('Tracing all branches until mu = %.3f to ensure axis alignment.\n', mu_target);

% FIGURE 1: BIFURCATION DIAGRAMS & PROFILES
fig1 = figure('Position', [50, 50, 1200, 900], 'Color', 'w');
outer = tiledlayout(fig1, 2, 1, 'TileSpacing', 'loose', 'Padding', 'compact');

%% TOP ROW: Bifurcation Diagrams
top = tiledlayout(outer, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
top.Layout.Tile = 1;

title_str = '\textbf{Steady-State Bifurcations of the Cahn-Hilliard Equation}';
sgtitle(title_str, 'Interpreter', 'latex', 'FontSize', 18);

colors = turbo(length(modes)); 
colors = colors * 0.9; % Darken slightly for better print contrast

% Plot 1: m = 0
ax1 = nexttile(top);
hold(ax1, 'on'); box(ax1, 'on'); grid(ax1, 'on'); ax1.GridAlpha = 0.2;

all_mu1 = [];
for i = 1:length(modes)
    k = modes(i);
    % Pass mu_target to guarantee this curve extends far enough to the right
    [mu_v, norm_v, ~, ~] = trace_branch_spectral(k, m1, L, 1, N, mu_target);
    plot(ax1, mu_v, norm_v, '-', 'Color', colors(i,:), ...
        'DisplayName', sprintf('Mode %d', k));
    all_mu1 = [all_mu1, mu_v];
end
xlabel('Bifurcation Parameter ($\mu$)', 'Interpreter', 'latex'); 
ylabel('$||u - m||_{L_2}$', 'Interpreter', 'latex');
title(sprintf('\\textbf{Trivial Mass } ( $m = %g$)', m1), 'Interpreter', 'latex');

% Plot 2: m \neq 0
ax2 = nexttile(top);
hold(ax2, 'on'); box(ax2, 'on'); grid(ax2, 'on'); ax2.GridAlpha = 0.2;

all_mu2 = [];
p_mu_p = {}; p_prof_p = {}; 
p_mu_n = {}; p_prof_n = {}; 

for i = 1:length(modes)
    k = modes(i);
    [mu_p, norm_p, prof_p, x_plot] = trace_branch_spectral(k, m2, L, 1, N, mu_target);  
    [mu_n, norm_n, prof_n, ~]      = trace_branch_spectral(k, m2, L, -1, N, mu_target); 
    
    p_mu_p{i} = mu_p; p_prof_p{i} = prof_p;
    p_mu_n{i} = mu_n; p_prof_n{i} = prof_n;
    
    plot(ax2, mu_p, norm_p, '-', 'Color', colors(i,:), ...
        'DisplayName', sprintf('Mode %d', k));
    plot(ax2, mu_n, norm_n, '--', 'Color', colors(i,:), ...
        'HandleVisibility', 'off');

    all_mu2 = [all_mu2, mu_p, mu_n];
end
xlabel('Bifurcation Parameter ($\mu$)', 'Interpreter', 'latex'); 
ylabel('$||u - m||_{L_2}$', 'Interpreter', 'latex');
title(sprintf('\\textbf{Transitional Mass } ($ m = %.3f$)', m2), 'Interpreter', 'latex');

% SYNCHRONIZE TOP AXES
global_mu_min = min([all_mu1, all_mu2]);
global_mu_max = max([all_mu1, all_mu2]);
padding = 0.05 * (global_mu_max - global_mu_min);

xlim([ax1, ax2], [global_mu_min - padding, global_mu_max + padding]);
ylim_max = max(ylim(ax2));
ylim([ax1, ax2], [0, ylim_max]);

% Zero lines and subcritical cutoff
plot(ax1, [0, global_mu_max+padding], [0, 0], 'k-', 'HandleVisibility','off');
plot(ax2, [0, global_mu_max+padding], [0, 0], 'k-', 'HandleVisibility','off');
xline(ax2, mu_subc, 'k:', 'LineWidth', 1.5, 'DisplayName', sprintf('Cutoff $n=%.2f$', n_subc));

% Phase styling annotations
text(ax2, global_mu_min, ylim_max*0.93, ' Solid: Droplet (Peak at $x=0$)', ...
    'Interpreter', 'latex', 'BackgroundColor', 'w', 'EdgeColor', 'k');
text(ax2, global_mu_min, ylim_max*0.84, ' Dashed: Bubble (Valley at $x=0$)', ...
    'Interpreter', 'latex', 'BackgroundColor', 'w', 'EdgeColor', 'k');

% Shared Legend for Top Row
h2 = findobj(ax2.Children, 'flat', '-property', 'DisplayName', '-and', '-not', {'DisplayName',''});
lgd_top = legend(ax2, h2, 'Orientation', 'horizontal', 'NumColumns', 6, 'Location', 'south');
lgd_top.Layout.Tile = 'south';
lgd_top.Layout.TileSpan = [1 2];

%% BOTTOM ROW: Solution Profiles
bottom = tiledlayout(outer, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
bottom.Layout.Tile = 2;

% Plot 3: Mode 1 Droplet
ax3 = nexttile(bottom);
hold(ax3, 'on'); box(ax3, 'on'); grid(ax3, 'on'); ax3.GridAlpha = 0.2;
idx_list = round(linspace(1, size(p_prof_p{1}, 2), 6));
cc = parula(length(idx_list) + 1);

for i = 1:length(idx_list)
    idx = idx_list(i);
    plot(ax3, x_plot, p_prof_p{1}(:, idx), 'Color', cc(i,:), ...
        'DisplayName', sprintf('$\\mu = %.3f$', p_mu_p{1}(idx)));
end
yline(ax3, m2, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Mean ($m$)');
xlabel('$x$', 'Interpreter', 'latex'); ylabel('$u(x)$', 'Interpreter', 'latex');
title('\textbf{Mode 1 Evolution: Droplet Phase}', 'Interpreter', 'latex');
xlim(ax3, [-pi*L, pi*L]);

% Plot 4: Mode 1 Bubble
ax4 = nexttile(bottom);
hold(ax4, 'on'); box(ax4, 'on'); grid(ax4, 'on'); ax4.GridAlpha = 0.2;

% FIX: We now generate a unique index list based specifically on the size of the negative branch
idx_list2 = round(linspace(1, size(p_prof_n{1}, 2), 6));

for i = 1:length(idx_list2)
    idx = idx_list2(i);
    plot(ax4, x_plot, p_prof_n{1}(:, idx), 'Color', cc(i,:), ...
        'DisplayName', sprintf('$\\mu = %.3f$', p_mu_n{1}(idx)));
end
yline(ax4, m2, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Mean ($m$)');
xlabel('$x$', 'Interpreter', 'latex'); ylabel('$u(x)$', 'Interpreter', 'latex');
title('\textbf{Mode 1 Evolution: Bubble Phase}', 'Interpreter', 'latex');
xlim(ax4, [-pi*L, pi*L]);
ylim([ax3, ax4], [min(ylim(ax4)), max(ylim(ax3))]); % Match Y limits

% Shared Legend for Bottom Row
h3 = findobj(ax3.Children, 'flat', '-property', 'DisplayName', '-and', '-not', {'DisplayName',''});
lgd_bot = legend(ax3, h3, 'Orientation', 'horizontal', 'NumColumns', 4);
lgd_bot.Layout.Tile = 'south';
lgd_bot.Layout.TileSpan = [1 2];


% FIGURE 2: THEORETICAL VS COMPUTED EIGENVALUES (Validation)
fig2 = figure('Position', [100, 100, 800, 700], 'Color', 'w');
val_layout = tiledlayout(fig2, 2, 1, 'TileSpacing', 'compact');

x_test = (2*pi*L/N) * (-N/2 : N/2-1)';
k_vec = [0:N/2-1, 0, -N/2+1:-1]' / L;
D2_test = real(ifft(bsxfun(@times, -k_vec.^2, fft(eye(N))))); 
num_evals = sort(real(eig(-D2_test)));
num_evals(num_evals < 1e-10) = []; 
num_evals = num_evals(1:2:end);    

k_range = 1:12;
mu_theo = 3*m2^2 + (k_range ./ L).^2;
mu_num  = 3*m2^2 + num_evals(k_range)';

% Subplot A: Parity Plot
ax_val1 = nexttile(val_layout);
hold(ax_val1, 'on'); box(ax_val1, 'on'); grid(ax_val1, 'on'); ax_val1.GridAlpha = 0.2;
plot(ax_val1, k_range, mu_theo, 'k-', 'LineWidth', 4, 'DisplayName', 'Theoretical $\mu_{k,c}$');
plot(ax_val1, k_range, mu_num, 'r.', 'MarkerSize', 24, 'DisplayName', 'Spectral Computed $\mu_{k,c}$');
ylabel('Critical $\mu$', 'Interpreter', 'latex');
title('\textbf{Validation: Spectral Method vs. Analytical Bifurcation Points}', 'Interpreter', 'latex');
legend(ax_val1, 'Location', 'northwest');
set(ax_val1, 'XTickLabel', []); 

% Subplot B: Error Plot
ax_val2 = nexttile(val_layout);
hold(ax_val2, 'on'); box(ax_val2, 'on'); grid(ax_val2, 'on'); ax_val2.GridAlpha = 0.2;
rel_error = abs(mu_theo - mu_num) ./ abs(mu_theo) + 1e-18; 
semilogy(ax_val2, k_range, rel_error, 'bo-', 'MarkerFaceColor', 'b', 'LineWidth', 2);
xlabel('Wavenumber Mode ($k$)', 'Interpreter', 'latex');
ylabel('Relative Error', 'Interpreter', 'latex');
title('\textbf{Relative Error Magnitude}', 'Interpreter', 'latex');
xlim([ax_val1, ax_val2], [1, 12]);
ylim(ax_val2, [1e-16, 1e-12]); 


% HELPER FUNCTIONS
function [mu_vec, norm_vec, prof_u, x_plot] = trace_branch_spectral(mode_n, m, L, signA, N, mu_target)
    x = (2*pi*L/N) * (-N/2 : N/2-1)';
    idx_0 = N/2 + 1; 
    
    n_vec = [0:N/2-1 0 -N/2+1:-1]';
    k_vec = n_vec / L; 
    D2 = real(ifft(bsxfun(@times, -k_vec.^2, fft(eye(N))))); 
    
    mu_crit = 3*m^2 + (mode_n / L)^2; 
    
    % We allow A to grow up to a very safe upper bound (4.0).
    % The loop is primarily governed by the target mu now.
    A_max = 4.0 * signA;        
    dA_base = 0.008 * signA;     
    dA = dA_base;
    
    mu_vec = mu_crit; 
    norm_vec = 0; 
    prof_u = [m*ones(N,1); m]; 
    
    A_last = 0;
    X_last = [m*ones(N,1); mu_crit; mu_crit*m - m^3];   
    dX_dA  = [cos(mode_n * x / L); 0; 0];               
    
    options = optimoptions('fsolve', 'Display', 'none', ...
                           'SpecifyObjectiveGradient', true, ...    
                           'FunctionTolerance', 1e-9, ...
                           'StepTolerance', 1e-9);
    
    while abs(A_last) < abs(A_max)
        A_curr = A_last + dA;
        if abs(A_curr) > abs(A_max) 
            dA = sign(dA) * (abs(A_max) - abs(A_last));
            A_curr = A_last + dA;
        end
        
        U0 = m + A_curr; 
        X0 = X_last + dX_dA * dA;
        
        [X_sol, fval, exitflag] = fsolve(@(X) spectral_sys(X, m, U0, D2, N, idx_0), X0, options);
        
        if exitflag > 0 && norm(fval) < 1e-5
            u_sol = X_sol(1:N);
            u_hat = fft(u_sol - mean(u_sol));
            [~, max_mode_idx] = max(abs(u_hat(2:floor(N/2))));
            if max_mode_idx ~= mode_n
                exitflag = -1; 
            end
        end
        
        if exitflag > 0 && norm(fval) < 1e-5
            mu_new = X_sol(N+1);
            
            mu_vec(end+1) = mu_new;
            norm_vec(end+1) = sqrt(mean((u_sol - m).^2));
            prof_u(:, end+1) = [u_sol; u_sol(1)];
            
            dX_dA = (X_sol - X_last) / dA;
            X_last = X_sol; 
            A_last = A_curr;
            
            % break condition: hit axis boundary
            if mu_new > mu_target
                break;
            end
            
            dA = sign(dA) * min(abs(dA)*1.02, 0.02); 
        else
            dA = dA / 2;
            if abs(dA) < 1e-6
                break; 
            end
        end
    end
    x_plot = [x; pi*L]; 
end

function [F, J] = spectral_sys(X, m, U0, D2, N, idx_0)
    u  = X(1:N);
    mu = X(N+1);
    C  = X(N+2);
    
    F_pde = D2*u + mu*u - u.^3 - C; 
    F_mass = mean(u) - m;
    F_amp = u(idx_0) - U0;
    
    F = [F_pde; F_mass; F_amp];
    
    if nargout > 1
        J_pde_u  = D2 + diag(mu - 3*u.^2);
        J_pde_mu = u;
        J_pde_C  = -ones(N, 1);
        
        J_mass_u  = ones(1, N) / N;
        J_mass_mu = 0;
        J_mass_C  = 0;
        
        J_amp_u = zeros(1, N);
        J_amp_u(idx_0) = 1;
        J_amp_mu = 0;
        J_amp_C  = 0;
        
        J = [J_pde_u,  J_pde_mu,  J_pde_C;
             J_mass_u, J_mass_mu, J_mass_C;
             J_amp_u,  J_amp_mu,  J_amp_C];
    end
end