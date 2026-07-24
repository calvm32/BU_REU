clc; clear; close all;

% -------------------------------------------------------------------------
% Parameters & Setup
% -------------------------------------------------------------------------
epsilon = 0.05;
alpha = 1.0; 
beta = 0.0; % Bifurcation at t=0

freeze_out_time = epsilon^(-2/3);
t0 = -2.0 * freeze_out_time; 
T = freeze_out_time + 100.0;
dt = 0.05;

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

% ICs - Added a specific non-zero mass to trigger quadratic resonances
sigma = 0.01;
u0_real = sigma*randn(1,Nx);
mass = 0.05; % Explicit non-zero mass for m>0 resonance analysis
u0_real = (u0_real - mean(u0_real)) + mass; 
u0_hat = fft(u0_real);

% Define mu(t) and nonlinear operator
mu = @(t) alpha * tanh(epsilon * t) + beta;
nonlin_op = @(u_hat, t) dealias_mask .* Laplacian_hat .* ...
    fft( real(ifft(u_hat)).^3 - mu(t)*real(ifft(u_hat)) ); 

problem = EvolutionProblem(L_operator, nonlin_op, u0_hat, [t0, T]);
solver = ETDRK4Solver(16); 

% We don't need a custom monitor for the resonance analysis anymore, 
% we will extract it directly from the `sol` output.
disp('Starting integration...');
sol = evolution_solve(problem, solver, dt, save_every=plot_every);
disp('Integration complete.');

% =========================================================================
% PART 2: RIGOROUS RESONANCE ANALYSIS (Extracting from `sol`)
% =========================================================================

% Number of modes to track for resonances (e.g., modes 1 through 5)
num_modes = 5; 
analyze_resonances(sol, mu, mass, L, Nx, num_modes);


%% --- Resonance Analysis Function ---
function analyze_resonances(sol, mu_func, mass, L, Nx, num_modes)
    
    t_grid = sol.time(:)';
    mu_grid = mu_func(t_grid);
    num_steps = length(t_grid);
    
    % Ensure correct orientation of solution data
    u_data = sol.solution;
    if size(u_data, 1) == num_steps && size(u_data, 2) == Nx
        u_data = u_data.'; % Make it Nx by num_steps
    end
    
    % Extract amplitude histories: 
    % MATLAB fft indices: 1 is k=0, 2 is k=1, 3 is k=2, etc.
    amp_history = zeros(num_modes, num_steps);
    for j = 1:num_modes
        amp_history(j, :) = abs(u_data(j+1, :)) / Nx;
    end
    
    % Theoretical eigenvalues
    k_hat = @(j) j / L;
    lambda_j = @(j, mu) (k_hat(j)^2) * (mu - 3*mass^2 - k_hat(j)^2);
    
    colors = lines(num_modes);
    figure('Name', 'Exact Resonance Analysis', 'Position', [150, 150, 1200, 800]);
    
    % --- Panel 1: Mode Amplitudes vs mu ---
    ax1 = subplot(2, 1, 1);
    hold(ax1, 'on'); set(ax1, 'YScale', 'log'); grid(ax1, 'on');
    title(ax1, sprintf('Mode Amplitudes (m = %.3f) - Detecting Resonance Zero-Crossings', mass), 'Interpreter', 'latex');
    xlabel(ax1, '$\mu(t)$', 'Interpreter', 'latex'); 
    ylabel(ax1, '$|\widehat{u}_j|$', 'Interpreter', 'latex');
    
    candidates = generate_driver_candidates(num_modes);
    
    for j = 1:num_modes
        u_j = amp_history(j, :);
        log_u = log(max(u_j, 1e-16));
        
        % Plot amplitude
        semilogy(ax1, mu_grid, u_j, 'LineWidth', 2, 'Color', colors(j, :), ...
            'DisplayName', sprintf('Mode %d', j));
        
        % Find downward spikes (local minima) using robust prominence
        spike_logical = islocalmin(log_u, 'MinProminence', 0.5);
        spike_indices = find(spike_logical);
        
        for sp = spike_indices
            t_spike = t_grid(sp);
            mu_spike = mu_grid(sp);
            
            % Estimate post-spike slope IN TIME (d/dt) using a forward window
            win = min(num_steps, sp + 10);
            if win - sp < 3, continue; end
            
            p_fit = polyfit(t_grid(sp+2:win), log_u(sp+2:win), 1);
            measured_slope_t = p_fit(1); % Measured d(ln|u_j|)/dt
            
            % Find which combination of eigenvalues matches this slope
            best_driver = "Homogeneous";
            best_err = abs(measured_slope_t - lambda_j(j, mu_spike));
            
            for c = 1:length(candidates)
                if candidates{c}.target ~= j, continue; end
                
                % Driver slope = sum(lambda_k)
                driver_rate = sum(arrayfun(@(k) lambda_j(abs(k), mu_spike), candidates{c}.indices));
                err = abs(measured_slope_t - driver_rate);
                
                if err < best_err
                    best_err = err;
                    best_driver = candidates{c}.label;
                end
            end
            
            % Annotate spike
            plot(ax1, mu_spike, u_j(sp), 'v', 'MarkerFaceColor', 'w', ...
                'MarkerEdgeColor', colors(j, :), 'MarkerSize', 8, 'HandleVisibility', 'off');
            text(ax1, mu_spike, u_j(sp)*0.2, sprintf(' $\\leftarrow$ %s', best_driver), ...
                'FontSize', 12, 'Color', colors(j, :), 'Interpreter', 'latex');
        end
    end
    legend(ax1, 'Location', 'northwest', 'Interpreter', 'latex');
    
    % --- Panel 2: Measured Slopes vs Theoretical Linear Rates ---
    ax2 = subplot(2, 1, 2);
    hold(ax2, 'on'); grid(ax2, 'on');
    title(ax2, 'Instantaneous Growth Rates: $\frac{d}{dt} \ln|\widehat{u}_j|$ vs Theoretical $\sum \lambda_k$', 'Interpreter', 'latex');
    xlabel(ax2, '$\mu(t)$', 'Interpreter', 'latex'); 
    ylabel(ax2, 'Growth Rate', 'Interpreter', 'latex');
    
    for j = 1:num_modes
        u_j = amp_history(j, :);
        log_u = log(max(u_j, 1e-16));
        
        % Numerical derivative with respect to TIME (dt)
        dt_grid = gradient(t_grid);
        d_log_u_dt = gradient(log_u) ./ dt_grid;
        
        % Plot empirical derivative
        plot(ax2, mu_grid, d_log_u_dt, 'LineWidth', 1.5, 'Color', colors(j, :), ...
            'DisplayName', sprintf('Empirical Rate Mode %d', j));
        
        % Plot base linear eigenvalue for reference
        nat_slope = arrayfun(@(m) lambda_j(j, m), mu_grid);
        plot(ax2, mu_grid, nat_slope, ':', 'Color', colors(j, :), 'LineWidth', 2, ...
            'HandleVisibility', 'off');
    end
    
    % Plot a theoretical reference line for the 1+2->3 resonance
    if num_modes >= 3
        res_slope_1_2 = arrayfun(@(m) lambda_j(1, m) + lambda_j(2, m), mu_grid);
        plot(ax2, mu_grid, res_slope_1_2, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5, ...
            'DisplayName', 'Theoretical $\lambda_1 + \lambda_2$ (Driver for Mode 3)');
    end
    
    legend(ax2, 'Location', 'southwest', 'Interpreter', 'latex');
    linkaxes([ax1, ax2], 'x');
end

function candidates = generate_driver_candidates(max_mode)
    % Generates all quadratic (2-body) and cubic (3-body) driving combinations
    candidates = {};
    idx = 1;
    
    % Quadratic Candidates: p + q = j or p - q = j
    for j = 1:max_mode
        for p = 1:max_mode
            for q = -max_mode:max_mode
                if q == 0, continue; end
                if (p + q) == j
                    candidates{idx}.target = j; %#ok<*AGROW>
                    candidates{idx}.indices = [p, q];
                    if q > 0
                        candidates{idx}.label = sprintf('%d+%d \\to %d', p, q, j);
                    else
                        candidates{idx}.label = sprintf('%d-%d \\to %d', p, abs(q), j);
                    end
                    idx = idx + 1;
                end
            end
        end
    end
    
    % Cubic Candidates: p + q + r = j
    for j = 1:max_mode
        for p = 1:max_mode
            for q = 1:max_mode
                for r = 1:max_mode
                    if (p + q + r) == j
                        candidates{idx}.target = j;
                        candidates{idx}.indices = [p, q, r];
                        candidates{idx}.label = sprintf('%d+%d+%d \\to %d', p, q, r, j);
                        idx = idx + 1;
                    end
                end
            end
        end
    end
end