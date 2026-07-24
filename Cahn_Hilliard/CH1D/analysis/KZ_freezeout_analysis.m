clc; clear; close all;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Description
% Find how domain wall density corresponds with different values (freezeout time, largest modes, etc.)
% and how theoretical values compare with computed values
% analyzing averages for diff. epsilon ( plot theoretical vs. computed values of density, freezout time, and largest wave mode )

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Parameters & Setup
epsilon_list = 10.^(linspace(-4, 0, 50));
plot_diff = false;

t0 = -2.0;
L = 100;
Lx = L*pi;
Nx = 2^12;
dx = Lx/Nx;

x = (-Nx/2:Nx/2-1)*dx;
kx = 2*pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
Laplacian_hat = -kx.^2;
L_operator = -(Laplacian_hat.^2);

avg_num = 10;
sigma = 0.01;

%% Storage for final plots
eps_out = epsilon_list;

density_comput = zeros(size(epsilon_list));
density_theory = zeros(size(epsilon_list));
density_diff = zeros(size(epsilon_list));

freezeout_time_comput = zeros(size(epsilon_list));
freezeout_time_theory = zeros(size(epsilon_list));
freezeout_time_diff = zeros(size(epsilon_list));

k_theory = zeros(size(epsilon_list));
k_comput = zeros(size(epsilon_list));
k_diff = zeros(size(epsilon_list));

%% Define the Evolution Problem
% For type 3, mu(t) = t*epsilon is linear with respect to epsilon
mu = @(t, eps_val) t * eps_val;

% Define the nonlinear operator handle without dealiasing mask (as in original type = 3)
nonlin_op = @(u_hat, t, eps_val) Laplacian_hat .* ...
    fft( real(ifft(u_hat)).^3 - mu(t, eps_val)*real(ifft(u_hat)) );

%% Setup the Solver
% Using 32 complex mean points as in the original type = 3 ETDRK4 formulation
solver = ETDRK4Solver(32);

%% Execute the Solve
for eidx = 1:length(epsilon_list)

    epsilon = epsilon_list(eidx);

    blowup_time = (epsilon/2)^(-2/3);
    T = min(blowup_time + 60.0, 150);

    blowup_time = (epsilon/2)^(-2/3);
    dt = min(0.01, 0.01 * blowup_time);

    t = t0:dt:T;
    num_time_steps = length(t);

    %% tracking average for each epsilon
    density_comput_avg = zeros(1,num_time_steps);
    density_theory_avg = zeros(1,num_time_steps);

    freezeout_time_comput_avg = 0;
    freezeout_time_theory_avg = 0;

    usq_avg = zeros(Nx,1);

    %% average runs for each epsilon
    for run_idx = 1:avg_num

        %% initial condition
        rng(100000*eidx + run_idx,'twister');

        u = sigma * randn(1,Nx);
        u_hat = fft(u);

        %% tracking once for each epsilon
        density_comput_eps = zeros(1,num_time_steps);
        density_theory_eps = zeros(1,num_time_steps);

        freezeout_time_comput_eps = 0;
        freezeout_time_theory_eps = blowup_time;

        %% Instantiate problem and solve via built-in ETD-RK4 solver
        curr_nonlin_op = @(u_hat_val, t_val) nonlin_op(u_hat_val, t_val, epsilon);
        problem = EvolutionProblem(L_operator, curr_nonlin_op, u_hat, [t0, T]);
        
        sol = evolution_solve(problem, solver, dt, save_every=1);

        %% Extract solved trajectory data
        u_data = sol.solution;
        if size(u_data, 1) == num_time_steps && size(u_data, 2) == Nx
            u_data = u_data';
        end
        t_solve = sol.time(:)';

        [~, freeze_idx] = min(abs(t_solve - blowup_time));

        %% diagnostics over solved time steps
        for n = 2:length(t_solve)

            t_curr = t_solve(n);
            u_hat_curr = u_data(:, n).';
            u_curr = real(ifft(u_hat_curr));

            density_theory_eps(n) = domainwall_density_theory(t_curr, u_hat_curr, Laplacian_hat, epsilon);
            density_comput_eps(n) = domainwall_density_comput(u_curr, Lx);

            mask = t_solve > 10;
            density_masked = density_comput_eps(mask);

            max_density = max(density_masked);
            tol = 1e-4*max_density;

            mask_idx = find(mask);
            freeze_idx_local = find(density_masked >= max_density - tol, 1, 'first');

            if ~isempty(freeze_idx_local)
                freeze_idx = mask_idx(freeze_idx_local);
                freezeout_time_comput_eps = t_solve(freeze_idx);
            end

            if n == freeze_idx
                usq_avg = usq_avg + (abs(u_hat_curr(:)).^2) / avg_num;
            end

        end

        %% running averages
        density_comput_avg = density_comput_avg + density_comput_eps/avg_num;
        density_theory_avg = density_theory_avg + density_theory_eps/avg_num;

        freezeout_time_comput_avg = freezeout_time_comput_avg + freezeout_time_comput_eps/avg_num;
        freezeout_time_theory_avg = freezeout_time_theory_avg + freezeout_time_theory_eps/avg_num;

        k_theory(eidx) = (epsilon/2)^(1/6);

    end

    %% POST PROCESS PER EPSILON
    [~,cut] = min(abs(t - blowup_time));

    density_comput(eidx) = density_comput_avg(cut);
    density_theory(eidx) = density_theory_avg(cut);
    density_diff(eidx) = density_theory_avg(cut) - density_comput_avg(cut);

    freezeout_time_comput(eidx) = freezeout_time_comput_avg;
    freezeout_time_theory(eidx) = freezeout_time_theory_avg;
    freezeout_time_diff(eidx) = freezeout_time_theory_avg - freezeout_time_comput_avg;

    kshift = fftshift(kx);

    [~, freeze_idx] = min(abs(t - blowup_time));
    if n == freeze_idx
        usq_avg = usq_avg + (abs(u_hat(:)).^2) / avg_num;
    end

    spec = fftshift(usq_avg);

    [~,idx] = max(spec);

    k_comput(eidx) = abs(kshift(idx));
    k_diff(eidx) = k_theory(eidx) - k_comput(eidx);

    fprintf("epsilon %.0f/%.0f = %.3f done\n", eidx, length(epsilon_list), epsilon);

end

%% Post-Processing
if ~plot_diff
    % Fit a line to log10(epsilon) vs log10(computed freeze-out time)
    valid_mask = freezeout_time_comput > 0; % Ensure no zero/NaN values break the log
    log_eps = log10(eps_out(valid_mask));
    log_t   = log10(freezeout_time_comput(valid_mask));

    % 1st order polynomial fit (y = m*x + b)
    p = polyfit(log_eps, log_t, 1);
    slope_computed = p(1);       % Should be very close to -0.6667
    intercept_computed = p(2);

    % Generate fitted line for plotting
    fit_line = 10.^(polyval(p, log10(eps_out)));
end

%% Plotting
figure;

if plot_diff

    subplot(1,3,1)
    plot(eps_out, density_diff);
    xlabel('\epsilon');
    ylabel('n(\epsilon)');
    title('Domain Wall Density at Freeze-Out');
    grid on;

    subplot(1,3,2)
    loglog(eps_out,freezeout_time_diff,'o-');
    xlabel('\epsilon');
    ylabel('t(\epsilon)');
    title('Freeze-Out Time');
    grid on;

    subplot(1,3,3)
    plot(eps_out,k_diff,'o-');
    xlabel('\epsilon');
    ylabel('k_{max}');
    title('Highest-Growing Wave Modes');
    grid on;

else

    subplot(1,3,1)
    loglog(eps_out, freezeout_time_comput, 'o', 'DisplayName', 'Computed'); hold on;
    loglog(eps_out, freezeout_time_theory, 'k--', 'DisplayName', 'Theory (\alpha = -2/3)');
    loglog(eps_out, fit_line, 'r-', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Fit (slope = %.3f)', slope_computed));
    xlabel('\epsilon');
    ylabel('Freeze-Out Time \hat{t}');
    title('Freeze-Out Time Scaling');
    legend('Location', 'northeast');
    grid on;

    subplot(1,3,1)
    plot(eps_out,freezeout_time_comput,'o-'); hold on;
    plot(eps_out,freezeout_time_theory,'k--');
    xlabel('\epsilon');
    ylabel('freeze-out time');
    title('Freeze-Out Time');
    legend('computed','theoretical');
    grid on;

    subplot(1,3,2)
    loglog(eps_out,density_comput,'o-'); hold on;
    loglog(eps_out,density_theory,'s-');
    xlabel('\epsilon');
    ylabel('Domain wall density');
    title('Domain Wall Density');
    legend('computed','theoretical', 'Location', 'northwest');
    grid on;

    subplot(1,3,3)
    plot(eps_out,k_comput,'o-'); hold on;
    plot(eps_out,k_theory,'s-');
    xlabel('\epsilon');
    ylabel('k_{max}');
    title('Highest-Growing Wave Modes');
    legend('computed','theoretical');
    grid on;

end


%% FUNCTIONS

function mu_val = mu_calculate(t, epsilon)
    mu_val = t*epsilon;
end

function P = power_spec(t,u_hat,Laplacian_k,epsilon)
    ksq = -Laplacian_k;
    P = abs(u_hat).^2 .* exp(t^2*ksq*epsilon - 2*t*ksq.^2);
end

function n = domainwall_density_theory(t,u_hat,Laplacian_k,epsilon)
    P = power_spec(t,u_hat,Laplacian_k,epsilon);
    ksq = -Laplacian_k;

    n = (1/pi)*sqrt(sum(ksq.*P)/sum(P));
end

function n = domainwall_density_comput(u,Lx)
    walls = sum(u .* circshift(u,-1) < 0);
    n = walls/Lx;
end

function t_hat = theoretical_freezeout_time(epsilon,kx)
    sigma = 0.01;
    P0 = sigma^2*ones(size(kx));
    dk = abs(kx(2)-kx(1));

    f = @(t) freezeout_equation(t,epsilon,kx,P0,dk);

    t_guess = 2^(2/3)*epsilon^(-2/3);
    t_hat = fzero(f,t_guess);
end

function val = freezeout_equation(t,epsilon,kx,P0,dk)
    arg = epsilon*(kx.^2)*t.^2 - 2*(kx.^4)*t;
    arg = min(arg,700);

    val = 3*sum(P0 .* exp(arg))*dk - epsilon*t;
end

function C = fit_freezeout_constant(epsilon_list,t_hat_list)
    x = epsilon_list(:).^(-2/3);
    y = t_hat_list(:);

    C = (x'*y)/(x'*x);
end

function val = l2_norm_periodic_1D(u_hat,Lx)
    Nx = length(u_hat);
    val = sqrt(Lx)* sqrt(sum(abs(u_hat).^2))/Nx;
end