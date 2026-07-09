clc; clear; close all;

% Parameters
epsilon = 0.05;
alpha = 1.0; 
beta = 0.0;
m = 0.0; % mass

% KZ time-scales
freeze_out_time = epsilon^(-2/3);
t0 = -2.0 * freeze_out_time; 
T = freeze_out_time + 100.0;
dt = 0.05;

tau = t0-100; % start time for calculating omega limit sets

% Domain setup [-L*pi, L*pi]
scale = 100;
Lx = 2 * scale * pi;
Nx = 2^11;
dx = Lx/Nx;
plot_dt = 1.0; 
plot_every = round(plot_dt / dt);

x = (-Nx/2:Nx/2-1)*dx;
kx = 2*pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
Laplacian_hat = -kx.^2;
L_operator = -(Laplacian_hat.^2);
kx_max = max(abs(kx));
dealias_mask = abs(kx) <= (2/3)*kx_max;

% Rigorous bounds (lemmas 1-2)
kappa_1 = 2 / (1+scale^2)^2;
C_ab = max(1-alpha-beta, 1+alpha-beta)^2*Lx/4;
C_m = Lx*3^7*m^4/(2^4);
kappa_2 = C_m + 4*C_ab;

% Global asymptotic bounds (t0 -> -infinity)
rho_0_global = sqrt(kappa_2 / kappa_1);
R_0_global = sqrt((C_m + 4*C_ab + rho_0_global^2)/2);

fprintf('Mathematical Bounds:\n');
fprintf('kappa_1 = %.4e, kappa_2 = %.4e\n', kappa_1, kappa_2);
fprintf('Global H-1 Absorbing Radius (rho_0) = %.4e\n', rho_0_global);
fprintf('Global H1 Absorbing Radius (R_0) = %.4e\n', R_0_global);

% KZ subspace
k_hat = epsilon^(1/6); 
[~, k_idx1] = min(abs(kx(1:Nx/2) - k_hat));
[~, k_idx2] = min(abs(kx(1:Nx/2) - k_hat*1.5)); 

%% Create absorbing balls of H1 norm
num_samples = 40; 
theta = linspace(0, 2*pi, num_samples);

% initialize larger H1 radius to capture behavior
R_init = 0.3*R_0_global;

U0_hat_ensemble = zeros(num_samples, Nx);
initial_H1_norms = zeros(num_samples, 1);
initial_Hinv_norms = zeros(num_samples, 1);

for i = 1:num_samples
    u_hat = zeros(1, Nx);
    
    % Enforce H^1 norm constraint: ||u||_{H^1}^2 ~ sum (1+k^2)|u_k|^2
    % divide the radius by sqrt(1+k^2) to make the circle isometric in H^1
    amp1 = (R_init * cos(theta(i)) * Nx) / sqrt(1 + kx(k_idx1)^2);
    amp2 = (R_init * sin(theta(i)) * Nx) / sqrt(1 + kx(k_idx2)^2);
    
    u_hat(k_idx1) = amp1;
    u_hat(k_idx2) = amp2;
    u_hat(Nx - k_idx1 + 2) = u_hat(k_idx1);
    u_hat(Nx - k_idx2 + 2) = u_hat(k_idx2);
    
    % Calculate exact discrete norms for tracking
    u_real = real(ifft(u_hat));
    ux_real = real(ifft(1i * kx .* u_hat));
    initial_H1_norms(i) = sum(u_real.^2 + ux_real.^2) * dx;
    
    % H-1 norm (ignoring mean)
    k_inv = kx; k_inv(1) = 1;
    u_Hinv_real = real(ifft(u_hat ./ (1i * k_inv)));
    initial_Hinv_norms(i) = sum(u_Hinv_real.^2) * dx;
end

U0_hat_ensemble(:,1) = m*Nx; % enforce mass = m

% initial H-1 norm to feed into the dynamic bound R(t; t0)
max_init_Hinv2 = max(initial_Hinv_norms);

%% Evolution
mu = @(t) alpha * tanh(epsilon * t) + beta;
nonlin_op = @(u_hat, t) dealias_mask .* Laplacian_hat .* ...
    fft( real(ifft(u_hat)).^3 - mu(t)*real(ifft(u_hat)) );

solver = ETDRK4Solver(16);

disp('Evolving the process U(t, tau) mapping B(tau) -> A(t)...');
ensemble_solutions = cell(num_samples, 1);

noise_amplitude = 1e-5; % ball radius at t0

for i = 1:num_samples
    % find the PBA as tau -> t_0
    prob1 = EvolutionProblem(L_operator, nonlin_op, U0_hat_ensemble(i, :), [tau, t0]);
    sol1 = evolution_solve(prob1, solver, dt, save_every=plot_every, monitors=EmptyMonitor.empty(1,0));
    
    u_t0 = sol1.solution(:, end).';
    
    % Solve anew at t0, establishing a finite tracking ball
    noise_amp1 = (noise_amplitude * cos(theta(i)) * Nx) / sqrt(1 + kx(k_idx1)^2);
    noise_amp2 = (noise_amplitude * sin(theta(i)) * Nx) / sqrt(1 + kx(k_idx2)^2);
    
    % to get nonzero, perturb slightly
    u_t0(k_idx1) = u_t0(k_idx1) + noise_amp1;
    u_t0(k_idx2) = u_t0(k_idx2) + noise_amp2;
    u_t0(Nx - k_idx1 + 2) = u_t0(k_idx1); 
    u_t0(Nx - k_idx2 + 2) = u_t0(k_idx2);
    
    % Evolve from t0 -> T
    prob2 = EvolutionProblem(L_operator, nonlin_op, u_t0, [t0, T]);
    sol2 = evolution_solve(prob2, solver, dt, save_every=plot_every, monitors=EmptyMonitor.empty(1,0));
    
    sol_combined.time = [sol1.time, sol2.time(2:end)];
    sol_combined.solution = [sol1.solution, sol2.solution(:, 2:end)];
    ensemble_solutions{i} = sol_combined;
end

t_data = ensemble_solutions{1}.time;
num_steps = length(t_data);

% t >= t0
start_step = find(t_data >= t0, 1);
if isempty(start_step)
    start_step = 1; 
end

%% Visualization
figure('Position', [50, 100, 1400, 450]);

video_filename = 'KZ_Pullback_Attractor.avi';
v = VideoWriter(video_filename);
v.FrameRate = 15;
open(v);

disp(['Recording animation to ', video_filename, '...']);

for step = start_step:num_steps
    t_current = t_data(step);
    
    % Extract current states
    k1_vals = zeros(1, num_samples);
    k2_vals = zeros(1, num_samples);
    current_H1_norms = zeros(1, num_samples);
    
    for i = 1:num_samples
        u_hat_current = ensemble_solutions{i}.solution(:, step).';
        k1_vals(i) = real(u_hat_current(k_idx1)) / Nx;
        k2_vals(i) = real(u_hat_current(k_idx2)) / Nx;
        
        u_real = real(ifft(u_hat_current));
        ux_real = real(ifft(1i * kx .* u_hat_current));
        current_H1_norms(i) = sum(u_real.^2 + ux_real.^2) * dx;
    end
    
    % Subplot 1: PBA projection onto wave modes
    subplot(1,3,1);
    plot(k1_vals, k2_vals, 'b-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
    max_val = max(max(abs(k1_vals)), max(abs(k2_vals))) + 1e-8; % Added small epsilon to avoid 0 limit
    xlim([-max_val, max_val]); ylim([-max_val, max_val]);
    title('Pullback Attractor $A(t)$', 'Interpreter', 'latex');
    xlabel(sprintf('KZ Mode (k_hat \\approx %.2f)', kx(k_idx1)));
    ylabel(sprintf('Neighboring Mode (k \\approx %.2f)', kx(k_idx2)));
    grid on; axis square;
    
    % Subplot 2: H^1 norm theory vs. computed
    subplot(1,3,2);

    dynamic_rho2 = max_init_Hinv2 * exp(kappa_1*(t0 - t_current)) + (kappa_2/kappa_1)*(1 - exp(kappa_1*(t0 - t_current)));
    dynamic_R = sqrt((C_m + dynamic_rho2 + 4*C_ab)/2);
    
    plot(1:num_samples, current_H1_norms, 'k-o', 'LineWidth', 1.5);
    yline(dynamic_R, 'r--', 'LineWidth', 2); 
    title(sprintf('Norm ||v||_{H^1} vs. Time')); % \nDynamic Bound R(t) \\approx %.2e', dynamic_R
    xlabel('Norm index');
    ylabel('||v||_{H^1}');
        
    max_H1 = max(current_H1_norms);
    upper_limit = max(max_H1 * 1.5, dynamic_R * 1.2); % Takes the larger of the two
    if upper_limit < 1e-9
        upper_limit = 1e-9;
    end
    ylim([0, upper_limit]);
    ylim([0, upper_limit]); 
    grid on; axis square;
    
    % Subplot 3: Solution
    subplot(1,3,3);
    u_real_samp = real(ifft(ensemble_solutions{1}.solution(:, step).'));
    plot(x, u_real_samp, 'k-', 'LineWidth', 1.5);
    title(sprintf('Solution')); %sprintf('Fourier Spectrum\nFreeze-out t_hat \\approx %.2f', freeze_out_time)
    xlabel('x'); 
    ylabel('u(x,t)');
    ylim([-1.2, 1.2]); 
    grid on; axis square;
    
    sgtitle(sprintf('Time t = %.2f', t_current), 'FontSize', 16, 'FontWeight', 'bold');
    drawnow;
    
    % Capture the frame and write to the video
    frame = getframe(gcf);
    writeVideo(v, frame);
end

% Close the video file
close(v);
disp('Visualization complete. Video saved successfully.');