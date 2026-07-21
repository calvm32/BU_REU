clear all;
close all;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Description and configuration
% Blueprint and solver of CH w/ mu ramped linearly

% 0: just plot once, for a single epsilon value
% 1: plot over a range of epsilon values

type = 0;

% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --%
    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    % % Parameters dt = 0.1;
L = 10;
Lx = L * pi;
Nx = 2 ^ 10;
epsilon = 0.001;

% max.subcritical mode n_subc = 3;
% 1, ..., n modes = subcritical, n + 1,
            ... modes = supercritical mass = (2 * n_subc + 1) * sqrt(2) / 20;

% if starting with a single dominant mode : k_init = 1;

% max.mode whose eigenvalue is > 0 % n_mu = 2;
% mu_n = 3 * mass ^ 2 + (n_mu / L) ^ 2;
% mu_next = 3 * mass ^ 2 + ((n_mu + 1) / L) ^ 2;
mu0 = 0;
% (mu_n + mu_next) / 2;

disp(sprintf('using mass    = %.4f', mass))
    disp(sprintf('using mu0     = %.4f', mu0))

        k_hat_sq = @(k)(k ^ 2) / (L ^ 2);
lambda_k = @(k, mu0, mass) k_hat_sq(k) * (-k_hat_sq(k) + mu0 - mass ^ 2);
lin_term = @(k, mu0, mass) - k_hat_sq(k) ^ 2 - 3 * k_hat_sq(k) * mass ^ 2 + mu0;
T_k = @(k, mu0, mass,
        ep)(-lin_term(k, mu0, mass) +
            sqrt(lin_term(k, mu0, mass) ^ 2 + 2 * log(10) * ep * k_hat_sq(k))) /
      (ep * k_hat_sq(k));
% time at which IC mode, when dominant,
    starts growing past starting value

            disp(sprintf('starting mode = %.0d', k_init))
                disp(sprintf('lambda(k)     = %.4d',
                             lambda_k(k_init, mu0, mass)))
                    disp(sprintf('freeze-out    = %.4f',
                                 T_k(k_init, mu0, mass, epsilon)))

        % spatial grid dx = 2 * Lx / Nx;
x = (-Nx / 2 : Nx / 2 - 1) * dx;

l2_thr = 1e0;
%
    when to say the solution has become large amplitude nonlinear
        quasistationary state

    % % Video paramaters show_video = true;
plot_every = 30;
plot_dmu = 0.005;

% % Fourier wavenumbers + linear operator kx = pi * [0:Nx / 2 - 1 - Nx /
                                                     2:-1] / Lx;
Laplacian_hat = -kx.^ 2;
L_operator = -(Laplacian_hat.^ 2);

kx_max = max(abs(kx));
dealias_mask = abs(kx) <= (2 / 3) * kx_max;
k_j = pi * [0:10] / Lx;

% % Initial Conditions

    % % Zero mean white noise % noise = 0.2 * (rand(1, Nx) - 0.5);
% noise = noise - mean(noise);

% % Zero mean white noise % u0 = mass + noise;

% % Start at a pure mode % u0 = 0.1 * cos(k_init * x) + mass;
% u_hat0 = fft(u0);

% % all modes on % uh0 = ones(size(x));
% u0 = ifft(uh0);
% u0 = u0 - mean(u0);

u_hat0 = ones(1, Nx);
u_hat0(1) = mass * Nx;

switch
type case 0 ep = epsilon;

    % muf = critical_bifurcation_eig(20, L, mass); % T = (muf - mu0) / ep;
    T = T_k(k_init, mu0, mass, ep) + 4000; mu = @(t) mu0 + t * ep;

    % Define problem % Nonlinear operator:
  dealias_mask.*Laplacian_hat.*
      fft(u ^ 3 - mu * u) nonlin_op = @(u_hat, t) dealias_mask.*Laplacian_hat.*
                                      fft(real(ifft(u_hat)).^
                                          3 - mu(t) * real(ifft(u_hat)));
  problem = EvolutionProblem(L_operator, nonlin_op, u_hat0, [ 0, T ]);

% Solver solver = ETDRK4Solver(32);

% Use BifurcationMonitorHeadless if you don't want plotting/video: % monitor =
    BifurcationMonitorHeadless(mu, kx, Lx, Nx);
% Use BifurcationMonitorInMu if you do : % monitor = BifurcationMonitorInMu(
    mu, kx, Lx, Nx, x, k_j, plot_every = plot_every, save_video = save_video,
    video_filename = sprintf('ep=%.5f_mu0=%.2f.avi', ep, mu0),
    video_framerate = 30);
monitor = BifurcationMonitorInMuModeHistory(
    mu, kx, Lx, Nx, x, k_j, ... plot_every = plot_every,
    ... save_video = show_video,
    ... video_filename = sprintf('ep=%.5f_mu0=%.2f.avi', ep, mu0),
    ... video_framerate = 30, subtract_mass = true, local_mass = mass);

% Execute sol = evolution_solve(problem, solver, dt, save_every = plot_every,
                                monitors = monitor);

% Find high amplitude time j_thr = find(monitor.l2_history > l2_thr, 1);
t_thr = monitor.t_grid(j_thr);
mu_thr = mu(t_thr);

% Find dominate mode at high amplitude dominate_modes =
    monitor.dominate_mode(1 : monitor.dom_mode_ind, :);
j_dom_mode_at_thr = find(dominate_modes( :, 1) < t_thr, 1, 'last');
time_dom_mode_at_thr = dominate_modes(j_dom_mode_at_thr, 1);
dom_mode = dominate_modes(j_dom_mode_at_thr, 2);
dom_mode_final = dominate_modes(end, 2);

        % MUS = 

        disp()

    case 1

        %% Save the results of each run (including solution) individually, and/ or results of all runs at once
        save_each_run = false;
        save_all_runs = false;

        % range of all epsilons (valid for type 1)
        EP = 1.15*10.^[-5]; %10.^[-4.5:1:-4];

        % store dominant mode and mu threshold for each epsilon
        DMODE = zeros(length(EP),1);
        DMODE_final = zeros(length(EP), 1);
        MUTHR = zeros(length(EP), 1);

        %% loop over range of epsilons
        for ii = 1:length(EP)
            ep = EP(ii);
        disp([ 'Running simulation for epsilon = ', num2str(ep) ]);
        if ep
          < 1e-3 muf_temp = 1.0;
        else
          muf_temp = muf;
        end muf_temp = min(muf_temp, muf);

        [ MUS, ~] = critical_bifurcation_inteig([1:10], 2 * Lx, mass, mu0);
        display(MUS)

            T = (muf_temp - mu0) / ep;
        % muf = mu0 + ep * T;
        % T = (muf - mu0) / ep

                                plot_every = round(plot_dmu / ep / dt);
        % make multiple of dt

                mu = @(t) mu0 + t * ep;

        % Define problem %
            Nonlinear operator
            : dealias_mask.*Laplacian_hat.*fft(u ^ 3 - mu * u) nonlin_op =
            @(u_hat, t) dealias_mask.*Laplacian_hat.*
            fft(real(ifft(u_hat)).^ 3 - mu(t) * real(ifft(u_hat)));
        problem = EvolutionProblem(L_operator, nonlin_op, u_hat0, [ 0, T ]);

        % Solver solver = ETDRK4Solver(16);

        % Use BifurcationMonitorHeadless if you don't want plotting/video: %
            monitor = BifurcationMonitorHeadless(mu, kx, Lx, Nx);
        % Use BifurcationMonitorInMu if you do : % monitor =
            BifurcationMonitorInMu(
                mu, kx, Lx, Nx, x, k_j, plot_every = plot_every,
                save_video = save_video,
                video_filename = sprintf('ep=%.5f_mu0=%.2f.avi', ep, mu0),
                video_framerate = 30);
        monitor = BifurcationMonitorInMuModeHistory(
            mu, kx, Lx, Nx, x, k_j, ... plot_every = plot_every,
            ... save_video = save_each_run,
            ... video_filename = sprintf('ep=%.5f_mu0=%.2f.avi', ep, mu0),
            ... video_framerate = 30);

        % Execute sol = evolution_solve(
            problem, solver, dt, save_every = plot_every, monitors = monitor);

        % Find high amplitude time j_thr = find(monitor.l2_history > l2_thr, 1);
        t_thr = monitor.t_grid(j_thr);
        mu_thr = mu(t_thr);

        % Find dominate mode at high amplitude dominate_modes =
            monitor.dominate_mode(1 : monitor.dom_mode_ind, :);
        j_dom_mode_at_thr = find(dominate_modes( :, 1) < t_thr, 1, 'last');
        time_dom_mode_at_thr = dominate_modes(j_dom_mode_at_thr, 1);
        dom_mode = dominate_modes(j_dom_mode_at_thr, 2);
        dom_mode_final = dominate_modes(end, 2);

        DMODE_final(ii) = dom_mode_final;
        DMODE(ii) = dom_mode;
        MUTHR(ii) = mu_thr;

        if save_each_run
          save(sprintf('dom_mode_ep=%.5f_mu0=%.2f.mat', ep, mu0), 'mu_thr',
               'dom_mode', 'dom_mode_final', 'ep', 'kx', 'Lx', 'mu0', 'muf',
               'l2_thr', 'mass', 'Nx', 'dt', 'T', 'plot_dmu', 'u0', 'sol');
        end

                end

            % Save collective run data if save_all_runs figure(20)
                  plot(EP, DMODE, '-o', 'LineWidth', 2)
                      save('dominate_modes_test.mat', 'MUTHR', 'DMODE',
                           'DMODE_final', 'EP', 'kx', 'Lx', 'mu0', 'muf',
                           'l2_thr', 'mass', 'Nx', 'dt', 'plot_dmu', 'u0');
        end

            end

                function MU =
                    critical_bifurcation_eig(k, L, mass) %
                    Computes the
                        j'th frequencies critical mu and associated eigenvalue MU =
                        3 * (mass) ^ 2 + (k / L) ^ 2;
        end

            function[MUS, KS] =
                critical_bifurcation_inteig(j, L, mass, mu0) %
                Computes the
                    j'th frequencies critical mu and associated eigenvalue KS =
                    2 * pi * j./ L;
        MUS = -mu0 + 2 * (3 * mass ^ 2 + KS.^ 2);
        end