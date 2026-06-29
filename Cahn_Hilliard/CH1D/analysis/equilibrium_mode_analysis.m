clc; clear; close all;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Description and configuration
% Find how different ICs and epsilon values affect equilibrium mode

% 0: single run  ( solution | fourier spectrum / dominant mode | L2 norm )
% 1: single mode IC ( plot x = IC mode VS. y = Eq. mode VS. color = probability )
% 2: single mode IC + vary epsilon ( plot 1 but varying in time = epsilon )
% 3: random IC ( plot x = epsilon VS. y = Eq. mode VS. color = probability )

type = 2;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Parameters
% heat map params
IC_modes = 0:10;
num_runs = 20;
xscale = 8;

epsilon = 0.025; % type == 0 or 1
epsilon_list = log(linspace(1, 3, 50)); % type == 2 or 3

% simulation params
t0 = -10.0;

mu_final = 2.0;
T = round(mu_final / epsilon);
dt = 0.1;
mu = @(t) epsilon * t;

mass = 0;

Lx = xscale*pi;
Nx = 2^10;
dx = 2*Lx/Nx;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% discretization
% time discretization
t = t0:dt:T;
num_time_steps = length(t);

% spatial grid (needed for patterned base state)
x = (-Nx/2:Nx/2-1)*dx;

%% Fourier wavenumbers
kx = pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
Laplacian_k = -kx.^2;

% Linear operator:
L_operator = -(Laplacian_k.^2);

%% 2/3 dealiasing mask
kx_max = max(abs(kx));
dealias_mask = abs(kx) <= (2/3)*kx_max;

%% ETDRK4 setup
E  = exp(dt*L_operator);
E2 = exp(dt*L_operator/2);
M = 16; % no. of points for complex means
r = exp(1i*pi*((1:M)-0.5)/M); % roots of unity
Lvec = L_operator(:);
LR = dt*Lvec(:,ones(M,1)) + r(ones(numel(L_operator),1),:);

Q  = dt*real(mean((exp(LR/2)-1)./LR,2)).';
f1 = dt*real(mean((-4-LR + exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2)).';
f2 = dt*real(mean((2+LR + exp(LR).*(-2+LR))./LR.^3 ,2)).';
f3 = dt*real(mean((-4-3*LR-LR.^2 + exp(LR).*(4-LR))./LR.^3 ,2)).';

clear LR Lvec

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

switch type

    % ------------------------------------------------------------------------------------------
    % ------------------------------------------------------------------------------------------

    case 0

        k0 = 1;
        u0 = mass + cos(2*pi*k0*x/(2*Lx));

        u_hat = fft(u0);
        u = u0;

        % Pre-allocate history arrays
        l2_hist   = zeros(1, num_time_steps);
        kdom_hist = zeros(1, num_time_steps);

        [~, max_index] = max(abs(fft(u0 - mass)));
        kdom_hist(1) = kx(max_index);
        l2_hist(1)   = sqrt(sum(u0.^2) * dx);

        %% Time loop
        for n = 2:num_time_steps
            t_prev = t(n-1);
            t_half = t_prev + dt/2;
            t_curr = t(n);

            u3_nonlinear = u.^3 - mu(t_prev)*u;
            Nu_hat = dealias_mask .* Laplacian_k .* fft(u3_nonlinear);

            a_hat = E2.*u_hat + Q.*Nu_hat;
            a = real(ifft(a_hat));
            Na_hat = dealias_mask .* Laplacian_k .* fft(a.^3 - mu(t_half)*a);

            b_hat = E2.*u_hat + Q.*Na_hat;
            b = real(ifft(b_hat));
            Nb_hat = dealias_mask .* Laplacian_k .* fft(b.^3 - mu(t_half)*b);

            c_hat = E2.*a_hat + Q.*(2*Nb_hat - Nu_hat);
            c = real(ifft(c_hat));
            Nc_hat = dealias_mask .* Laplacian_k .* fft(c.^3 - mu(t_curr)*c);

            u_hat = E.*u_hat + f1.*Nu_hat + 2*f2.*(Na_hat + Nb_hat) + f3.*Nc_hat;
            u = real(ifft(u_hat));

            [~, max_index] = max(abs(fft(u - mass)));
            kdom_hist(n) = kx(max_index);
            l2_hist(n)   = sqrt(sum(u.^2) * dx);
        end

        % Equilibrium time
        t_eq = find_equilibrium_time(kdom_hist, l2_hist, t);
        if isnan(t_eq)
            fprintf('Equilibrium not detected within simulation window.\n');
        else
            fprintf('Equilibrium reached at t ≈ %.3f\n', t_eq);
        end

        %% Plotting — 2x2 layout
        % Reference wavenumbers
        k_ref = pi*(0:6)/Lx;

        % Fourier spectrum of final state (fftshift for display)
        u_hat_final  = fft(u - mass);
        kx_shifted   = fftshift(kx);
        uhat_shifted = max(fftshift(abs(u_hat_final)), 1e-65);

        figure('Position', [100 100 1200 800]);

        % Subplot 1 — solution u(x) at final time
        subplot(2,2,1)
        plot(x, u, 'LineWidth', 1.5, 'Color', [0.18 0.45 0.69])
        xlabel('x'); ylabel('u(x, T)')
        title(sprintf('Solution at t = %.2f', t(end)))
        grid on

        % Subplot 2 — Fourier spectrum (log scale)
        subplot(2,2,2)
        semilogy(kx_shifted, uhat_shifted, 'LineWidth', 1.5, 'Color', [0.85 0.33 0.10])
        xlim([-8 8])
        ylim([1e-14 max(uhat_shifted)*10])
        hold on
        for kj = k_ref
            xline(kj,  '--r', 'Alpha', 0.4)
            xline(-kj, '--r', 'Alpha', 0.4)
        end
        hold off
        xlabel('k'); ylabel('|\hat{u}(k)|')
        title('Fourier spectrum (final state)')
        grid on

        % Subplot 3 — dominant wavenumber vs time
        subplot(2,2,3)
        stairs(t, abs(kdom_hist), 'LineWidth', 2, 'Color', [0.49 0.18 0.56])
        hold on
        for kj = k_ref
            yline(kj, '--r', 'Alpha', 0.4)
        end
        if ~isnan(t_eq)
            xline(t_eq, '-g', 'LineWidth', 1.5, ...
                  'Label', sprintf('eq. t=%.1f', t_eq), 'LabelVerticalAlignment', 'bottom')
        end
        hold off
        xlabel('t'); ylabel('dominant |k|')
        title('Dominant wavenumber vs time')
        grid on

        % Subplot 4 — L2 norm vs time with equilibrium marker
        subplot(2,2,4)
        plot(t, l2_hist, 'LineWidth', 1.5, 'Color', [0.13 0.55 0.55])
        hold on
        if ~isnan(t_eq)
            xline(t_eq, '--g', 'LineWidth', 2, ...
                  'Label', sprintf('t_{eq}≈%.1f', t_eq), 'LabelVerticalAlignment', 'bottom')
            eq_idx = find(t >= t_eq, 1);
            plot(t(eq_idx), l2_hist(eq_idx), 'og', 'MarkerSize', 8, 'MarkerFaceColor', 'g')
        end
        hold off
        xlabel('t'); ylabel('||u||_2')
        title('L_2 norm vs time')
        grid on

        sgtitle(sprintf('Swift-Hohenberg  |  \\epsilon=%.4f  t_0=%.1f  mass=%.2f  IC mode=%d', ...
                        epsilon, t0, mass, k0))

        saveas(gcf, 'type0_single_run.png')

    % ------------------------------------------------------------------------------------------
    % ------------------------------------------------------------------------------------------

    case 1

        Heat = zeros(length(IC_modes), length(IC_modes));

        %% IC loop
        for ic = 1:length(IC_modes)

            k0 = IC_modes(ic);
            
            u0 = mass + cos(2*pi*k0*x/(2*Lx));

            for run = 1:num_runs

                u_hat = fft(u0);
                u = u0;
                    
                %% Time loop
                for n = 2:num_time_steps
                    t_prev = t(n-1);
                    t_half = t_prev + dt/2;
                    t_curr = t(n);
                    
                    u3_nonlinear = u.^3 - mu(t_prev)*u;
                    Nu_hat = dealias_mask .* Laplacian_k .* fft(u3_nonlinear);
                    
                    a_hat = E2.*u_hat + Q.*Nu_hat;
                    a = real(ifft(a_hat));
                    Na_hat = dealias_mask .* Laplacian_k .* fft(a.^3 - mu(t_half)*a);
                    
                    b_hat = E2.*u_hat + Q.*Na_hat;
                    b = real(ifft(b_hat));
                    Nb_hat = dealias_mask .* Laplacian_k .* fft(b.^3 - mu(t_half)*b);
                    
                    c_hat = E2.*a_hat + Q.*(2*Nb_hat - Nu_hat);
                    c = real(ifft(c_hat));
                    Nc_hat = dealias_mask .* Laplacian_k .* fft(c.^3 - mu(t_curr)*c);
                    
                    u_hat = E.*u_hat + f1.*Nu_hat + 2*f2.*(Na_hat + Nb_hat) + f3.*Nc_hat;
                    u = real(ifft(u_hat));
                end

                [~,max_index] = max(abs(fft(u-mass)));
                final_mode = round(kx(max_index)*Lx/pi);

                row = find(IC_modes==final_mode);

                if ~isempty(row)
                    Heat(row,ic) = Heat(row,ic) + 1;
                end

                fprintf('done w/ run=%.0f/%.0f; IC tested=%.0f/%.0f\n', run, num_runs, ic, length(IC_modes));
            end
        end

        %% plotting
        figure
        imagesc(IC_modes,IC_modes,Heat/num_runs)
        axis xy
        xlabel('IC mode number')
        ylabel('Eq, mode number')

        title('Likelihood of equilibrium modes reached for various initial conditions');
        text(0.5, -0.1, sprintf('Parameters: T = %.0f,  t0 = %.0f, \\epsilon = %.3f\nEach IC ran for %.0f trials', T, t0, epsilon, num_runs), 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontSize', 10);

        xticks(IC_modes(1)-0.5:1:IC_modes(end)+0.5);
        yticks(IC_modes(1)-0.5:1:IC_modes(end)+0.5);
        grid('on')
        GridColor = [0.5 0.5 0.5];

        levels = num_runs + 1;
        darkBlue = [0.02 0.12 0.55]; 
        white = [1 1 1];

        cmap = [linspace(darkBlue(1),white(1),levels)', ...
                linspace(darkBlue(2),white(2),levels)', ...
                linspace(darkBlue(3),white(3),levels)'];

        colormap(flipud(cmap))
        clim([0 1])
        colorbar

    % ------------------------------------------------------------------------------------------
    % ------------------------------------------------------------------------------------------

    case 2

        %% Parallel loop over epsilon
        num_eps = length(epsilon_list);
        num_IC = length(IC_modes);
        Heat_cell = cell(num_eps, 1);

        if isempty(gcp('nocreate'))
            parpool;
        end

        parfor eidx = 1:num_eps
            eps_val = epsilon_list(eidx);                   % FIXED: was shadowing outer epsilon
            mu_local = @(t_val) eps_val * t_val;            % FIXED: was using outer mu handle

            T_local = mu_final / eps_val;                   % FIXED: each worker needs its own T
            t_local = t0:dt:T_local;                        % FIXED: each worker needs its own t
            num_steps_local = length(t_local);              % FIXED: each worker needs its own length

            Heat_local = zeros(num_IC, num_IC);

            for ic = 1:num_IC
                k0 = IC_modes(ic);

                for run = 1:num_runs

                    u0 = mass + cos(2*pi*k0*x/(2*Lx));

                    u_hat = fft(u0);
                    u = u0;

                    for n = 2:num_steps_local             % FIXED: use local step count
                        t_prev = t_local(n-1);            % FIXED: use local t
                        t_half = t_prev + dt/2;
                        t_curr = t_local(n);              % FIXED: use local t

                        u3_nonlinear = u.^3 - mu_local(t_prev)*u;
                        Nu_hat = dealias_mask .* Laplacian_k .* fft(u3_nonlinear);

                        a_hat = E2.*u_hat + Q.*Nu_hat;
                        a = real(ifft(a_hat));
                        Na_hat = dealias_mask .* Laplacian_k .* fft(a.^3 - mu_local(t_half)*a);

                        b_hat = E2.*u_hat + Q.*Na_hat;
                        b = real(ifft(b_hat));
                        Nb_hat = dealias_mask .* Laplacian_k .* fft(b.^3 - mu_local(t_half)*b);

                        c_hat = E2.*a_hat + Q.*(2*Nb_hat - Nu_hat);
                        c = real(ifft(c_hat));
                        Nc_hat = dealias_mask .* Laplacian_k .* fft(c.^3 - mu_local(t_curr)*c);

                        u_hat = E.*u_hat + f1.*Nu_hat + 2*f2.*(Na_hat + Nb_hat) + f3.*Nc_hat;
                        u = real(ifft(u_hat));
                    end

                    [~, max_index] = max(abs(fft(u - mass)));
                    final_mode = round(kx(max_index)*Lx/pi);

                    row = find(IC_modes == final_mode);
                    if ~isempty(row)
                        Heat_local(row, ic) = Heat_local(row, ic) + 1;
                    end
                end

                fprintf('epsilon idx %d/%d | IC %d/%d done\n', eidx, num_eps, ic, num_IC);
            end

            Heat_cell{eidx} = Heat_local;
        end

        %% Reassemble 3D Heat array
        Heat = zeros(num_IC, num_IC, num_eps);
        for eidx = 1:num_eps
            Heat(:,:,eidx) = Heat_cell{eidx};
        end

        %% Plotting / Video
        v = VideoWriter('bifurcation_heatmap.avi');
        v.FrameRate = 10;                                   % FIXED: was 10*length() which is too fast
        open(v);

        levels = num_runs + 1;
        darkBlue = [0.02 0.12 0.55];
        white    = [1 1 1];
        cmap = [linspace(darkBlue(1),white(1),levels)', ...
                linspace(darkBlue(2),white(2),levels)', ...
                linspace(darkBlue(3),white(3),levels)'];

        for eidx = 1:num_eps
            figure(1); clf

            imagesc(IC_modes, IC_modes, Heat(:,:,eidx)/num_runs)
            axis xy
            xlabel('IC mode number')
            ylabel('Eq. mode number')
            title('Likelihood of equilibrium modes reached for various initial conditions');

            T_local = epsilon_list(eidx) > 0 ? mu_final / epsilon_list(eidx) : abs(t0);  % local T for label
            text(0.5, -0.1, sprintf('Parameters: T = %.0f,  t0 = %.0f, \\epsilon = %.4f\nEach IC ran for %.0f trials', ...
                T_local, t0, epsilon_list(eidx), num_runs), ...
                'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontSize', 10);

            xticks(IC_modes(1)-0.5:1:IC_modes(end)+0.5);
            yticks(IC_modes(1)-0.5:1:IC_modes(end)+0.5);
            grid on

            colormap(flipud(cmap))
            clim([0 1])
            colorbar
            drawnow

            frame = getframe(gcf);
            writeVideo(v, frame);
        end

        close(v);

    % ------------------------------------------------------------------------------------------
    % ------------------------------------------------------------------------------------------

    case 3

        %% Random ICs: sweep epsilon, random small-amplitude ICs each run
        % Plot: x = epsilon, y = Eq. mode, color = fraction of runs landing there
        sigma = 0.05;   % amplitude of random IC

        num_eps = length(epsilon_list);
        num_IC  = length(IC_modes);

        % Heat(eq_mode_idx, eps_idx) — marginalised over random ICs
        Heat = zeros(num_IC, num_eps);

        for eidx = 1:num_eps
            eps_val  = epsilon_list(eidx);
            mu_local = @(t_val) eps_val * t_val;
            T_local  = eps_val > 0 ? mu_final / eps_val : abs(t0);
            t_local  = t0:dt:T_local;
            num_steps_local = length(t_local);

            for run = 1:num_runs
                % Genuinely random small-amplitude IC
                u0    = mass + sigma * randn(1, Nx);
                u_hat = fft(u0);
                u     = u0;

                for n = 2:num_steps_local
                    t_prev = t_local(n-1);
                    t_half = t_prev + dt/2;
                    t_curr = t_local(n);

                    u3_nonlinear = u.^3 - mu_local(t_prev)*u;
                    Nu_hat = dealias_mask .* Laplacian_k .* fft(u3_nonlinear);

                    a_hat = E2.*u_hat + Q.*Nu_hat;
                    a = real(ifft(a_hat));
                    Na_hat = dealias_mask .* Laplacian_k .* fft(a.^3 - mu_local(t_half)*a);

                    b_hat = E2.*u_hat + Q.*Na_hat;
                    b = real(ifft(b_hat));
                    Nb_hat = dealias_mask .* Laplacian_k .* fft(b.^3 - mu_local(t_half)*b);

                    c_hat = E2.*a_hat + Q.*(2*Nb_hat - Nu_hat);
                    c = real(ifft(c_hat));
                    Nc_hat = dealias_mask .* Laplacian_k .* fft(c.^3 - mu_local(t_curr)*c);

                    u_hat = E.*u_hat + f1.*Nu_hat + 2*f2.*(Na_hat + Nb_hat) + f3.*Nc_hat;
                    u = real(ifft(u_hat));
                end

                [~, max_index] = max(abs(fft(u - mass)));
                final_mode = round(kx(max_index)*Lx/pi);

                row = find(IC_modes == final_mode);
                if ~isempty(row)
                    Heat(row, eidx) = Heat(row, eidx) + 1;
                end
            end

            fprintf('epsilon idx %d/%d done\n', eidx, num_eps);
        end

        % Normalise each epsilon column so values are fractions in [0,1]
        col_sums = sum(Heat, 1);
        Heat_norm = Heat ./ max(col_sums, 1);

        %% Plotting
        figure('Position', [100 100 900 550])
        imagesc(1:num_eps, IC_modes, Heat_norm)
        axis xy
        xlabel(sprintf('\\epsilon index  (\\epsilon \\in [%.3f, %.3f])', ...
                       epsilon_list(1), epsilon_list(end)))
        ylabel('Eq. mode number')
        title(sprintf('Equilibrium mode likelihood — random ICs\n\\sigma=%.3f   \\mu_{final}=%.2f   runs/\\epsilon=%d', ...
                      sigma, mu_final, num_runs))

        % Sparse x-tick labels showing actual epsilon values
        tick_step = max(1, floor(num_eps/10));
        xtick_idx = 1:tick_step:num_eps;
        xticks(xtick_idx)
        xticklabels(arrayfun(@(e) sprintf('%.3f', e), epsilon_list(xtick_idx), ...
                             'UniformOutput', false))
        xtickangle(45)

        levels = num_runs + 1;
        darkBlue = [0.02 0.12 0.55];
        white    = [1 1 1];
        cmap = [linspace(darkBlue(1),white(1),levels)', ...
                linspace(darkBlue(2),white(2),levels)', ...
                linspace(darkBlue(3),white(3),levels)'];
        colormap(flipud(cmap))
        clim([0 1])
        colorbar

        saveas(gcf, 'bifurcation_heatmap_type3.png')

end

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Helper: find the time at which the system first reaches equilibrium.
%
%  Equilibrium is defined as the first time the dominant mode has been
%  constant AND the relative L2-norm change has been below EQ_REL_TOL
%  for EQ_WINDOW consecutive steps.
%
%  Returns NaN if equilibrium is not reached within the simulation.

function t_eq = find_equilibrium_time(kdom_hist, l2_hist, t_vec)

    EQ_WINDOW  = 50;
    EQ_REL_TOL = 1e-4;

    N    = length(kdom_hist);
    t_eq = NaN;

    if N < EQ_WINDOW + 1
        return
    end

    for i = (EQ_WINDOW + 1):N
        window_k  = kdom_hist(i - EQ_WINDOW : i);
        window_l2 = l2_hist(i - EQ_WINDOW : i);

        % All dominant modes in the window are identical
        if all(window_k == window_k(1))
            l2_range = max(window_l2) - min(window_l2);
            l2_ref   = max(abs(window_l2(1)), 1e-14);
            if l2_range / l2_ref < EQ_REL_TOL
                t_eq = t_vec(i - EQ_WINDOW);   % start of the stable window
                return
            end
        end
    end
end