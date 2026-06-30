clc; clear; close all;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Description and configuration
% Find how different ICs and epsilon values affect equilibrium mode

% 0: single run  ( solution | fourier spectrum / dominant mode | L2 norm )
% 1: single mode IC ( plot x = IC mode VS. y = Eq. mode VS. color = probability )
% 2: single mode IC + vary epsilon ( plot 1 but varying in time = epsilon )
% 3: random IC in x ( plot x = epsilon VS. y = Eq. mode VS. color = probability )
% 4: random IC in Fourier space ( plot x = epsilon VS. y = Eq. mode VS. color = probability )
% 5: directed IC in Fourier space ( plot x = x-direction of IC, y = y-direction of IC, color = Eq. mode  )
% 6: heuristic test 

type = 4;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Parameters
% heat map params
num_runs = 20;
xscale = 8;
IC_modes = 0:xscale+4;

epsilon = 0.1; % type == 0 or 1
epsilon_exp = -1.*linspace(1, 6, 20);
epsilon_list = 10.^(flip(epsilon_exp)); % type == 2, 3, etc.

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
        mode_index = round(abs(kx(max_index))*Lx/pi);

        kdom_hist(1) = mode_index;
        l2_hist(1)   = sqrt(dx*sum(u0.^2)); % sqrt(sum(u0.^2) * dx);

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

            u_centered=u-mean(u);
            [~, max_index] = max(abs(fft(u_centered))); % replace u - u_mass by u_centered b/c mass not perfectly conserved
            mode_index = round(abs(kx(max_index))*Lx/pi);

            kdom_hist(n) = mode_index;
            l2_hist(n)   = sqrt(dx*sum(u.^2)); % sqrt(sum(u.^2) * dx);
        end

        % Equilibrium time
        t_select = find_selection_time(kdom_hist, t);
        if isnan(t_select)
            fprintf('Equilibrium not detected within simulation window.\n');
        else
            fprintf('Equilibrium first reached at t = %.3f\n', t_select);
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
        if ~isnan(t_select)
            xline(t_select, '-g', 'LineWidth', 1.5, ...
                  'Label', sprintf('first eq. at t = %.1f', t_select), 'LabelVerticalAlignment', 'bottom')
        end
        hold off
        xlabel('t'); ylabel('dominant |k|')
        title('Dominant wavenumber vs time')
        grid on

        % Subplot 4 — L2 norm vs time with equilibrium marker
        subplot(2,2,4)
        plot(t, l2_hist, 'LineWidth', 1.5, 'Color', [0.13 0.55 0.55])
        hold on
        if ~isnan(t_select)
            xline(t_select, '--g', 'LineWidth', 2, ...
                  'Label', sprintf('first eq. time = %.1f', t_select), 'LabelVerticalAlignment', 'bottom')
            eq_idx = find(t >= t_select, 1);
            plot(t(eq_idx), l2_hist(eq_idx), 'og', 'MarkerSize', 8, 'MarkerFaceColor', 'g')
        end
        hold off
        xlabel('t'); ylabel('||u||_2')
        title('L_2 norm vs time')
        grid on

        sgtitle(sprintf('\\epsilon=%.4f  t_0=%.1f  mass=%.2f  IC mode=%d', ...
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
                %final_mode = round(abs(kx(max_index))*Lx/pi);

                uhat = fft(u);

                % ignore numerical noise
                tol = 1e-10;

                if norm(u - mean(u),2) < tol
                    final_mode = 0;
                else
                    uhat(1) = 0;              % ignore DC only if not constant
                    [~,idx] = max(abs(uhat));
                    final_mode = round(abs(kx(idx))*Lx/pi);
                end

                row = find(IC_modes == final_mode);
                if ~isempty(row)
                    Heat(row, ic) = Heat(row, ic) + 1;
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

                u0 = mass + cos(2*pi*k0*x/(2*Lx));

                for run = 1:num_runs

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

                    u_centered=u-mean(u);
                    [~, max_index] = max(abs(fft(u_centered)));
                    %final_mode = round(abs(kx(max_index))*Lx/pi);

                    uhat = fft(u);

                    % ignore numerical noise
                    tol = 1e-10;

                    if norm(u - mean(u),2) < tol
                        final_mode = 0;
                    else
                        uhat(1) = 0;              % ignore DC only if not constant
                        [~,idx] = max(abs(uhat));
                        final_mode = round(abs(kx(idx))*Lx/pi);
                    end

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

            if epsilon_list(eidx) > 0
                T_local = mu_final / epsilon_list(eidx);
            else
                T_local = abs(t0);
            end

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
            if eps_val > 0
                T_local = mu_final / eps_val;
            else
                T_local = abs(t0);
            end
            T_local = min(180, T_local);
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
                %final_mode = round(abs(kx(max_index))*Lx/pi);

                uhat = fft(u);

                % ignore numerical noise
                tol = 1e-10;

                if norm(u - mean(u),2) < tol
                    final_mode = 0;
                else
                    uhat(1) = 0;              % ignore DC only if not constant
                    [~,idx] = max(abs(uhat));
                    final_mode = round(abs(kx(idx))*Lx/pi);
                end

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
        title(sprintf('Equilibrium mode likelihood — random ICs\n\\sigma=%.3f   \\mu_{final}=%.2f   runs/\\epsilon=%d,   length=%d*2\\pi', ...
                      sigma, mu_final, num_runs, xscale))

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

    % ------------------------------------------------------------------------------------------
    % ------------------------------------------------------------------------------------------

    case 4

        %% Random ICs: sweep epsilon, random small-amplitude ICs each run
        % Plot: x = epsilon, y = Eq. mode, color = fraction of runs landing there
        sigma = 0.5;   % amplitude of random IC

        num_eps = length(epsilon_list);
        num_IC  = length(IC_modes);

        % Heat(eq_mode_idx, eps_idx) — marginalised over random ICs
        Heat = zeros(num_IC, num_eps);

        for eidx = 1:num_eps
            eps_val  = epsilon_list(eidx);
            mu_local = @(t_val) eps_val * t_val;
            if eps_val > 0
                T_local = mu_final / eps_val;
            else
                T_local = abs(t0);
            end
            T_local = min(180, T_local);
            t_local  = t0:dt:T_local;
            num_steps_local = length(t_local);

            for run = 1:num_runs
                % random in Fourier space small-amplitude IC
                u_hat = zeros(1,Nx);

                % Randomly activate modes
                activation_prob = 1.0;      % percent of modes active

                for k = 2:Nx/2

                    if rand < activation_prob

                        a = sigma*(randn + 1i*randn);

                        % positive frequency
                        u_hat(k) = a;

                        % negative frequency (Hermitian symmetry)
                        u_hat(Nx-k+2) = conj(a);

                    end

                end

                u_hat(1) = 0; % 0 mode = 0
                u_hat(Nx/2+1) = 0; % nyquist mode real
                u = real(ifft(u_hat));

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
                %final_mode = round(abs(kx(max_index))*Lx/pi);

                uhat = fft(u);

                % ignore numerical noise
                tol = 1e-4;

                if norm(u - mean(u),2) < tol
                    final_mode = 0;
                else
                    uhat(1) = 0;              % ignore  only if not constant
                    [~,idx] = max(abs(uhat));
                    final_mode = round(abs(kx(idx))*Lx/pi);
                end

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
        title(sprintf('Equilibrium mode likelihood — random F.ICs\n\\sigma=%.3f   \\mu_{final}=%.2f   runs/\\epsilon=%d,   length=%d*2\\pi,   prob=%.2f', ...
                      sigma, mu_final, num_runs, xscale, activation_prob))

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

        saveas(gcf, 'bifurcation_heatmap_type4.png')

    % ------------------------------------------------------------------------------------------
    % ------------------------------------------------------------------------------------------

    case 5

        %% fix mode, fix epsilon, sweep mode activation direction
        % Plot: x = x-direction, y = y-direction, color = Eq. mode
        sigma = 0.5;   % amplitude of random IC

        % Fourier wavenumbers
        [X,Y] = meshgrid(linspace(0,1,30),linspace(0,1,30));

        % Heat(eq_mode_idx, eps_idx) — marginalised over random ICs
        Heat = zeros(num_IC, num_eps);

        for eidx = 1:num_eps
            eps_val  = epsilon_list(eidx);
            mu_local = @(t_val) eps_val * t_val;
            if eps_val > 0
                T_local = mu_final / eps_val;
            else
                T_local = abs(t0);
            end
            T_local = min(180, T_local);
            t_local  = t0:dt:T_local;
            num_steps_local = length(t_local);

            for run = 1:num_runs
                % random in Fourier space small-amplitude IC
                u_hat = zeros(1,Nx);

                % Randomly activate modes
                activation_prob = 1.0;      % percent of modes active

                for k = 2:Nx/2

                    if rand < activation_prob

                        a = sigma*(randn + 1i*randn);

                        % positive frequency
                        u_hat(k) = a;

                        % negative frequency (Hermitian symmetry)
                        u_hat(Nx-k+2) = conj(a);

                    end

                end

                u_hat(1) = 0; % 0 mode = 0
                u_hat(Nx/2+1) = 0; % nyquist mode real
                u = real(ifft(u_hat));

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
                %final_mode = round(abs(kx(max_index))*Lx/pi);

                uhat = fft(u);

                % ignore numerical noise
                tol = 1e-4;

                if norm(u - mean(u),2) < tol
                    final_mode = 0;
                else
                    uhat(1) = 0;              % ignore  only if not constant
                    [~,idx] = max(abs(uhat));
                    final_mode = round(abs(kx(idx))*Lx/pi);
                end

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
        title(sprintf('Equilibrium mode likelihood — random F.ICs\n\\sigma=%.3f   \\mu_{final}=%.2f   runs/\\epsilon=%d,   length=%d*2\\pi,   prob=%.2f', ...
                      sigma, mu_final, num_runs, xscale, activation_prob))

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

        saveas(gcf, 'bifurcation_heatmap_type5.png')


    % ------------------------------------------------------------------------------------------
    % ------------------------------------------------------------------------------------------

    case 6

        %% Case 6: Animated dominant-mode evolution + t_eq comparison plot
        %
        % For a fixed epsilon:
        %   - Sweep over IC modes (single cosine ICs, like case 1)
        %   - At every time step, record dominant mode for each IC
        %   - Animate as binary heatmap: x = IC mode, y = dominant mode, cell = black if active
        %   - After all runs: plot t_eq vs theoretical activation time t_k = 2*k_hat^2/epsilon
        %     and overlay the k*(t) = sqrt(mu(t)/2) drift curve

        % ---- parameters you may want to tune -------------------------
        epsilon_6   = epsilon;          % uses the epsilon defined at top of script
        IC_modes_6  = 1:xscale;        % single-mode ICs to sweep (skip 0 — trivial)
        num_runs_6  = num_runs;         % runs per IC (for noise robustness; IC is deterministic
                                        % so >1 run only matters if you add noise — set 1 for speed)
        record_every = 5;              % record dominant mode every N time steps (keeps video manageable)
        % --------------------------------------------------------------

        mu_6    = @(t_val) epsilon_6 * t_val;
        T_6     = mu_final / epsilon_6;
        t_6     = t0:dt:T_6;
        Nt_6    = length(t_6);

        % Recompute ETDRK4 coefficients for this epsilon's T (E/E2/Q/f1/f2/f3 depend only on
        % L_operator and dt, which are unchanged, so we can reuse them directly)

        num_IC_6    = length(IC_modes_6);
        rec_idx     = 1:record_every:Nt_6;          % indices we record
        num_rec     = length(rec_idx);
        t_rec       = t_6(rec_idx);

        % Storage: dom_mode_hist(ic_idx, rec_step) = dominant mode index
        dom_mode_hist = zeros(num_IC_6, num_rec);

        % t_eq per IC (first time dominant mode stabilises — reuse find_selection_time)
        t_eq_6 = nan(1, num_IC_6);

        %% Main loop over ICs
        for ic = 1:num_IC_6

            k0   = IC_modes_6(ic);
            u0   = mass + cos(2*pi*k0*x/(2*Lx));   % single cosine IC
            u_hat_ic = fft(u0);
            u_ic     = u0;

            kdom_hist_6 = zeros(1, Nt_6);

            % record dominant mode at t0
            u_centered = u_ic - mean(u_ic);
            [~, midx] = max(abs(fft(u_centered)));
            kdom_hist_6(1) = round(abs(kx(midx)) * Lx / pi);

            %% Time loop
            for n = 2:Nt_6
                t_prev = t_6(n-1);
                t_half = t_prev + dt/2;
                t_curr = t_6(n);

                u3_nl  = u_ic.^3 - mu_6(t_prev)*u_ic;
                Nu_hat = dealias_mask .* Laplacian_k .* fft(u3_nl);

                a_hat  = E2.*u_hat_ic + Q.*Nu_hat;
                a      = real(ifft(a_hat));
                Na_hat = dealias_mask .* Laplacian_k .* fft(a.^3 - mu_6(t_half)*a);

                b_hat  = E2.*u_hat_ic + Q.*Na_hat;
                b      = real(ifft(b_hat));
                Nb_hat = dealias_mask .* Laplacian_k .* fft(b.^3 - mu_6(t_half)*b);

                c_hat  = E2.*a_hat + Q.*(2*Nb_hat - Nu_hat);
                c      = real(ifft(c_hat));
                Nc_hat = dealias_mask .* Laplacian_k .* fft(c.^3 - mu_6(t_curr)*c);

                u_hat_ic = E.*u_hat_ic + f1.*Nu_hat + 2*f2.*(Na_hat + Nb_hat) + f3.*Nc_hat;
                u_ic     = real(ifft(u_hat_ic));

                % Dominant mode detection
                tol_6 = 1e-10;
                if norm(u_ic - mean(u_ic), 2) < tol_6
                    kdom_hist_6(n) = 0;
                else
                    uhat_tmp    = fft(u_ic);
                    uhat_tmp(1) = 0;
                    [~, idx]    = max(abs(uhat_tmp));
                    kdom_hist_6(n) = round(abs(kx(idx)) * Lx / pi);
                end
            end

            % Store subsampled history
            dom_mode_hist(ic, :) = kdom_hist_6(rec_idx);

            % Find equilibrium time for this IC
            t_eq_6(ic) = find_selection_time(kdom_hist_6, t_6);

            fprintf('Case 6: IC mode %d/%d done\n', ic, num_IC_6);
        end

        % ---- all mode values that appear (for axis limits) ----------
        all_modes_6 = unique([IC_modes_6, reshape(dom_mode_hist, 1, [])]);
        all_modes_6 = all_modes_6(~isnan(all_modes_6));
        y_axis_6    = 0:max(all_modes_6);   % eq. mode axis includes 0

        % -----------------------------------------------------------------
        %% VIDEO: animated binary heatmap
        % -----------------------------------------------------------------
        v6 = VideoWriter('case6_dominant_mode_evolution.avi');
        v6.FrameRate = 15;
        open(v6);

        fig6a = figure('Position', [100 100 900 700]);

        % Colormap: white = inactive, dark blue = active
        darkBlue = [0.02 0.12 0.55];
        cmap6    = [1 1 1; darkBlue];   % 0 -> white, 1 -> dark blue

        for ri = 1:num_rec

            % Build binary matrix: rows = eq mode, cols = IC mode
            Frame = zeros(length(y_axis_6), num_IC_6);
            for ic = 1:num_IC_6
                dom = dom_mode_hist(ic, ri);
                row = find(y_axis_6 == dom);
                if ~isempty(row)
                    Frame(row, ic) = 1;
                end
            end

            clf
            imagesc(IC_modes_6, y_axis_6, Frame)
            axis xy
            colormap(cmap6)
            clim([0 1])

            xlabel('IC mode k_0', 'FontSize', 12)
            ylabel('Dominant mode at time t', 'FontSize', 12)

            mu_now = mu_6(t_rec(ri));
            k_star = sqrt(max(mu_now/2, 0));   % preferred mode k*(t) in physical wavenumber units
            % Convert to mode index: k_hat = n/N => n = k_hat * N = k* * xscale
            n_star = k_star * xscale;

            title(sprintf('Dominant mode evolution  |  \\epsilon=%.4f  |  t=%.2f  |  \\mu=%.3f  |  k^*(t)=%.2f', ...
                          epsilon_6, t_rec(ri), mu_now, n_star), 'FontSize', 11)

            % Overlay k*(t) drift curve as horizontal line (preferred mode index)
            hold on
            yline(n_star, '--r', 'LineWidth', 2, ...
                  'Label', 'k^*(t)', 'LabelVerticalAlignment', 'bottom', ...
                  'LabelHorizontalAlignment', 'right')

            % Mark the theoretical boundary n* = N*sqrt(mu_final/2)
            n_boundary = xscale * sqrt(mu_final/2);
            yline(n_boundary, '--', 'Color', [0.8 0.5 0], 'LineWidth', 1.5, ...
                  'Label', 'n^* boundary', 'LabelVerticalAlignment', 'top', ...
                  'LabelHorizontalAlignment', 'right')
            hold off

            grid on
            drawnow

            frame6 = getframe(fig6a);
            writeVideo(v6, frame6);
        end

        close(v6);
        fprintf('Video saved: case6_dominant_mode_evolution.avi\n');

        % -----------------------------------------------------------------
        %% STATIC PLOT: t_eq vs theoretical quantities
        % -----------------------------------------------------------------

        % Theoretical activation time: t_k = 2*k_hat^2 / epsilon
        % where k_hat = k / xscale (since Lx = xscale*pi, khat = k/xscale)
        k_hat_vec   = IC_modes_6 / xscale;
        t_k_theory  = 2 * k_hat_vec.^2 / epsilon_6;  % from integrated eigenvalue zero-crossing

        % k*(t) at t_eq: the preferred mode index at the moment each IC reaches equilibrium
        % k_star_at_teq(ic) = xscale * sqrt( mu_6(t_eq_6(ic)) / 2 )  [as mode index n]
        k_star_at_teq = xscale * sqrt(max(mu_6(t_eq_6), 0) / 2);

        % Dense curve for k*(t) overlay
        t_dense   = linspace(max(t0, 0), T_6, 500);
        n_star_curve = xscale * sqrt(mu_6(t_dense) / 2);

        fig6b = figure('Position', [100 100 1100 500]);

        % --- Left panel: t_eq vs IC mode, overlaid with t_k theory ---
        subplot(1,2,1)
        scatter(IC_modes_6, t_eq_6, 60, IC_modes_6, 'filled', 'DisplayName', 't_{eq} (empirical)')
        hold on
        plot(IC_modes_6, t_k_theory, 'r--', 'LineWidth', 2, 'DisplayName', 't_k = 2\hat{k}^2/\epsilon')

        % Mark the n* boundary
        n_boundary = xscale * sqrt(mu_final/2);
        xline(n_boundary, '--', 'Color', [0.8 0.5 0], 'LineWidth', 1.5, ...
              'Label', 'n^*', 'LabelVerticalAlignment', 'bottom')
        hold off

        colormap(gca, parula)
        colorbar
        xlabel('IC mode k_0', 'FontSize', 12)
        ylabel('t_{eq}', 'FontSize', 12)
        title(sprintf('Equilibrium time vs IC mode\n\\epsilon=%.4f,  \\mu_{final}=%.2f,  L=%d\\pi', ...
                      epsilon_6, mu_final, xscale), 'FontSize', 11)
        legend('Location', 'northwest', 'FontSize', 9)
        grid on

        % --- Right panel: scatter of t_eq vs t_k (should be linear if theory holds) ---
        subplot(1,2,2)

        % Only plot ICs where t_eq was detected (not NaN)
        valid = ~isnan(t_eq_6);

        scatter(t_k_theory(valid), t_eq_6(valid), 60, IC_modes_6(valid), 'filled')
        hold on

        % Reference line y = x (perfect agreement)
        t_range = [min(t_k_theory(valid)), max(t_k_theory(valid))];
        plot(t_range, t_range, 'k--', 'LineWidth', 1.5, 'DisplayName', 'y = x  (perfect)')

        % Also fit a line
        if sum(valid) > 2
            p = polyfit(t_k_theory(valid), t_eq_6(valid), 1);
            t_fit = linspace(t_range(1), t_range(2), 100);
            plot(t_fit, polyval(p, t_fit), 'r-', 'LineWidth', 1.5, ...
                 'DisplayName', sprintf('fit: t_{eq} = %.2f t_k + %.2f', p(1), p(2)))
        end
        hold off

        colormap(gca, parula)
        cb = colorbar;
        cb.Label.String = 'IC mode k_0';
        xlabel('t_k = 2\hat{k}^2/\epsilon  (theory)', 'FontSize', 12)
        ylabel('t_{eq}  (empirical)', 'FontSize', 12)
        title('Empirical vs theoretical activation time', 'FontSize', 11)
        legend('Location', 'northwest', 'FontSize', 9)
        grid on

        sgtitle(sprintf('Case 6 — \\epsilon=%.4f,  N=%d,  \\mu_{final}=%.2f', ...
                        epsilon_6, xscale, mu_final), 'FontSize', 13)

        saveas(fig6b, 'case6_teq_comparison.png')
        fprintf('Plot saved: case6_teq_comparison.png\n');

end

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Helper: find the time at which the system first reaches equilibrium.
%
% Selection time := the first occurrence of a mode that then
% remains unchanged for EQ_WINDOW consecutive time steps.
%
% Returns the FIRST time that equilibrium mode is reached.
% Returns NaN if equilibrium is never detected.
function t_select = find_selection_time(kdom_hist, t_vec)

    EQ_WINDOW = 50;

    N = length(kdom_hist);
    t_select = NaN;

    if N < EQ_WINDOW
        return
    end

    % Look for the first window of identical modes
    for i = EQ_WINDOW:N

        window = kdom_hist(i-EQ_WINDOW+1:i);

        if all(window == window(end))

            eq_mode = window(end);

            % Return the FIRST time this mode was ever reached
            first_idx = find(kdom_hist == eq_mode, 1, 'first');

            t_select = t_vec(first_idx);
            return

        end
    end
end