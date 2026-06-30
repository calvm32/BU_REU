clc; clear; close all;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Description and configuration
% Galerkin projection of 1D Cahn-Hilliard
%
% u_t = -d_xx(d_xx u + mu(t)u - u^3)
%
% Modes |k|<=3
%
% This version evolves everything in mu-space: mu(t) = epsilon*t, and all
% time-series plots (dominant wavenumber, L2 norm) use mu as the x-axis
% instead of t. The 2x2 panel is rebuilt at each mu-frame and written out
% as an AVI movie so you can watch the dynamics "running in mu".

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------


%% Parameters

epsilon  = 0.02;
mu_final = 2;
t0       = 0;
T        = mu_final/epsilon;

K = -3:3;
Nmodes = length(K);
index_of = @(k) find(K==k);

L = 8; % half-domain size
Lx = L*pi;

%% Movie / frame parameters

Nframes        = 200;              % number of evenly-spaced mu-frames
tspan          = linspace(t0,T,Nframes);   % uniform in t => uniform in mu since mu=eps*t
mu_vec_target  = epsilon*tspan;            % the mu-axis we evolve over

eq_window      = 10;                % # consecutive frames the dominant mode must hold to call it "equilibrium"

%% Build cubic interaction tensor

C = zeros(Nmodes,Nmodes,Nmodes,Nmodes);

for kk = 1:Nmodes
    k = K(kk);

    for mm = 1:Nmodes
        m = K(mm);

        for nn = 1:Nmodes
            n = K(nn);

            for pp = 1:Nmodes
                p = K(pp);

                if m+n+p==k
                    C(kk,mm,nn,pp)=1;
                end
            end
        end
    end
end


% ------------------------------------------------------------------------------------------
%% Initial condition
% ------------------------------------------------------------------------------------------

rng(1)

u0 = 1e-2*(randn(Nmodes,1)+1i*randn(Nmodes,1));

% Make solution real
u0(index_of(0)) = real(u0(index_of(0)));
for k=1:3
    u0(index_of(-k)) = conj(u0(index_of(k)));
end

%% Solve on the prescribed mu-uniform tspan
% Passing tspan as a vector (rather than [t0 T]) makes ode45 return the
% solution interpolated exactly at these points. The internal adaptive
% steps are unaffected/unrestricted in accuracy -- this just controls the
% *output* sampling, which is what we need for evenly-spaced movie frames.

opts = odeset('RelTol',1e-8,'AbsTol',1e-10);
[t,U] = ode45(@(t,u) rhs(t,u,epsilon,K,C,L), tspan, u0, opts);

mu_vec = epsilon*t;   % mu-axis actually realized (should match mu_vec_target)

%% Reconstruct real-space solution at every frame

Nx = 512;
x  = linspace(-L*pi,L*pi,Nx);

Ux = zeros(length(t),Nx);

for j=1:length(t)
    temp = zeros(1,Nx);

    for ii=1:Nmodes
        temp = temp + U(j,ii)*exp(1i*K(ii)*x/L);
    end

    Ux(j,:) = real(temp);
end

%% L2 norm history (consistent normalization: mean-square over domain)

dx = x(2)-x(1);
L2_hist = sqrt(trapz(x,Ux.^2,2)/(2*L*pi));   % length(t) x 1

%% Dominant Fourier mode history
% For the Galerkin solution this is just the |k| with largest |U_k|,
% excluding the k=0 (mean) mode.

kdom_hist = zeros(length(t),1);
notzero   = K~=0;

for j=1:length(t)
    coeffs = abs(U(j,:));
    coeffs(~notzero) = -Inf;          % exclude k=0 from the max-search
    [~,idx] = max(coeffs);
    kdom_hist(j) = abs(K(idx));
end

%% Equilibrium (mode-selection) time, sliding-window criterion
% "Equilibrium" here = first mu at which the dominant |k| has held its
% value for eq_window consecutive frames in a row.

mu_select = find_selection_time(kdom_hist, mu_vec, eq_window);

if isnan(mu_select)
    fprintf('Equilibrium not detected within simulation window.\n');
else
    fprintf('Equilibrium (mode selection) first reached at mu = %.4f\n', mu_select);
end

%% Reference wavenumbers for overlay lines

k_ref = 0:6;     % integer Galerkin modes |k| = 0..3 matter most, but keep a few extra for context

%% ------------------------------------------------------------------------------------------
%% Build the mu-evolving 2x2 movie
%% ------------------------------------------------------------------------------------------

vidObj = VideoWriter('galerkin_ch_mu_evolution.avi','Motion JPEG AVI');
vidObj.FrameRate = 20;
vidObj.Quality   = 90;
open(vidObj);

fig = figure('Position',[100 100 1200 800]);

% Precompute global y-limits so the movie doesn't jitter frame to frame
ylim_real = [min(Ux(:)) max(Ux(:))]*1.1;
ylim_L2   = [0 max(L2_hist)*1.1];
ylim_kdom = [0 max(kdom_hist)*1.2 + 0.5];

for j = 1:length(t)

    clf(fig);

    mu_now = mu_vec(j);

    % Subplot 1: real-space solution at current mu
    subplot(2,2,1)
    plot(x, Ux(j,:), 'LineWidth', 1.5) % , 'Color', [0.18 0.45 0.69]
    xlabel('x'); ylabel('u(x)')
    title(sprintf('Real-space solution, \\mu = %.3f', mu_now))
    ylim(ylim_real)
    grid on

    % Subplot 2: Fourier spectrum at current mu (Galerkin coefficients)
    subplot(2,2,2)
    stem(K, abs(U(j,:)), 'LineWidth', 1.2)
    xlabel('k'); ylabel('|U_k|')
    title(sprintf('Fourier spectrum, \\mu = %.3f', mu_now))
    xlim([min(K)-0.5 max(K)+0.5])
    ylim_spec = [0 max(abs(U(:)))*1.1 + 1e-12];
    ylim(ylim_spec)
    grid on

    % Subplot 3: dominant wavenumber vs mu (history up to current frame)
    subplot(2,2,3)
    stairs(mu_vec(1:j), kdom_hist(1:j), 'LineWidth', 2) % , 'Color', [0.85 0.33 0.10]
    hold on
    for kj = k_ref
        yline(kj, '--r', 'Alpha', 0.3)
    end
    if ~isnan(mu_select) && mu_now >= mu_select
        % xline(mu_select, '-g', 'LineWidth', 1.5, ...
        %     'Label', sprintf('eq. at \\mu=%.2f', mu_select), 'LabelVerticalAlignment','bottom')
    end
    hold off
    xlabel('\mu'); ylabel('dominant |k|')
    xlim([mu_vec(1) mu_vec(end)])
    ylim(ylim_kdom)
    title('Dominant wavenumber vs \mu')
    grid on

    % Subplot 4: L2 norm vs mu (history up to current frame)
    subplot(2,2,4)
    plot(mu_vec(1:j), L2_hist(1:j), 'LineWidth', 1.5) % , 'Color', [0.13 0.55 0.55]
    hold on
    if ~isnan(mu_select) && mu_now >= mu_select
        % xline(mu_select, '--g', 'LineWidth', 1.5, ...
        %     'Label', sprintf('eq. \\mu=%.2f', mu_select), 'LabelVerticalAlignment','bottom')
        eq_idx = find(mu_vec(1:j) >= mu_select, 1);
        % if ~isempty(eq_idx)
        %     plot(mu_vec(eq_idx), L2_hist(eq_idx), 'og', 'MarkerSize', 8, 'MarkerFaceColor','g')
        % end
    end
    hold off
    xlabel('\mu'); ylabel('||u||_2')
    xlim([mu_vec(1) mu_vec(end)])
    ylim(ylim_L2)
    title('L_2 norm vs \mu')
    grid on

    sgtitle(sprintf('Galerkin Cahn-Hilliard \\quad \\epsilon=%.4f \\quad \\mu(t)=\\epsilon t \\quad \\mu = %.3f / %.3f', ...
                     epsilon, mu_now, mu_vec(end)))

    drawnow;
    frame = getframe(fig);
    writeVideo(vidObj, frame);
end

close(vidObj);
close(fig);

fprintf('Movie written to galerkin_ch_mu_evolution.avi (%d frames)\n', length(t));

%% Also save a static final-frame PNG for quick reference

fig2 = figure('Position',[100 100 1200 800]);

subplot(2,2,1)
plot(x, Ux(end,:), 'LineWidth', 1.5) % , 'Color', [0.18 0.45 0.69]
xlabel('x'); ylabel('u(x)')
title(sprintf('Real-space solution, \\mu = %.3f', mu_vec(end)))
grid on

subplot(2,2,2)
stem(K, abs(U(end,:)), 'filled', 'LineWidth') % , 1.2, 'Color', [0.85 0.33 0.10]
xlabel('k'); ylabel('|U_k|')
title('Fourier spectrum (final \mu)')
grid on

subplot(2,2,3)
stairs(mu_vec, kdom_hist, 'LineWidth', 2) % , 'Color', [0.49 0.18 0.56]
hold on
for kj = k_ref
    yline(kj, '--r', 'Alpha', 0.3)
end
if ~isnan(mu_select)
    % xline(mu_select, '-g', 'LineWidth', 1.5, ...
    %     'Label', sprintf('eq. at \\mu=%.2f', mu_select), 'LabelVerticalAlignment','bottom')
end
hold off
xlabel('\mu'); ylabel('dominant |k|')
title('Dominant wavenumber vs \mu')
grid on

subplot(2,2,4)
plot(mu_vec, L2_hist, 'LineWidth', 1.5) % , 'Color', [0.13 0.55 0.55]
hold on
if ~isnan(mu_select)
    % xline(mu_select, '--g', 'LineWidth', 1.5, ...
    %     'Label', sprintf('eq. \\mu=%.2f', mu_select), 'LabelVerticalAlignment','bottom')
    eq_idx = find(mu_vec >= mu_select, 1);
    % plot(mu_vec(eq_idx), L2_hist(eq_idx), 'og', 'MarkerSize', 8, 'MarkerFaceColor','g')
end
hold off
xlabel('\mu'); ylabel('||u||_2')
title('L_2 norm vs \mu')
grid on

sgtitle(sprintf('\\epsilon=%.4f  mu\\_final=%.2f  L=%d', epsilon, mu_final, L))

saveas(fig2, 'galerkin_ch_mu_final.png')
close(fig2);


%% Functions

function dudt = rhs(t,u,epsilon,K,C,L)

    Nmodes = length(K);
    mu = epsilon*t;
    dudt = zeros(Nmodes,1);

    for kk = 1:Nmodes
        nonlinear = 0;

        for mm = 1:Nmodes
            for nn = 1:Nmodes
                for pp = 1:Nmodes
                    nonlinear = nonlinear + C(kk,mm,nn,pp)*u(mm)*u(nn)*u(pp);
                end
            end
        end

        lambda = ...
            -(K(kk)^4)/(L^4) ...
            + mu*(K(kk)^2)/(L^2);

        dudt(kk) = ...
            lambda*u(kk) ...
            - (K(kk)^2)/(L^2)*nonlinear;

    end

end


function mu_select = find_selection_time(kdom_hist, mu_vec, window)
% find_selection_time  First mu at which the dominant wavenumber has held
% a constant value for `window` consecutive samples in a row.
%
% This is the same sliding-window stability idea used in
% find_equilibrium_time elsewhere in your codebase: once kdom_hist(j) has
% been equal to kdom_hist(j-window+1:j) for `window` consecutive frames,
% we call mu_vec(j-window+1) the selection (equilibrium) mu -- i.e. the
% mu at which the *run* of stability began, not where it's confirmed.
%
% Returns NaN if no such run of length `window` occurs.

    N = length(kdom_hist);
    mu_select = NaN;

    if N < window
        return
    end

    for j = window:N
        win = kdom_hist(j-window+1:j);
        if all(win == win(1))
            mu_select = mu_vec(j-window+1);
            return
        end
    end

end