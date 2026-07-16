clc; clear; close all;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Description
% Galerkin projection of 1D Cahn-Hilliard
%
% u_t = -d_xx(d_xx u + mu(t)u - u^3)
%
% Modes |k|<=3
%
% evolves everything in mu-space: mu(t) = epsilon*t, and all
% time-series plots (dominant wavenumber, L2 norm) use mu as the x-axis
% instead of t. The 2x2 panel is rebuilt at each mu-frame

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------


%% Parameters
epsilon = 10^(-4);
mu_final = 0.5;
t0 = -0.2 / epsilon;
T = mu_final/epsilon;
m = 0.5; % final mass

K = -3:3;
Nmodes = length(K);
index_of = @(k) find(K==k);

L = 10; % half-domain size
Lx = L*pi;

%% Movie / frame parameters

Nframes  = 200;                     % number of evenly-spaced mu-frames
tspan = linspace(t0,T,Nframes);     % uniform in t => uniform in mu since mu=eps*t
mu_vec_target = epsilon*tspan;      % the mu-axis we evolve over
mu0 = epsilon*t0;

eq_window = 10;                     % # consecutive frames the dominant mode must hold to call it "equilibrium"

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


%% Print the ODE system being solved (human-readable, built from K and C)
print_galerkin_odes(K, C, L, m);

%% Initial condition
rng(1)

u0 = 1e-2*(randn(Nmodes,1)+1i*randn(Nmodes,1));

% Make solution real
for k=1:max(K)
    u0(index_of(-k)) = conj(u0(index_of(k)));
end

u0(index_of(0)) = m; % mean mass

%% Solve in mu space
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

%% L2 norm history
dx = x(2)-x(1);
L2_hist = sqrt(trapz(x,Ux.^2,2)/(2*L*pi));   % length(t) x 1

%% Dominant Fourier mode history
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

% if isnan(mu_select)
%     fprintf('Equilibrium not detected within simulation window.\n');
% else
%     fprintf('Equilibrium (mode selection) first reached at mu = %.4f\n', mu_select);
% end

k_ref = 0:6;     % integer Galerkin modes |k| = 0..3 matter most, but keep a few extra for context

%% Build the movie 
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
    plot(x, Ux(j,:), 'LineWidth', 1.5, 'Color', [0.18 0.45 0.69])
    xlabel('x'); ylabel('u(x)')
    title(sprintf('Real-space solution, \\mu = %.3f', mu_now))
    ylim(ylim_real)
    grid on
    
    % Subplot 2: Fourier spectrum at current mu (Galerkin coefficients)
    subplot(2,2,2)
    stem(K, abs(U(j,:)), 'LineWidth', 1.2)
    yscale("log");
    xlabel('k'); ylabel('|U_k|')
    title(sprintf('Fourier spectrum, \\mu = %.3f', mu_now))
    xlim([min(K)-0.5 max(K)+0.5])
    ylim_spec = [0 max(abs(U(:)))*1.1 + 1e-12];
    ylim(ylim_spec)
    grid on
    
    % Subplot 3: dominant wavenumber vs mu (history up to current frame)
    subplot(2,2,3)
    stairs(mu_vec(1:j), kdom_hist(1:j), 'LineWidth', 2, 'Color', [0.49 0.18 0.56])
    hold on
    % for kj = k_ref
    %     yline(kj, '--r', 'Alpha', 0.3)
    % end
    % if ~isnan(mu_select) && mu_now >= mu_select
    %     xline(mu_select, '-g', 'LineWidth', 1.5, ...
    %         'Label', sprintf('eq. at \\mu=%.2f', mu_select), 'LabelVerticalAlignment','bottom')
    % end
    hold off
    xlabel('\mu'); ylabel('dominant |k|')
    xlim([mu_vec(1) mu_vec(end)])
    ylim(ylim_kdom)
    title('Dominant wavenumber vs \mu')
    grid on
    
    % Subplot 4: L2 norm vs mu (history up to current frame)
    subplot(2,2,4)
    plot(mu_vec(1:j), L2_hist(1:j), 'LineWidth', 1.5, 'Color', [0.13 0.55 0.55])
    hold on
    % if ~isnan(mu_select) && mu_now >= mu_select
    %     xline(mu_select, '--g', 'LineWidth', 1.5, ...
    %         'Label', sprintf('eq. \\mu=%.2f', mu_select), 'LabelVerticalAlignment','bottom')
    %     eq_idx = find(mu_vec(1:j) >= mu_select, 1);
    %     if ~isempty(eq_idx)
    %         plot(mu_vec(eq_idx), L2_hist(eq_idx), 'og', 'MarkerSize', 8, 'MarkerFaceColor','g')
    %     end
    % end
    hold off
    xlabel('\mu'); ylabel('||u||_2')
    xlim([mu_vec(1) mu_vec(end)])
    ylim(ylim_L2)
    title('L_2 norm vs \mu')
    grid on
    
    sgtitle(sprintf('Galerkin Cahn-Hilliard    \\epsilon=%.4f    \\mu(t)=\\epsilon t    \\mu = %.3f / %.3f    m = %.3f', ...
                     epsilon, mu_now, mu_vec(end), m)) 
    
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
plot(x, Ux(end,:), 'LineWidth', 1.5, 'Color', [0.18 0.45 0.69])
xlabel('x'); ylabel('u(x)')
title(sprintf('Real-space solution, \\mu = %.3f', mu_vec(end)))
grid on

subplot(2,2,2)
stem(K, abs(U(end,:)), 'filled', 'LineWidth', 1.2, 'Color', [0.85 0.33 0.10])
xlabel('k'); ylabel('|U_k|')
title('Fourier spectrum (final \mu)')
grid on

subplot(2,2,3)
stairs(mu_vec, kdom_hist, 'LineWidth', 2, 'Color', [0.49 0.18 0.56])
hold on
% for kj = k_ref
%     yline(kj, '--r', 'Alpha', 0.3)
% end
% if ~isnan(mu_select)
%     xline(mu_select, '-g', 'LineWidth', 1.5, ...
%         'Label', sprintf('eq. at \\mu=%.2f', mu_select), 'LabelVerticalAlignment','bottom')
% end
hold off
xlabel('\mu'); ylabel('dominant |k|')
title('Dominant wavenumber vs \mu')
grid on

subplot(2,2,4)
plot(mu_vec, L2_hist, 'LineWidth', 1.5, 'Color', [0.13 0.55 0.55])
hold on
% if ~isnan(mu_select)
%     xline(mu_select, '--g', 'LineWidth', 1.5, ...
%         'Label', sprintf('eq. \\mu=%.2f', mu_select), 'LabelVerticalAlignment','bottom')
%     eq_idx = find(mu_vec >= mu_select, 1);
%     plot(mu_vec(eq_idx), L2_hist(eq_idx), 'og', 'MarkerSize', 8, 'MarkerFaceColor','g')
% end
hold off
xlabel('\mu'); ylabel('||u||_2')
title('L_2 norm vs \mu')
grid on

sgtitle(sprintf('\\epsilon=%.4f  mu\\_final=%.2f  L=%d', epsilon, mu_final, L))

saveas(fig2, 'galerkin_ch_mu_final.png')
% close(fig2);


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
        
        if K(kk)==0
            dudt(kk)=0;
            continue
        end
    end
end


function mu_select = find_selection_time(kdom_hist, mu_vec, window)
% find_selection_time  First mu at which the dominant wavenumber has held
% a constant value for `window` consecutive samples in a row

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


function print_galerkin_odes(K, C, L, m)
    
    Nmodes = length(K);
    
    fprintf('\n========================================================\n');
    fprintf(' Galerkin-projected Cahn-Hilliard system (modes |k| <= %d)\n', max(abs(K)));
    fprintf(' PDE:  u_t = -d_xx( d_xx u + mu(t) u - u^3 )\n');
    fprintf(' Domain half-length L = %g  (Lx = %g*pi)\n', L, L);
    fprintf('========================================================\n\n');
    
    for kk = 1:Nmodes
        k = K(kk);
        
        % collect & group nonlinear (cubic) terms from C
        terms = containers.Map('KeyType','char','ValueType','double');
        
        for mm = 1:Nmodes
            for nn = 1:Nmodes
                for pp = 1:Nmodes
                    if C(kk,mm,nn,pp) == 1
                        triplet = sort([K(mm) K(nn) K(pp)]);
                        key = sprintf('%d_%d_%d', triplet(1), triplet(2), triplet(3));
                        if isKey(terms, key)
                            terms(key) = terms(key) + 1;
                        else
                            terms(key) = 1;
                        end
                    end
                end
            end
        end
        
        keysList = keys(terms);
        if isempty(keysList)
            nl_str = '0';
        else
            nl_parts = cell(1,length(keysList));
            for ii = 1:length(keysList)
                idxs  = sscanf(keysList{ii}, '%d_%d_%d')';
                coeff = terms(keysList{ii});
                nl_parts{ii} = format_monomial(coeff, idxs);
            end
            nl_str = strjoin(nl_parts, ' + ');
        end
        
        % linear coefficient lambda_k(mu) = -k^4/L^4 + mu*k^2/L^2
        lin_str = format_linear(k);
        
        % prefactor on the bracket, -(k^2/L^2)
        if k == 0
            fprintf('du_{%d}/dt = -0/L^2 * [ %s ]      (mean mode: no linear growth term)\n\n', ...
                k, nl_str);
        else
            prefactor_str = sprintf('-%d/L^2', k^2);
            fprintf('du_{%2d}/dt = %s * u_{%d}   +   %s * [ %s ]\n\n', ...
                k, lin_str, k, prefactor_str, nl_str);
        end
    end
    
    fprintf('========================================================\n');
    fprintf('Note: u_{-k} = conj(u_k) is enforced by the reality condition;\n');
    fprintf('      mu is shorthand for mu(t) = epsilon*t.\n');
    fprintf('========================================================\n\n');
    
end


%% formatting
function s = format_monomial(coeff, idxs)
    
    nzero = sum(idxs==0); % count how many zero modes appear
    idxs = idxs(idxs~=0); % remove them
    
    % base numeric coefficient string (suppress "1" if m will be present)
    if coeff == 1 && nzero > 0
        coeffstr = '';
    else
        coeffstr = sprintf('%d', coeff);
    end
    
    % coefficient gets multiplied by m^nzero
    if nzero == 1
        if isempty(coeffstr), coeffstr = 'm'; else, coeffstr = sprintf('%s*m', coeffstr); end
    elseif nzero == 2
        if isempty(coeffstr), coeffstr = 'm^2'; else, coeffstr = sprintf('%s*m^2', coeffstr); end
    elseif nzero == 3
        if isempty(coeffstr), coeffstr = 'm^3'; else, coeffstr = sprintf('%s*m^3', coeffstr); end
    end
    
    if isempty(idxs)
        monomial = '';
    else
        uvals = unique(idxs);
        parts = {};
        
        for ii=1:length(uvals)
            v = uvals(ii);
            cnt = sum(idxs==v);
            
            if cnt==1
                parts{end+1} = sprintf('u_{%d}',v);
            else
                parts{end+1} = sprintf('u_{%d}^%d',v,cnt);
            end
        end
        
        monomial = strjoin(parts,'*');
    end
    
    if isempty(monomial)
        if isempty(coeffstr)
            s = '1';
        else
            s = coeffstr;
        end
    elseif isempty(coeffstr)
        s = monomial;
    else
        s = sprintf('%s*%s',coeffstr,monomial);
    end
end

function s = format_linear(k)
    % Format lambda_k(mu) = -k^4/L^4 + mu*k^2/L^2 as an exact symbolic expression in L.

    if k == 0
        s = '( 0 )';
    else
        s = sprintf('( -%d/L^4 + %d/L^2*mu )', k^4, k^2);
    end
end


function rfun = r_k(k, L, epsilon, t0)
    % closed-form solution
    
    const_term = @(t) (3*sqrt(pi)*k)/(sqrt(epsilon)*L) * exp(-k^6/(epsilon*L^6)) * ...
            exp( (2*k^4/(L^4))*t - (epsilon*k^2/(L^2)).*t.^2 );
        
    img_term = @(t) ( erfi((sqrt(epsilon)*k/L)*t - k^3/(sqrt(epsilon)*L^3)) ...
                    - erfi((sqrt(epsilon)*k/L)*t0 - k^3/(sqrt(epsilon)*L^3)) );
                
    IC_term = @(t) exp( (2*k^4/(L^4))*t - (epsilon*k^2/(L^2)).*t.^2 - (2*k^4/(L^4))*t0 + (epsilon*k^2/(L^2)).*t0.^2);
    
    % IC
    r_k_0 = 10; %(const_term(t0).*img_term(t0)/(1-IC_term(t0)))^(-1/2); % must be nonzero
    
    rfun = @(t) (const_term(t).*img_term(t) + r_k_0^(-2)*IC_term(t)).^(-1/2);
end

function [MUS, KS] = critical_bifurcation_eig(j, L, mass, mu0)
    % Computes the j'th frequencies critical mu and associated eigenvalue
    KS = 2 * pi * j./ L;
    MUS = -mu0 + 2*(3 * mass^2 + KS.^2);    
end

function [MUS, KS] = critical_bifurcation(j, L, mass)
    % Computes the j'th frequencies critical mu and associated eigenvalue
    KS = 2 * pi * j./ L;
    MUS = KS.^2 + 3*mass^2;    
end

%% Print Critical Bifurcations
fprintf('========================================================\n');
fprintf(' Theoretical Critical Bifurcations (integrated eig)\n');
fprintf('========================================================\n');
for kk = 1:Nmodes
    k_val = K(kk);
    if k_val >= 0 % Only print for positive wave numbers
        [mu_crit, ks_val] = critical_bifurcation_eig(k_val, L, m, mu0);
        fprintf('Mode |k| = %2d : Critical mu = %.4f\n', k_val, mu_crit);
    end
end
fprintf('\n');

fprintf('========================================================\n');
fprintf(' Theoretical Critical Bifurcations (eigenvalue)\n');
fprintf('========================================================\n');
for kk = 1:Nmodes
    k_val = K(kk);
    if k_val >= 0 % Only print for positive wave numbers
        [mu_crit, ks_val] = critical_bifurcation(k_val, L, m);
        fprintf('Mode |k| = %2d : Critical mu = %.4f\n', k_val, mu_crit);
    end
end
fprintf('\n');