clc; clear; close all;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Description
% Uses Galerkin projection of 1D Cahn-Hilliard
%
% u_t = -d_xx(d_xx u + mu(t)u - u^3)
%
% Modes |k|<=3
%
% evolves everything in mu-space: mu(t) = epsilon*t, and all
% dynamic passage through bifurcations in 1D Cahn-Hilliard

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Parameters
epsilon  = 0.02;
mu_final = 1; % 2;
t0       = 0;
T        = mu_final/epsilon;

K = -3:3;
Nmodes = length(K);
index_of = @(k) find(K==k);

L = 8; % half-domain size
Lx = L*pi;

%% Movie / frame parameters
Nframes        = 250;                       
tspan          = linspace(t0,T,Nframes);    
mu_vec         = epsilon*tspan;             

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

% calculate theoretical bifurcation pts
mu_BP = zeros(3,1);
for k = 1:3
    mu_BP(k) = (k/L)^2; 
end

mu_plot = linspace(0, mu_final, 500);
branch_U = zeros(3, length(mu_plot));
for k = 1:3
    valid_idx = mu_plot >= mu_BP(k);
    branch_U(k, valid_idx) = sqrt((mu_plot(valid_idx) - mu_BP(k)) / 3);
end

% ICs = slight perturbation
u0 = 1e-3*(randn(Nmodes,1)+1i*randn(Nmodes,1));

% Enforce reality condition
u0(index_of(0)) = real(u0(index_of(0)));
for k=1:3
    u0(index_of(-k)) = conj(u0(index_of(k)));
end

%% Solve the dynamic system (ode45)
opts = odeset('RelTol',1e-8,'AbsTol',1e-10);
[t,U] = ode45(@(t,u) rhs(t,u,epsilon,K,C,L), tspan, u0, opts);

%% Reconstruct real-space solution
Nx = 256;
x  = linspace(-Lx,Lx,Nx);
Ux = zeros(length(t),Nx);

for j=1:length(t)
    temp = zeros(1,Nx);
    for ii=1:Nmodes
        temp = temp + U(j,ii)*exp(1i*K(ii)*x/L);
    end
    Ux(j,:) = real(temp);
end

L2_hist = sqrt(trapz(x,Ux.^2,2)/(2*Lx));

%% Build the movie
vidObj = VideoWriter('CH_Dynamic_Bifurcation.avi','Motion JPEG AVI');
vidObj.FrameRate = 15;
vidObj.Quality   = 95;
open(vidObj);

fig = figure('Position',[100 100 1200 800], 'Color', 'w');

% Precompute limits
ylim_real = [min(Ux(:)) max(Ux(:))]*1.1;
ylim_spec = [0 max(abs(U(:)))*1.1 + 1e-6];
colors    = lines(3); % Colors for modes 1, 2, 3

for j = 1:length(t)
    clf(fig);
    mu_now = mu_vec(j);

    % Subplot 1: Real-space solution
    subplot(2,2,1)
    plot(x, Ux(j,:), 'LineWidth', 2)
    xlabel('x'); ylabel('u(x)')
    title(sprintf('Real-Space Evolution \\quad \\mu = %.3f', mu_now))
    xlim([-Lx Lx]); ylim(ylim_real); grid on; box on;

    % Subplot 2: Fourier Spectrum
    subplot(2,2,2)
    stem(K, abs(U(j,:)), 'filled', 'LineWidth', 1.5)
    xlabel('Wavenumber k'); ylabel('Amplitude |U_k|')
    title('Instantaneous Fourier Spectrum')
    xlim([min(K)-0.5 max(K)+0.5]); ylim(ylim_spec); grid on; box on;

    % Subplot 3: Live Bifurcation Diagram
    subplot(2,2,3)
    hold on;
    % Plot theoretical static diagram
    plot([0 mu_final], [0 0], 'k-', 'LineWidth', 2, 'DisplayName', 'Trivial eq.');
    for k = 1:3
        plot(mu_plot, branch_U(k,:), '--', 'LineWidth', 1.5, 'Color', colors(k,:), ...
            'DisplayName', sprintf('Mode %d Branch', k));
        % Plot Bifurcation Points (BP)
        plot(mu_BP(k), 0, 'ks', 'MarkerFaceColor', colors(k,:), 'MarkerSize', 8, ...
            'HandleVisibility', 'off');
    end
    
    % Plot dynamic tracer points (current state of the simulation)
    plot(mu_now, abs(U(j, index_of(1))), 'o', 'MarkerFaceColor', colors(1,:), 'MarkerEdgeColor', 'k', 'MarkerSize', 10, 'DisplayName', 'Dyn |U_1|');
    plot(mu_now, abs(U(j, index_of(2))), 'o', 'MarkerFaceColor', colors(2,:), 'MarkerEdgeColor', 'k', 'MarkerSize', 10, 'DisplayName', 'Dyn |U_2|');
    plot(mu_now, abs(U(j, index_of(3))), 'o', 'MarkerFaceColor', colors(3,:), 'MarkerEdgeColor', 'k', 'MarkerSize', 10, 'DisplayName', 'Dyn |U_3|');
    
    xlabel('\mu (Bifurcation Parameter)'); ylabel('Steady State Amplitudes |U_k|')
    title('Dynamic Trajectory over Static Bifurcations')
    xlim([0 mu_final]); ylim(ylim_spec); 
    legend('Location','northwest','FontSize',9);
    grid on; box on; hold off;

    % Subplot 4: L2 Norm History
    subplot(2,2,4)
    plot(mu_vec(1:j), L2_hist(1:j), 'LineWidth', 2)
    hold on;
    % Overlay vertical lines for where theoretical BPs occur
    for k = 1:3
        xline(mu_BP(k), 'r:', 'LineWidth', 1.2, 'Alpha', 0.5);
    end
    hold off;
    xlabel('\mu'); ylabel('||u||_2')
    title('Global Energy (L_2 Norm)')
    xlim([0 mu_final]); ylim([0 max(L2_hist)*1.1+1e-4]); 
    grid on; box on;

    sgtitle(sprintf('Cahn-Hilliard Mode Selection: Dynamic Passage Through Bifurcations (\\epsilon = %.3f)', epsilon), 'FontSize', 14, 'FontWeight', 'bold');

    drawnow;
    frame = getframe(fig);
    writeVideo(vidObj, frame);
end

close(vidObj);
close(fig);
fprintf('Movie saved to CH_Dynamic_Bifurcation.avi\n');


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