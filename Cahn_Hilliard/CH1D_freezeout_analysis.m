clc; clear; close all;

%% Parameters
epsilon_list = linspace(0.01, 0.1, 10);

plot_diff = false;

t0 = -2.0;
dt = 0.1;

scale = 100;
Lx = scale*pi;

Nx = 2^12;
dx = Lx/Nx;

%% storage for final plots
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

%% loop over epsilon
for eidx = 1:length(epsilon_list)

    epsilon = epsilon_list(eidx);

    x = (-Nx/2:Nx/2-1)*dx;

    kx = 2*pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
    Laplacian_hat = -kx.^2;
    L_operator = -(Laplacian_hat.^2);

    blowup_time = (epsilon/2)^(-2/3); %theoretical_freezeout_time(epsilon,kx);

    T = blowup_time + 60.0;

    t = t0:dt:T;
    num_time_steps = length(t);

    %% tracking average for each epsilon
    density_comput_avg = zeros(1,num_time_steps);
    density_theory_avg = zeros(1,num_time_steps);

    freezeout_time_comput_avg = 0;
    freezeout_time_theory_avg = 0;

    usq_avg = zeros(Nx,1);

    %% average runs for each epsilon
    avg_num = 10;

    for run_idx=1:avg_num

        %% initial condition
        sigma = 0.01;

        rng(100000*eidx + run_idx,'twister');

        u = sigma * randn(1,Nx);
        u_hat = fft(u);

        mu = @(t) t*epsilon;

        %% tracking once for each epsilon
        density_comput_eps = zeros(1,num_time_steps);
        density_theory_eps = zeros(1,num_time_steps);

        freezeout_time_comput_eps = 0;
        freezeout_time_theory_eps = blowup_time;

        %% ETDRK4
        E  = exp(dt*L_operator);
        E2 = exp(dt*L_operator/2);

        M = 32;
        r = exp(1i*pi*((1:M)-0.5)/M);
        Lvec = L_operator(:);
        LR = dt*Lvec(:,ones(M,1)) + r(ones(numel(L_operator),1),:);

        Q  = dt*real(mean((exp(LR/2)-1)./LR,2)).';
        f1 = dt*real(mean((-4-LR + exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2)).';
        f2 = dt*real(mean((2+LR + exp(LR).*(-2+LR))./LR.^3 ,2)).';
        f3 = dt*real(mean((-4-3*LR-LR.^2 + exp(LR).*(4-LR))./LR.^3 ,2)).';

        clear LR Lvec

        %% time loop
        for n = 2:num_time_steps

            t_prev = t(n-1);
            t_half = t_prev + dt/2;
            t_curr = t(n);

            u3 = u.^3 - mu(t_prev)*u;
            Nu_hat = Laplacian_hat .* fft(u3);

            a_hat = E2.*u_hat + Q.*Nu_hat;
            a = real(ifft(a_hat));
            Na_hat = Laplacian_hat .* fft(a.^3 - mu(t_half)*a);

            b_hat = E2.*u_hat + Q.*Na_hat;
            b = real(ifft(b_hat));
            Nb_hat = Laplacian_hat .* fft(b.^3 - mu(t_half)*b);

            c_hat = E2.*a_hat + Q.*(2*Nb_hat - Nu_hat);
            c = real(ifft(c_hat));
            Nc_hat = Laplacian_hat .* fft(c.^3 - mu(t_curr)*c);

            u_hat = E.*u_hat + f1.*Nu_hat + 2*f2.*(Na_hat + Nb_hat) + f3.*Nc_hat;
            u = real(ifft(u_hat));

            %% diagnostics
            density_theory_eps(n) = domainwall_density_theory(t_curr, u_hat, Laplacian_hat, epsilon);
            density_comput_eps(n) = domainwall_density_comput(u, Lx);

            mask = t > 10;
            density_masked = density_comput_eps(mask);

            max_density = max(density_masked);
            tol = 1e-4*max_density;

            mask_idx = find(mask);
            freeze_idx_local = find(density_masked >= max_density - tol,1,'first');

            if ~isempty(freeze_idx_local)
                freeze_idx = mask_idx(freeze_idx_local);
                freezeout_time_comput_eps = t(freeze_idx);
            end

            usq_avg = usq_avg + abs(u_hat(:)).^2/avg_num;

        end

        %% running averages
        density_comput_avg = density_comput_avg + density_comput_eps/avg_num;
        density_theory_avg = density_theory_avg + density_theory_eps/avg_num;

        freezeout_time_comput_avg = freezeout_time_comput_avg + freezeout_time_comput_eps/avg_num;
        freezeout_time_theory_avg = freezeout_time_theory_avg + freezeout_time_theory_eps/avg_num;

        k_theory(eidx) = (epsilon/2)^(1/6);

    end

    %% POST PROCESS
    [~,cut] = min(abs(t - blowup_time));

    density_comput(eidx) = density_comput_avg(cut);
    density_theory(eidx) = density_theory_avg(cut);
    density_diff(eidx) = density_theory_avg(cut) - density_comput_avg(cut);

    freezeout_time_comput(eidx) = freezeout_time_comput_avg;
    freezeout_time_theory(eidx) = freezeout_time_theory_avg;
    freezeout_time_diff(eidx) = freezeout_time_theory_avg - freezeout_time_comput_avg;

    kshift = fftshift(kx);
    spec = fftshift(usq_avg);

    [~,idx] = max(spec);

    k_comput(eidx) = abs(kshift(idx));
    k_diff(eidx) = k_theory(eidx) - k_comput(eidx);

    fprintf("epsilon = %.3f done\n", epsilon);

end

%% FINAL PLOTS

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
    loglog(eps_out,density_comput,'o-'); hold on;
    loglog(eps_out,density_theory,'s-');
    xlabel('\epsilon');
    ylabel('Domain wall density');
    title('Domain Wall Density');
    legend('comput','theory', 'Location', 'northwest');
    grid on;

    subplot(1,3,2)
    loglog(eps_out,freezeout_time_comput,'o-'); hold on;
    loglog(eps_out,freezeout_time_theory,'k--');
    xlabel('\epsilon');
    ylabel('freeze-out time');
    title('Freeze-Out Time');
    legend('comput','theory');
    grid on;

    subplot(1,3,3)
    plot(eps_out,k_comput,'o-'); hold on;
    plot(eps_out,k_theory,'s-');
    xlabel('\epsilon');
    ylabel('k_{max}');
    title('Highest-Growing Wave Modes');
    legend('comput','theory');
    grid on;

end

%% FUNCTIONS

function P = power_spec(t,u_hat,Laplacian_hat,epsilon)
    ksq = -Laplacian_hat;
    P = abs(u_hat).^2 .* exp(t^2*ksq*epsilon - 2*t*ksq.^2);
end

function n = domainwall_density_theory(t,u_hat,Laplacian_hat,epsilon)
    P = power_spec(t,u_hat,Laplacian_hat,epsilon);
    ksq = -Laplacian_hat;

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