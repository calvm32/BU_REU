function galerkin_proj_data_varification_explicit_eq()
    %% This script attempts to numerically solve the Galerkin projection 
    % for the first three modes. It does not align well with the data.
    saved = load('dom_mode_ep=0.00032_mu0=-0.20.mat');
    sol = saved.sol;
    solution = squeeze(sol.solution);
    epsilon = saved.EP(2);
    mu0 = saved.mu0;
    mu_max = 0.45;
    
    ktilde2 = @(k) (k / 10)^2; 
    lambda1 = @(mu) mu * ktilde2(1) - ktilde2(1)^2;
    lambda2 = @(mu) mu * ktilde2(2) - ktilde2(2)^2;
    lambda3 = @(mu) mu * ktilde2(3) - ktilde2(3)^2;
    
    % --- EXTRACT AND SHIFT THE TRUE DATA ---
    % sol.solution is assumed to be complex Fourier modes (modes x time)
    % Index 2 is k=1, Index 3 is k=2, Index 4 is k=3
    
    % Find the instantaneous phase of the fundamental mode at each time step
    phi = angle(solution(2, :)); 
    
    % Shift the modes by rotating them by -k * phi. 
    % Then take the real part to get the signed amplitudes.
    r1_data = real(solution(2, :) .* exp(-1i * 1 * phi)) / saved.Nx;
    r2_data = real(solution(3, :) .* exp(-1i * 2 * phi)) / saved.Nx; 
    r3_data = real(solution(4, :) .* exp(-1i * 3 * phi)) / saved.Nx;
    
    mu_grid = mu0 + epsilon * sol.time;

    % --- SET UP AND SOLVE THE ODE ---
    r0 = [r1_data(1); r2_data(1); r3_data(1)]; 
    tspan = [mu0, mu_max];
    
    options = odeset('RelTol', 1e-12, 'AbsTol', 1e-14);
    [mu_mesh, r_sol] = ode45(@galerkin, tspan, r0, options);
    
    r1_est = r_sol(:, 1)';
    r2_est = r_sol(:, 2)';
    r3_est = r_sol(:, 3)';
    
    % --- PLOT THE COMPARISON ---
    figure('Position', [200 200 800 500]);
    hold on;
    
    % Match the exact dynamic color palette framework of the other file
    colors = lines(3); 
    legend_entries = cell(1, 6);
    
    % Pack data arrays into cells to eliminate the if/else blocks
    pde_data = {r1_data, r2_data, r3_data};
    est_data = {r1_est, r2_est, r3_est};
    
    % Plot Full PDE Data (Thick, lighter lines)
    for k = 1:3
        light_color = colors(k, :) + (1 - colors(k, :)) * 0.6;
        plot(mu_grid, abs(pde_data{k}), '-', 'Color', light_color, 'LineWidth', 4);
        legend_entries{k} = sprintf('PDE k=%d', k);
    end
    
    % Plot Galerkin Data (Dashed lines)
    for k = 1:3
        plot(mu_mesh, abs(est_data{k}), '--', 'Color', colors(k, :), 'LineWidth', 2);
        legend_entries{3 + k} = sprintf('Galerkin k=%d', k);
    end
    hold off;
    
    set(gca, 'YScale', 'log');
    xlabel('\mu (Slow Time Scale)', 'FontSize', 12);
    ylabel('Mode Amplitudes |U_k|', 'FontSize', 12);
    title(sprintf('PDE Data vs Galerkin Approximation (\\epsilon = %.4f)', epsilon), 'FontSize', 14);
    
    legend(legend_entries, 'Location', 'SouthEast', 'FontSize', 9, 'NumColumns', 2);
    grid on;
    set(gca, 'FontSize', 11);
    xlim([mu0, mu_max]);
    
    function drdmu = galerkin(mu, r)
        % epsilon d/dt r1 = lambda1(mu) r1 - f(1) [3r1(1r1^2 + 2 r2^2 + 2 r3^2) + 3r3(r2^2 + r1^2)] 
        % epsilon d/dt r2 = lambda2(mu) r2 - f(2) [3r2(2r1^2 + 1 r2^2 + 2 r3^2) + 6r1 r2 r3]
        % epsilon d/dt r2 = lambda3(mu) r3 - f(2) [3r3(2r1^2 + 2 r2^2 + 1 r3^2) + 3r1r2^2 + r1^3]
    
        r1 = r(1);
        r2 = r(2);
        r3 = r(3);
    
        bracket1 = 3*r1*(1*r1^2 + 2*r2^2 + 2*r3^2) + 3*r3*(r2^2 + r1^2);
        bracket2 = 3*r2*(2*r1^2 + 1*r2^2 + 2*r3^2) + 6*r1*r2*r3;
        bracket3 = 3*r3*(2*r1^2 + 2*r2^2 + 1*r3^2) + 3*r1*r2^2 + r1^3;
    
        %% Assemble Derivative Vector (Scaled by 1/epsilon)
        drdmu = zeros(3, 1);
    
        drdmu(1) = (1 / epsilon) * (lambda1(mu) * r1 - ktilde2(1) * bracket1);
        drdmu(2) = (1 / epsilon) * (lambda2(mu) * r2 - ktilde2(2) * bracket2);
        drdmu(3) = (1 / epsilon) * (lambda3(mu) * r3 - ktilde2(3) * bracket3);
    end
end