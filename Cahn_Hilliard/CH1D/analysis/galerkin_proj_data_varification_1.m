function galerkin_proj_data_varification_1()
    %% This script attempts to numerically solve the Galerkin projection 
    % for the first three modes. It does not align well with the data.
    saved = load('dominant_modes.mat');
    sol = saved.sol;
    solution = squeeze(sol.solution);
    epsilon = saved.EP;
    mu0 = saved.mu0;
    mu_max = 0.45;
    
    f = @(k) (k / 10)^2; 
    lambda1 = @(mu) mu * f(1) - f(1)^2;
    lambda2 = @(mu) mu * f(2) - f(2)^2;
    lambda3 = @(mu) mu * f(3) - f(3)^2;
    
    % --- EXTRACT AND SHIFT THE TRUE DATA ---
    % sol.solution is assumed to be complex Fourier modes (modes x time)
    % Index 2 is k=1, Index 3 is k=2, Index 4 is k=3
    
    % Find the instantaneous phase of the fundamental mode at each time step
    phi = angle(solution(2, :)); 
    
    % Shift the modes by rotating them by -k * phi. 
    % Then take the real part to get the signed amplitudes.
    r1_data = real(solution(2, :) .* exp(-1i * 1 * phi)); % This equals abs()
    r2_data = real(solution(3, :) .* exp(-1i * 2 * phi)); % This can be negative!
    r3_data = real(solution(4, :) .* exp(-1i * 3 * phi)); % This can be negative!
    
    mu_grid = mu0 + epsilon * sol.time;

    % --- SET UP AND SOLVE THE ODE ---
    % IMPORTANT: Do not use [1; 1; 1]. Use the actual shifted initial conditions
    % from your data so the ODE starts at the exact same physical state!
    r0 = [r1_data(1); r2_data(1); r3_data(1)]; 
    tspan = [mu0, mu_max];
    
    options = odeset('RelTol', 1e-9, 'AbsTol', 1e-9);
    [mu_mesh, r_sol] = ode15s(@galerkin, tspan, r0, options);
    
    r1_est = r_sol(:, 1)';
    r2_est = r_sol(:, 2)';
    r3_est = r_sol(:, 3)';
    
    % --- PLOT THE COMPARISON ---
    figure('Color', 'w');
    
    % Plot ODE estimates (using raw signed values, not abs!)
    semilogy(mu_mesh, abs(r1_est), 'b-', 'LineWidth', 2); hold on;
    semilogy(mu_mesh, abs(r2_est), 'b--', 'LineWidth', 2);
    semilogy(mu_mesh, abs(r3_est), 'b-.', 'LineWidth', 2);
    
    % Plot shifted true data
    semilogy(mu_grid, abs(r1_data), 'r-', 'LineWidth', 3);
    semilogy(mu_grid, abs(r2_data), 'r--', 'LineWidth', 3);
    semilogy(mu_grid, abs(r3_data), 'r-.', 'LineWidth', 3);
    hold off;
    
    % Rest of your formatting...
    xlabel('\mu (Slow Time Scale)', 'FontSize', 12);
    ylabel('Signed Amplitudes r_j(\mu)', 'FontSize', 12);
    xlim([-0.2, 0.4])
    title('Phase-Centered Galerkin Comparison', 'FontSize', 13);
    grid on;
    set(gca, 'FontSize', 11);
    
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
    
        %% 4. Assemble Derivative Vector (Scaled by 1/epsilon)
        drdmu = zeros(3, 1);
    
        drdmu(1) = (1 / epsilon) * (lambda1(mu) * r1 - f(1) * bracket1);
        drdmu(2) = (1 / epsilon) * (lambda2(mu) * r2 - f(2) * bracket2);
        drdmu(3) = (1 / epsilon) * (lambda3(mu) * r3 - f(3) * bracket3);
    end
end
