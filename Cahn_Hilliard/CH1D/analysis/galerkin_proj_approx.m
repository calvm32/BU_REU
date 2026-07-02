function galerkin_proj_approx()
    %% This script attempts to numerically solve the Galerkin projection 
    % for the first three modes. 
    saved = load('dom_mode_ep=0.00100_mu0=-0.30.mat');
    sol = saved.sol;
    solution = squeeze(sol.solution);
    epsilon = saved.ep;
    mu0 = saved.mu0;
    mu_max = 0.8;
    
    ktilde2 = @(k) (k / 10)^2; 

    lambda = @(mu, k) mu * ktilde2(k) - ktilde2(k)^2;
    lambda1 = @(mu) lambda(mu, 1);
    lambda2 = @(mu) lambda(mu, 2);
    lambda3 = @(mu) lambda(mu, 3);

    nu = @(mu, k) 1/epsilon * (mu.^2 * ktilde2(k) / 2 - mu * ktilde2(k)^2);
    nu1 = @(mu) nu(mu, 1);
    nu2 = @(mu) nu(mu, 2);
    nu3 = @(mu) nu(mu, 3);
    
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

    % Find initial conditions
    r0 = [r1_data(1); r2_data(1); r3_data(1)]; 
    
    % Some constants 
    a = (ktilde2(3) - 3 * ktilde2(1)) / (2 * epsilon);
    b = (3 * ktilde2(1)^2 - ktilde2(3)^2) / epsilon;
    
    % The "std" and "mean" of our "normal distribution"
    sigma = 1 / sqrt( a);
    mean = - b / (2 * a);
    

    

    r1_est = r0(1) * exp(nu1(mu_grid) - nu1(mu0));
    r2_est = r0(2) * exp(nu2(mu_grid) - nu2(mu0));
    r3_hom = exp(nu3(mu_grid) - nu3(mu0)) .* r0(3);

    C3 = sqrt(pi / (4 * a)) / epsilon * ktilde2(3) * r0(1)^3 * exp(b^2 / (4 * a) - 3* nu1(mu0));
    r3_particular = C3 * exp(nu3(mu_grid)) .* ...
        (erf( (mu_grid - mean) / sigma ) - erf( (mu0 - mean) / sigma ));
    r3_est = r3_hom + r3_particular;
    
    % Surpassing times
    % 1, 2 surpassing time
    A = -3 / 2 * ktilde2(1);
    B = 15 * ktilde2(1)^2;
    C = epsilon * log(abs(r0(1))) - epsilon * log(abs(r0(2))) - A * mu0^2 - B * mu0;
    mu_12 = max(roots([A, B, C]));

    % 1, 3 surpassing time
    A = -4 * ktilde2(1);
    B = 80 * ktilde2(1)^2;
    C0 = C3 / r0(1) * (1 - erf( (mu0 - mean) / sigma ));
    C = -epsilon * log(C0) - epsilon * nu1(mu0);
    mu_13 = max(roots([A, B, C]));
    
    % --- PLOT THE COMPARISON ---
    figure('Position', [200 200 800 500]);
    theme("light");    
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
        plot(mu_grid, abs(est_data{k}), '--', 'Color', colors(k, :), 'LineWidth', 2);
        legend_entries{3 + k} = sprintf('Approx k=%d', k);
    end
    hold off;
    
    % Plot surpassing times
    xline(max(mu_12), '-.', 'Color', colors(2, :), 'LineWidth', 2, 'Label', '\mu_{12}');
    xline(max(mu_13), '-.', 'Color', colors(3, :), 'LineWidth', 2, 'Label', '\mu_{13}');

    set(gca, 'YScale', 'log');
    xlabel('\mu', 'FontSize', 12);
    ylabel('$|\hat{u}_k|$', 'FontSize', 12, 'Interpreter', 'latex');
    title(sprintf('PDE Data vs Galerkin Approximation (\\epsilon = %.4f)', epsilon), 'FontSize', 14);
    
    legend(legend_entries, 'Location', 'SouthEast', 'FontSize', 9, 'NumColumns', 2);
    grid on;
    set(gca, 'FontSize', 11);
    xlim([mu0, mu_max]);
end