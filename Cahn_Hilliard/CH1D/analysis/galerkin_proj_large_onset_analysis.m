function galerkin_proj_large_onset_analysis()
    %% This script attempts to numerically solve the Galerkin projection 
    % for the first three modes.
    saved = load('dominate_modes.mat');
    epsilon = 10.^linspace(-6, -3, 10);
    mu0 = saved.mu0;

    amp_thr = 1e-1;
    
    ktilde2 = @(k) (k / 10)^2; 

    lambda = @(mu, k) mu * ktilde2(k) - ktilde2(k)^2;
    lambda1 = @(mu) lambda(mu, 1);
    lambda2 = @(mu) lambda(mu, 2);
    lambda3 = @(mu) lambda(mu, 3);

    nu = @(mu, k) 1./epsilon * (mu.^2 * ktilde2(k) / 2 - mu * ktilde2(k)^2);
    nu1 = @(mu) nu(mu, 1);
    nu2 = @(mu) nu(mu, 2);
    nu3 = @(mu) nu(mu, 3);

    % Find initial conditions
    r0 = ones(1, 3) / saved.Nx; 
    
    % Some constants 
    a = (ktilde2(3) - 3 * ktilde2(1)) ./ (2 * epsilon);
    b = (3 * ktilde2(1)^2 - ktilde2(3)^2) ./ epsilon;
    
    % The "std" and "mean" of our "normal distribution"
    sigma = 1 ./ sqrt( a);
    mean = - b ./ (2 * a);
    

    mu1_large = ktilde2(1) + sqrt(ktilde2(1)^2 + 2 / ktilde2(1) *  ...
        (-epsilon * log(abs(r0(1))) + epsilon * log(amp_thr) + epsilon .* nu1(mu0)));
    mu2_large = ktilde2(2) + sqrt(ktilde2(2)^2 + 2 / ktilde2(2) *  ...
        (-epsilon * log(abs(r0(2))) + epsilon * log(amp_thr) + epsilon .* nu2(mu0)));
    mu3_hom_large = ktilde2(3) + sqrt(ktilde2(3)^2 + 2 / ktilde2(3) *  ...
        (-epsilon * log(abs(r0(3))) + epsilon * log(amp_thr) + epsilon .* nu3(mu0)));

    C = sqrt(pi ./ (4 * a)) ./ epsilon * ktilde2(3) * r0(1)^3 .* ...
        exp(b.^2 ./ (4 * a) - 3 * nu1(mu0)) .* (1 - erf((mu0 - mean) ./ sigma));

    mu3_part_large = ktilde2(3) + sqrt(ktilde2(3)^2 + 2 / ktilde2(3) * ...
     (epsilon .* log(amp_thr) - epsilon .* log(C)));
    
    
    % --- PLOT THE COMPARISON ---
    figure('Position', [200 200 800 500]);
    theme("light");    
    hold on;
    
    semilogx(epsilon, mu1_large, 'r-', 'LineWidth', 2); hold on;
    semilogx(epsilon, mu2_large, 'g--', 'LineWidth', 2);
    semilogx(epsilon, mu3_hom_large, 'b-.', 'LineWidth', 2);
    semilogx(epsilon, mu3_part_large, 'm:', 'LineWidth', 2);
    
    % Formatting the plot
    grid on;
    xlabel('\epsilon (Log Scale)', 'FontSize', 12);
    ylabel('\mu_{large}', 'FontSize', 12);
    title('Evolution of \mu_{large} over \epsilon', 'FontSize', 14);
    legend('\mu_1 large', '\mu_2 large', '\mu_3 hom large', '\mu_3 part large', ...
           'Location', 'best');
end