function galerkin_proj_large_onset_analysis()
    %% This scripts attempts to predict the high amplitude onset time and 
    % resulting mode and compares it to solution data.
    saved = load('dominate_modes.mat');

    data_epsilon = saved.EP;
    selected_mode_thr = saved.DMODE * saved.Lx / pi;
    mu_thr = saved.MUTHR;

    epsilon = 10.^[-6:0.01:-2];
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
    nu4 = @(mu) nu(mu, 4);

    % Find initial conditions
    r0 = ones(1, 4) / saved.Nx; 
    
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
    mu4_hom_large = ktilde2(4) + sqrt(ktilde2(4)^2 + 2 / ktilde2(4) *  ...
        (-epsilon * log(abs(r0(4))) + epsilon * log(amp_thr) + epsilon .* nu4(mu0)));

    C = sqrt(pi ./ (4 * a)) ./ epsilon * ktilde2(3) * r0(1)^3 .* ...
        exp(b.^2 ./ (4 * a) - 3 * nu1(mu0)) .* (1 - erf((mu0 - mean) ./ sigma));

    mu3_part_large = ktilde2(3) + sqrt(ktilde2(3)^2 + 2 / ktilde2(3) * ...
     (epsilon .* log(amp_thr) - epsilon .* log(C)));
    
    
    % --- PLOT THE COMPARISON ---
    figure('Position', [200 200 800 500], 'Resize', false);
    theme("light");    
    hold on;
    
    num_colors = max(selected_mode_thr);
    color_palette = lines(num_colors); 
   
    %% Apply the color palette and lock the values so 1=Color 1, 3=Color 3
    colormap(color_palette);
    clim([1, num_colors]); 
    
    % Setup the colorbar
    cb = colorbar;
    ylabel(cb, 'Selected Mode', 'FontSize', 12);
    cb.Ticks = 1:num_colors;
    
    %% Plot the analytical lines
    % Grab the handles so we can make a clean legend later
    h1 = semilogx(epsilon, mu1_large, '-', 'Color', color_palette(1, :), 'LineWidth', 2);
    
    h2 = semilogx(epsilon, mu2_large, '-', 'Color', color_palette(2, :), 'LineWidth', 2);
    
    h3_hom = semilogx(epsilon, mu3_hom_large, '-', 'Color', color_palette(3, :), 'LineWidth', 2);
    
    h3_part = semilogx(epsilon, mu3_part_large, '--', 'Color', color_palette(3, :), 'LineWidth', 2);
   
    h4_hom = semilogx(epsilon, mu4_hom_large, '-', 'Color', color_palette(4, :), 'LineWidth', 2);

    %% Plot the discrete data (Scatter Plot)
    scatter(data_epsilon, mu_thr, 10, selected_mode_thr, 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.7);


    %% Formatting the plot
    set(gca, 'XScale', 'log'); % Ensure the whole axis is log-scaled for epsilon
    grid on;
    
    xlabel('\epsilon (Log Scale)', 'FontSize', 12);
    ylabel('\mu_{thr}', 'FontSize', 12);
    title('Large amplitude onset time and selected mode', 'FontSize', 14);
    
    %% Add a clean legend just for the lines
    legend([h1, h2, h3_hom, h3_part, h4_hom], ...
        '\mu_1 approx', '\mu_2 approx', '\mu_3 hom approx', '\mu_3 part approx', '\mu_4 hom approx', ...
        'Location', 'northwest', 'FontSize', 10);
    
    hold off;
end