%% Predict the high amplitude onset time and resulting mode

% PARAMETERS
delta = 2^(-10); 
amp_thr = 1/sqrt(20 * pi);


% FUNCTIONS
ktilde2 = @(k) (k / 10)^2; 

% Use element-wise operations (.*, ./, .^) for grid evaluation
Lambdak = @(mu, k) (mu.^2 .* ktilde2(k) / 2 - mu .* ktilde2(k)^2);

a = @(eps) (ktilde2(3) - 3 * ktilde2(1)) ./ (2 * eps);
b = @(eps) (3 * ktilde2(1)^2 - ktilde2(3)^2) ./ eps;

sigma = @(eps) 1 ./ sqrt(a(eps));
mean_val = @(eps) -b(eps) ./ (2 * a(eps)); 
% Onset surface equations
muk_hom_onset =  @(eps, mu0, k) ktilde2(k) + sqrt(ktilde2(k)^2 + 2 / ktilde2(k) .* ...
    (-eps .* log(abs(delta)) + eps .* log(amp_thr) + Lambdak(mu0, k)));

mu1_onset = @(eps, mu0) muk_hom_onset(eps, mu0, 1);
mu2_onset = @(eps, mu0) muk_hom_onset(eps, mu0, 2);
mu3_hom_onset = @(eps, mu0) muk_hom_onset(eps, mu0, 3);
mu4_onset = @(eps, mu0) muk_hom_onset(eps, mu0, 4);
mu5_onset = @(eps, mu0) muk_hom_onset(eps, mu0, 5);

   
    
C = @(eps, mu0) sqrt(pi ./ (4 * a(eps))) ./ eps .* ktilde2(3) .* delta^3 .* ...
    exp(b(eps).^2 ./ (4 * a(eps)) - 3 * Lambdak(mu0, 1) ./ eps) .* ...
    (1 - erf((mu0 - mean_val(eps)) ./ sigma(eps)));
    
mu3_res_onset =  @(eps, mu0) ktilde2(3) + sqrt(ktilde2(3)^2 + 2 / ktilde2(3) .* ...
    (eps .* log(amp_thr) - eps .* log(C(eps, mu0))));

% GRID SETUP
% Create a dense grid over the specified domain
eps_vec = logspace(-6, -2, 1200); 
mu0_vec = linspace(-0.5, 0, 1200);
[Eps, Mu0] = meshgrid(eps_vec, mu0_vec);

% Evaluate surfaces
M1    = evaluate_real(mu1_onset, Eps, Mu0);
M2    = evaluate_real(mu2_onset, Eps, Mu0);
M3_h  = evaluate_real(mu3_hom_onset, Eps, Mu0);
M3_r  = evaluate_real(mu3_res_onset, Eps, Mu0);
M4 = evaluate_real(mu4_onset, Eps, Mu0);
M5 = evaluate_real(mu5_onset, Eps, Mu0);

% Stack surfaces and find the minimum value and its corresponding index (mode)
AllSurfaces = cat(3, M1, M2, M3_h, M3_r, M4, M5);
[MinSurface, MinIdx] = min(AllSurfaces, [], 3, 'omitnan');



%% PLOT 1: Make a contour plot of the minimum surface for the when each
% approximation hits L'_thr
figure('Position', [200 200 800 600], 'Resize', true);
hold on;

% 1. Plot the regions identifying which mode is the minimum (This shows the ridges)
% We use slightly transparent colors to act as a background map
[~, h_reg] = contourf(Eps, Mu0, MinIdx, [1 2 3 4 5 6], 'LineStyle', 'none');
alpha(0.3); 

% 2. Plot the contour lines of the minimum surface itself
[C_lines, h_lines] = contour(Eps, Mu0, MinSurface, 20, 'k', 'LineWidth', 1);
clabel(C_lines, h_lines, 'FontSize', 8, 'Color', 'k'); % Add labels to the contour lines

% 3. Plot simulated data
grid_data = load('dominate_modes_grid.mat');
[Eps_scatter, Mu0_scatter] = meshgrid(grid_data.EP, grid_data.MU0_vec);

x_points = Eps_scatter(:);
y_points = Mu0_scatter(:);
c_points = grid_data.DMODE(:) * grid_data.Lx / pi;

valid = ~isnan(c_points);
x_valid = x_points(valid);
y_valid = y_points(valid);
c_valid = c_points(valid);

% Map points to color scale
c_mapped = c_valid;

% Shift Mode 4 and higher
shift_mask = c_valid >= 4;
c_mapped(shift_mask) = c_valid(shift_mask) + 1;

% Disambiguate Mode 3 (Index 3 vs Index 4)
mode3_mask = (c_valid == 3);
eps_3 = x_valid(mode3_mask);
mu0_3 = y_valid(mode3_mask);

% Evaluate theoretical onset times for the scatter coordinates
val_3_hom  = evaluate_real(mu3_hom_onset, eps_3, mu0_3);
val_3_part = evaluate_real(mu3_res_onset, eps_3, mu0_3);

% If partial onset is smaller (earlier), it falls in the partial regime (Index 4)
is_partial = val_3_part < val_3_hom; 

% Apply the mapping
idx_3 = find(mode3_mask);
c_mapped(idx_3(is_partial)) = 4;
c_mapped(idx_3(~is_partial)) = 3;

scatter(x_valid, y_valid, 20, c_mapped, ...
    'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1, ...
    'MarkerFaceAlpha', 0.6, ... % Face transparency
    'MarkerEdgeAlpha', 0.2);
    
% Expand clim slightly in case the simulation found mode 5 or 6
max_mapped_color = max([6; c_mapped]); 
clim([1, max_mapped_color]); 

% Make sure your colormap has enough colors if DMODE > 4 exists
colors = colormap(lines(max_mapped_color));

% 4. Formatting
set(gca, 'XScale', 'log');
xlim([1e-6, 1e-2]);
ylim([-0.5, 0]);

xlabel('$\epsilon$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\mu_0$', 'Interpreter', 'latex', 'FontSize', 14);
title('Contours of Minimum Onset $\mu$ and Dominant Modes', 'Interpreter', 'latex', 'FontSize', 16);

% 5. Custom Legend for the regions
p = gobjects(1, length(6));
for i = 1:6
    p(i) = plot(NaN, NaN, 's', 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'k', 'MarkerSize', 10);
end

legend(p, {'\mu_1 Onset', '\mu_2 Onset', '\mu_3 Homogeneous', '\mu_3 Resonant', '\mu_4 Homogeneous', '\mu_5 Homogeneous'}, ...
       'Location', 'best', 'Interpreter', 'tex');
   
grid on;
box on;
hold off;

%% Asch paper
% Solving b1(mu) = 0
mu_out_b1 =  @(mu0) ktilde2(1) + sqrt(ktilde2(1)^2 + 2 / ktilde2(1) .* ...
    Lambdak(mu0, 1));

% mu for when lambda3 = 3 * lambda1
mu_13_res = 13 * ktilde2(1);

% Solving b3(mu) = 0 (Fixed Lambdak(mu_13_res, 3))
mu_out_b3_res = @(mu0) ktilde2(3) + sqrt(ktilde2(3)^2 + 2 / ktilde2(3) .* ...
    (Lambdak(mu_13_res, 3) - 3 * Lambdak(mu_13_res, 1) + 3 * Lambdak(mu0, 1)));

%% PLOT 2: Plot the Asch condition as a function of mu0
figure('Position', [200 200 800 600], 'Resize', true);
hold on;

plot(mu0_vec, mu_out_b1(mu0_vec), 'Color', colors(1,:), 'DisplayName', 'b_1');

b3_res = mu_out_b3_res(mu0_vec);
after_res = b3_res  > mu_13_res;
plot(mu0_vec(after_res), b3_res(after_res), 'Color', colors(4,:), 'DisplayName', 'b_{3, res}');

yline(mu_13_res, '--', 'Color', colors(4,:), 'DisplayName', '\mu_{1^3:3}', 'Interpreter', 'tex')

title('Asch Paper Algorithm')
xlabel('\mu_0')
ylabel('\mu_{out}')

legend('Location', 'best');
grid on;
hold off;


% HELPER FUNCTION
function Z = evaluate_real(func, X, Y)
    % Evaluates the function and masks imaginary results as NaN
    % This prevents invalid onset times from affecting the minimum calculation
    Z = func(X, Y);
    imag_mask = imag(Z) ~= 0;
    Z(imag_mask) = NaN;
    Z = real(Z);
end