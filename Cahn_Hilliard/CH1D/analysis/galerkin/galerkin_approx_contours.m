%% Load data
file_list = dir('sample_*.mat');

NUM_FILES = length(file_list);

% Preallocate arrays to maximum possible size for speed
Eps_data    = zeros(NUM_FILES, 1);
Mu0_data    = zeros(NUM_FILES, 1);
mu_out_data = zeros(NUM_FILES, 1);
dmode_data  = zeros(NUM_FILES, 1);
flag_values = false(NUM_FILES, 1); % Logical array for flags

fprintf('Loading %d files...\n', NUM_FILES);

l2_thr = 1e0;
for k = 1:NUM_FILES
    current_file = file_list(k).name;
    data = load(current_file);
    ep = data.ep;      Eps_data(k) = ep;
    mu0 = data.mu0;    Mu0_data(k) = mu0;

    l2_hist = data.L2_HIST;
    dmode_hist = data.DMODE_HIST;

    t_grid = data.dt * 0:length(l2_hist);
    mu_grid = t_grid * ep + mu0;

    % Find where L2 threshold is reached
    j_l2_thr = find(l2_hist > l2_thr & mu_grid > 0, 1);
    if isempty(j_l2_thr)
        [max_val, max_idx] = max(l2_hist);
        fprintf('Max L2 is %.3f, which occurs at mu = %.5f\n', max_val, mu_grid(max_idx));
        disp(ep, mu0);

        fprintf('File %s never reaches the l2 threshold.\n', current_file)
        disp(max(l2_hist(mu_grid > 0)))
        mu_out_data(k) = -1;
        dmode_data(k)  = -1;
        flag_values(k) = true;
        error('strange')
        continue;
    end

    t_l2_thr       = t_grid(j_l2_thr);
    mu_out_data(k) = mu_grid(j_l2_thr);

    % Find dominate mode at high amplitude
    j_dom_mode_at_thr = find(dmode_hist(:, 1) < t_l2_thr, 1, 'last');
    if isempty(j_dom_mode_at_thr)
        j_dom_mode_at_thr = 1; 
    end
    dmode_data(k) = int32(dmode_hist(j_dom_mode_at_thr, 2) * data.Lx / pi);  
    
    % Detect if amplitude flat lines and flag
    amp_hists = data.AMP_HISTS;
    is_float64 = (string(data.datatype) == "Float64");
    has_flatlined = any(abs(amp_hists(:)) < 1e-16); 
    flag_values(k) = (is_float64 && has_flatlined); 
      
    clear data t_grid mu_grid l2_hist dmode_hist
end

disp('Plotting...')

%% Predict the high amplitude onset time and resulting mode

% --- PARAMETERS ---
delta = 2^(-10); 
amp_thr = 0.8e-1; %1/sqrt(20 * pi);


% --- FUNCTIONS ---
ktilde2 = @(k) (k / 10)^2; 

% Use element-wise operations (.*, ./, .^) for grid evaluation
Lambdak = @(mu, k) (mu.^2 .* ktilde2(k) / 2 - mu .* ktilde2(k)^2);

% Onset surface equations
muk_hom_onset =  @(eps, mu0, k) ktilde2(k) + sqrt(ktilde2(k)^2 + 2 / ktilde2(k) .* ...
    (-eps .* log(abs(delta)) + eps .* log(amp_thr) + Lambdak(mu0, k)));

mu1_onset = @(eps, mu0) muk_hom_onset(eps, mu0, 1);
mu2_onset = @(eps, mu0) muk_hom_onset(eps, mu0, 2);
mu3_hom_onset = @(eps, mu0) muk_hom_onset(eps, mu0, 3);
mu4_hom_onset = @(eps, mu0) muk_hom_onset(eps, mu0, 4);
mu5_onset = @(eps, mu0) muk_hom_onset(eps, mu0, 5);

% Particular solutions

a3 = (ktilde2(3) - 3 * ktilde2(1)) / 2;
b3 = 3 * ktilde2(1)^2 - ktilde2(3)^2;

sigma3 = @(eps) sqrt(eps ./ a3);
mean3 = -b3 / (2 * a3); 

a4 = (ktilde2(4) - 2 * ktilde2(1) - ktilde2(2)) / 2;
b4 = ktilde2(2) + 2 * ktilde2(1)^2 - ktilde2(4)^2;

sigma4 = @(eps) sqrt(eps ./ a4);
mean4 = -b4 / (2 * a4); 


C3 = @(eps, mu0) sqrt(pi ./ (4 * a3 .* eps)) .* ktilde2(3).* delta.^3 .* ...
    exp(- 3 * Lambdak(mu0, 1) ./ eps + b3^2 ./ (4 * a3 .* eps)) .* ...
    erfc((mu0 - mean3) ./ sigma3(eps));

mu3_res_onset = @(eps, mu0) ktilde2(3) + sqrt(ktilde2(3)^2 + 2 / ktilde2(3) .* ...
         (eps .* log(amp_thr) - eps .* log(C3(eps, mu0))));


C4 = @(eps, mu0) sqrt(pi ./ (4 * a4 .* eps)) .* ktilde2(4) .* delta.^3 .* ...
    exp(- 2 * Lambdak(mu0, 1) ./ eps - Lambdak(mu0, 2) + b4^2 ./ (4 * a4 .* eps)) .* ...
    erfc((mu0 - mean4) ./ sigma4(eps));

mu4_res_onset = @(eps, mu0) ktilde2(4) + sqrt(ktilde2(4)^2 + 2 / ktilde2(4) .* ...
    (eps .* log(amp_thr) - eps .* log(C4(eps, mu0))));

% --- GRID SETUP ---
% Create a dense grid over the specified domain
eps_vec = logspace(-6, -2, 2500); 
mu0_vec = linspace(-0.5, 0, 2500);
[Eps, Mu0] = meshgrid(eps_vec, mu0_vec);

% Evaluate surfaces
M1    = evaluate_real(mu1_onset, Eps, Mu0);
M2    = evaluate_real(mu2_onset, Eps, Mu0);
M3_h  = evaluate_real(mu3_hom_onset, Eps, Mu0);
M3_r  = evaluate_real(mu3_res_onset, Eps, Mu0);
M4_h = evaluate_real(mu4_hom_onset, Eps, Mu0);
M4_r = evaluate_real(mu4_res_onset, Eps, Mu0);
M5 = evaluate_real(mu5_onset, Eps, Mu0);

% Stack surfaces and find the minimum value and its corresponding index (mode)
AllSurfaces = cat(3, M1, M2, M3_h, M3_r, M4_h, M4_r, M5);
[MinSurface, MinIdx] = min(AllSurfaces, [], 3, 'omitnan');



%% PLOT 1: Make a contour plot of the minimum surface for the when each
% approximation hits L'_thr
figure('Position', [200 200 800 600], 'Resize', true);
hold on;

% 1. Plot the regions identifying which mode is the minimum (This shows the ridges)
% We use slightly transparent colors to act as a background map
[~, h_reg] = contourf(Eps, Mu0, MinIdx, [1 2 3 4 5 6 7], 'LineStyle', 'none');
alpha(0.3); 

% 2. Plot the contour lines of the minimum surface itself
[C_lines, h_lines] = contour(Eps, Mu0, MinSurface, 20, 'k', 'LineWidth', 1);
clabel(C_lines, h_lines, 'FontSize', 8, 'Color', 'k'); % Add labels to the contour lines


%% Plot real data
c_valid = dmode_data;
c_mapped = c_valid;


% Move Mode 4 out of the way (will be split into 5 or 6 later)
c_mapped(c_valid == 4) = 5; 

% Shift Mode 5+ up to clear space for the split regimes
c_mapped(c_valid >= 5) = 7; 

% Disambiguate Mode 3 (Index 3 vs 4)
mode3_mask = (c_valid == 3);
if any(mode3_mask)
    % Extract coordinate subsets for evaluation
    eps_3 = Eps_data(mode3_mask);
    mu0_3 = Mu0_data(mode3_mask);

    % Evaluate physical threshold onsets
    val_3_hom  = evaluate_real(mu3_hom_onset, eps_3, mu0_3);
    val_3_part = evaluate_real(mu3_res_onset, eps_3, mu0_3);

    idx_3 = find(mode3_mask);
    is_particular_3 = (val_3_part < val_3_hom);

    % Map the sub-indices back to c_mapped
    c_mapped(idx_3(is_particular_3))  = 4; % Particular regime
    c_mapped(idx_3(~is_particular_3)) = 3; % Homogeneous regime
end


% Disambiguate Mode 4 (Index 5 vs 6)
mode4_mask = (c_valid == 4);
if any(mode4_mask)
    % Extract coordinate subsets for evaluation
    eps_4 = Eps_data(mode4_mask);
    mu0_4 = Mu0_data(mode4_mask);

    % Evaluate physical threshold onsets
    val_4_hom  = evaluate_real(mu4_hom_onset, eps_4, mu0_4);
    val_4_part = evaluate_real(mu4_res_onset, eps_4, mu0_4);

    % Sub-allocate based on which onset happens earlier
    is_particular_4 = (val_4_part < val_4_hom);
    idx_4 = find(mode4_mask);
    
    % Map the sub-indices back to c_mapped
    c_mapped(idx_4(is_particular_4))  = 6; % Particular regime
    c_mapped(idx_4(~is_particular_4)) = 5; % Homogeneous regime
end

scatter(Eps_data, Mu0_data, 20, c_mapped, ...
    'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1, ...
    'MarkerFaceAlpha', 0.8, ... 
    'MarkerEdgeAlpha', 0.4);

    
% Expand clim slightly in case the simulation found mode 5 or 6
max_mapped_color = max([7; c_mapped]); 
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
p = gobjects(1, length(7));
for i = 1:7
    p(i) = plot(NaN, NaN, 's', 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'k', 'MarkerSize', 10);
end

legend(p, {'\mu_1 Onset', '\mu_2 Onset', '\mu_3 Homogeneous', '\mu_3 Resonant', '\mu_4 Homogeneous', '\mu_4 Resonant', '\mu_5 Homogeneous'}, ...
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
% figure('Position', [200 200 800 600], 'Resize', true);
% hold on;
% 
% plot(mu0_vec, mu_out_b1(mu0_vec), 'Color', colors(1,:), 'DisplayName', 'b_1');
% 
% b3_res = mu_out_b3_res(mu0_vec);
% after_res = b3_res  > mu_13_res;
% plot(mu0_vec(after_res), b3_res(after_res), 'Color', colors(4,:), 'DisplayName', 'b_{3, res}');
% 
% yline(mu_13_res, '--', 'Color', colors(4,:), 'DisplayName', '\mu_{1^3:3}', 'Interpreter', 'tex')
% 
% title('Asch Paper Algorithm')
% xlabel('\mu_0')
% ylabel('\mu_{out}')
% 
% legend('Location', 'best');
% grid on;
% hold off;


% --- HELPER FUNCTION ---
function Z = evaluate_real(func, X, Y)
    % Evaluates the function and masks imaginary results as NaN
    % This prevents invalid onset times from affecting the minimum calculation
    Z = func(X, Y);
    imag_mask = imag(Z) ~= 0;
    Z(imag_mask) = NaN;
    Z = real(Z);
end