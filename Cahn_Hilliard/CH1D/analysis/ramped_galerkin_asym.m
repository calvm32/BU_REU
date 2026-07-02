clear; clc;

load('dominant_modes.mat')

%% This scripts attempts to estimate solutions to the Galerkin projections
% of the first three modes assuming the amplitude of the first mode is 
% significantly larger than the other two.

%% 1. Define Parameters and Anonymous Function f
f = @(k) (k/10)^2; 
epsilon = EP;
mu0_orig = mu0;
mu0 = -0.1;
mu_max = 0.5;        
num_points = 10000;   % Higher resolution for precise numerical integration

% Initial conditions
r1_0 = 2e-1;          
r2_0 = 5e-4;         
r3_0 = 5e-8;        

%% 2. Generate Grid and Linear Growth Rates
mu = linspace(mu0, mu_max, num_points);

lambda1 = mu .* f(1) - f(1)^2;
lambda2 = mu .* f(2) - f(2)^2;
lambda3 = mu .* f(3) - f(3)^2;

%% 3. Compute r1(\mu) Numerically
% \int_{mu0}^{mu} lambda1(x) dx
int_lambda1 = cumtrapz(mu, lambda1); 
r1 = r1_0 .* exp(int_lambda1 ./ epsilon);

%% 4. Compute r2(\mu) Numerically
% Integrand: \lambda_2(x) - 6*f(2)*r1^2(x)
integrand_r2 = lambda2 - 6 * f(2) * (r1.^2);
int_integrand_r2 = cumtrapz(mu, integrand_r2);

r2 = r2_0 .* exp(int_integrand_r2 ./ epsilon);

%% 5. Compute r3(\mu) Numerically
% Integral from mu0 to mu of Lambda3
Lambda3 = lambda3 - 6 * f(3) * (r1.^2);
int_Lambda3 = cumtrapz(mu, Lambda3);

% Homogeneous component of r3
r3_H = r3_0 .* exp(int_Lambda3 ./ epsilon);

% Particular (driven) component of r3 computed via a stabilized loop
r3_P = zeros(size(mu));

for i = 1:length(mu)
    % Combine the integrals in the exponent BEFORE evaluating exp()
    % This represents: (1/epsilon) * \int_y^{mu(i)} Lambda3(x) dx
    exponent = (int_Lambda3(i) - int_Lambda3(1:i)) ./ epsilon;
    
    % Evaluate the stabilized integrand for all y <= mu(i)
    stabilized_integrand = (r1(1:i).^3) .* exp(exponent);
    
    % Integrate with respect to y (from mu0 up to current mu(i))
    if i > 1
        r3_P(i) = - (f(3) / epsilon) * trapz(mu(1:i), stabilized_integrand);
    else
        r3_P(i) = 0; % Boundary condition at mu0
    end
end

% Final combined exact numerical solution
r3 = r3_H + r3_P;

%% 6. Plotting the Purely Numerical Results
figure('Color', 'w');
semilogy(mu, abs(r1), 'b-', 'LineWidth', 2); hold on;
semilogy(mu, abs(r2), 'r--', 'LineWidth', 2);
semilogy(mu, abs(r3), 'g-.', 'LineWidth', 2);
hold off;

xlabel('\mu (Slow Time Scale)', 'FontSize', 12);
ylabel('Amplitudes r_j(\mu)', 'FontSize', 12);
title('Pure Numerical Evaluation of Integral Forms (cumtrapz)', 'FontSize', 13);
legend('r_1 (Dominant Fundamental)', 'r_2 (Numerical Integral)', 'r_3 (Numerical Integral)', ...
       'Location', 'NorthWest', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 11);


amp = abs(sol.solution);
figure
plot(mu, abs(r1),  'b-', 'LineWidth', 2);hold on;
plot(mu0_orig + epsilon * sol.time, amp(2, :), 'r--', 'LineWidth', 2);
xlim([mu0_orig, MUTHR])
hold off;

figure
semilogy(mu, abs(r2),  'b-', 'LineWidth', 2); hold on;
semilogy(mu0_orig + epsilon * sol.time, amp(3, :), 'r--', 'LineWidth', 2);
xlim([mu0_orig, MUTHR])
hold off;

figure
semilogy(mu, abs(r3),  'b-', 'LineWidth', 2);hold on;
semilogy(mu0_orig + epsilon * sol.time, amp(4, :), 'r--', 'LineWidth', 2);
xlim([mu0_orig, MUTHR])
hold off;

