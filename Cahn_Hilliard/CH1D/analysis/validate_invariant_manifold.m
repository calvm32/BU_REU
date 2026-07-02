clear; clc;

k = 3;
L = 10;
epsilon = 0.05;
t0 = -10;

% closed-form solution
k_hat = k/L;
lambda = @(t) k_hat^2*( -k_hat^2 + epsilon*t );

const_term = @(t) (3*sqrt(pi)*k)/(sqrt(epsilon)*L) * exp(-k^6/(epsilon*L^6)) * ...
         exp( (2*k^4/(L^4))*t - (epsilon*k^2/(L^2)).*t.^2 );

img_term = @(t) ( erfi((sqrt(epsilon)*k/L)*t - k^3/(sqrt(epsilon)*L^3)) ...
                  - erfi((sqrt(epsilon)*k/L)*t0 - k^3/(sqrt(epsilon)*L^3)) );

IC_term = @(t) exp( (2*k^4/(L^4))*t - (epsilon*k^2/(L^2)).*t.^2 - (2*k^4/(L^4))*t0 + (epsilon*k^2/(L^2)).*t0.^2);

% IC
r_k_0 = 10; %(const_term(t0).*img_term(t0)/(1-IC_term(t0)))^(-1/2); % must be nonzero

r_k = @(t) (const_term(t).*img_term(t) + r_k_0^(-2)*IC_term(t)).^(-1/2);

% derivative
dt = 1e-6;
drdt = @(t) (r_k(t+dt)-r_k(t-dt))/(2*dt);

% Right-hand side of ODE
rhs = @(t) lambda(t).*r_k(t) - (k^2*3/L^2).*r_k(t).^3;

% Compare
t = linspace(t0,20,500);   % avoid t=0 where the formula is singular

err = drdt(t) - rhs(t);

fprintf('Maximum absolute error = %.3e\n', max(abs(err)));

figure;
% plot(t,rhs(t),'LineWidth',1.5)
% plot(t,drdt(t),'LineWidth',1.5, 'Color', '[0.1,0.1,0.1]')
plot(t,err,'LineWidth',1.5)
xlabel('t')
ylabel('dr/dt - RHS')
grid on
%title('Residual of ODE')

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------


syms t t0 k L eps r0 real
assumeAlso([k L eps r0] > 0)
assumeAlso(t0,'real')

A = k^4/L^4;
B = k^2*eps/L^2;
C = 3*k^2/L^2;

% Closed-form solution
const = (3*sqrt(sym(pi))*k)/(sqrt(eps)*L) ...
    * exp(-k^6/(eps*L^6)) ...
    * exp(2*A*t - B*t^2);

img = erfi((sqrt(eps)*k/L)*t - k^3/(sqrt(eps)*L^3)) ...
    - erfi((sqrt(eps)*k/L)*t0 - k^3/(sqrt(eps)*L^3));

IC = r0^(-2)*exp(2*A*t - B*t^2);

r = (const*img + IC)^(-1/2);

% Symbolic derivative
drdt = simplify(diff(r,t),'Steps',100);

% RHS of ODE
lambda = (k^2/L^2)*(-k^2/L^2 + eps*t);
rhs = lambda*r - C*r^3;

% Residual
residual = simplify(expand(drdt-rhs),'Steps',200);

disp(residual)