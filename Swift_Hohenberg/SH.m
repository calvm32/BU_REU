L = 20 * pi; % width of periodic boundary
mu = 0.25;
T = 100;
Nt = 5000; % number of timesteps
N = 1024; % number of collocation points

x = linspace(-L / 2, L / 2, N); 
t = linspace(0, T, Nt);
h = T / Nt;

sigma = 1.0;
%u0 = sigma * randn(1, N);
%u0 = exp((-x.^2)/sigma);
u0 = sqrt(mu) * cos(1.19 * x);



k = fftshift((2 * pi / L) .* (- N / 2 : N / 2 - 1));

linear_hat_diag = -(1 - k.^2).^2 + mu;
inv_operator_denom = 1 - h * linear_hat_diag;

u = zeros(Nt, N);
hat_u = zeros(Nt, N);

u(1, :) = u0;
hat_u(1, :) = fft(u0);

% TODO, timestep t = T is not included!!
for i = 2 : Nt
    % Compute the nonlinear contribution
    nonlinear = -u(i-1, :).^3;
    nonlinear_hat = fft(nonlinear);

    hat_u(i, :) = (hat_u(i-1, :) + h * nonlinear_hat) ./ inv_operator_denom;

    u(i, :) = real(ifft(hat_u(i, :))); % invert u_hat
end

figure;
imagesc(x, linspace(0, T, Nt), u);
xlabel('Space (x)'); ylabel('Time (t)');
title('Swift-Hohenberg Pattern Evolution');
colorbar;