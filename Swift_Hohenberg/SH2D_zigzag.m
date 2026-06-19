
function [period_map] = sim()
    Lx = 2 * pi;
    Ly = 20 * pi;
    T = 500;
    Nt = 1000; % number of timesteps
    N = 256; % number of collocation points

    x = linspace(-Lx/2, Lx/2, N); 
    y = linspace(-Ly/2, Ly/2, N);
    [X, Y] = meshgrid(x, y);
    h = T / Nt;
    
    kx = fftshift((2 * pi / Lx) .* (- N / 2 : N / 2 - 1));
    ky = fftshift((2 * pi / Ly) .* (- N / 2 : N / 2 - 1));
    [Kx, Ky] = meshgrid(kx, ky);
    

    noise_std = 0.1;
    u_perturb = (rand(N, N) - 0.5) * noise_std;

    function [u, hat_u] = sh_solve(u0, k, mu)
        % Use pseudospectral method + Crank-Nicolson to solve
        laplacian_symbol = -k^2 * Kx.^2 - Ky.^2;
        linear_hat_diag = -(1 + laplacian_symbol).^2 + mu;
        inv_operator = 1 ./ (1 - 1.0 * h * linear_hat_diag);
        
        u = u0;
        hat_u = fft2(u0);
        
        for k = 1 : Nt
            % Compute the nonlinear contribution
            nonlinear = -u.^3;
            nonlinear_hat = fft2(nonlinear);

            hat_u = (hat_u + h * nonlinear_hat) .* inv_operator;
            % Crank-Nicolson
            %hat_u = (hat_u + h * nonlinear_hat + 0.5 * h * linear_hat_diag .* hat_u) .* inv_operator;   
            u = real(ifft2(hat_u)); % invert u_hat
        end
    end

    function [period] = zigzag_periodicity2(k, mu)
        kappa = k^2 - 1;
        up = sqrt(4 / 3 * (mu - kappa^2)) * cos(k * X);
        
        u0 = up + u_perturb;
        u = sh_solve(u0, k, mu);
        
        % Find the crest line
        crest_positions = zeros(N, 1);
        
        for row = 1:N
            [~, col_idx] = max(u(row, :));
            crest_positions(row) = x(col_idx);
        end
        
        normalized = crest_positions - mean(crest_positions);
        slice_fft = fft(normalized);
        
        [~, max_idx] = max(abs(slice_fft));
        max_freq = ky(max_idx);

        period = 2 * pi / abs(max_freq)

        % fig = figure;
        % imagesc(x, y, u);
        % hold on;
        % plot(crest_positions, y);
        % xlabel('X'); ylabel('Y');
        % title('Swift-Hohenberg Pattern Evolution');
        % hold off;
        % colorbar;
        % 
        % uiwait(fig);
    end

    function [period] = zigzag_periodicity(k, mu)
        kappa = k^2 - 1;
        up = sqrt(4 / 3 * (mu - kappa^2)) * cos(k * X);
        
        u0 = up + u_perturb;
        u = sh_solve(u0, k, mu);
        
        column_index = 3*N / 4 ; 
        slice = u(:, column_index);
        normalized = slice - mean(slice); 
        slice_fft = fft(normalized);
       
        [~, max_idx] = max(abs(slice_fft));
        max_freq = ky(max_idx);

        period = 2 * pi / abs(max_freq);
         
        % fig = figure;
        % imagesc(x, y, u);
        % xlabel('X'); ylabel('Y');
        % title('Swift-Hohenberg Pattern Evolution');
        % colorbar;
        % 
        % uiwait(fig);
    end

    % Plot period results over phase portrait
    num_mu_samples = 40;
    num_kappa_samples = 40;
    mu_samples = linspace(0, 1, num_mu_samples);
    kappa_samples = linspace(-1, 0, num_kappa_samples);
    period_map = zeros(num_mu_samples, num_kappa_samples);
    
    for i = 2:num_mu_samples
        mu = mu_samples(i)

        % Find lower and upper bound for acceptable k
        lower_bound = -sqrt(mu / 3); % Eckhaus condition
        upper_bound = -mu^2 / 512; % Zig_zag condition

        % Find indices associated with the bounds
        lower_index = find(kappa_samples > lower_bound, 1, 'first');
        upper_index = find(kappa_samples <= upper_bound, 1, 'last');

        % Handle empty case
        if isempty(lower_index), lower_index = NaN; end
        if isempty(upper_index), upper_index = NaN; end

        for j = lower_index:upper_index
            kappa = kappa_samples(j);
            k = sqrt(kappa + 1);
            %period_map(i, j) = i*j;
            period_map(i, j) = zigzag_periodicity2(k, mu);
        end
    end

    figure;
    imagesc(kappa_samples, mu_samples, period_map);
    set(gca, 'YDir', 'normal'); 
    colorbar; 
    colormap('parula');

    hold on; 
    
    % Now your lines will layer on top of the imagesc heatmap
    plot(kappa_samples, 3 * kappa_samples.^2, 'r--', 'LineWidth', 2); 
    plot(-mu_samples.^2 / 512, mu_samples, 'r--', 'LineWidth', 2);
    hold off;

    xlabel('\kappa'); ylabel('\mu');
    title('Zig-zag Periodicity with Stability Boundaries');
    legend('Simulation Heatmap', 'Eckhaus Boundary', 'Zig-Zag Boundary', 'Location', 'NorthWest');
end

period_map = sim();