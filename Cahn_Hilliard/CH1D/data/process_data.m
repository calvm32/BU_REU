function process(filename, datatype)
    m = matfile(filename);
    
    EP = m.EP;
    MU0 = m.MU0;
    Lx = m.Lx;
    Nx = m.Nx;
    terminate_thr = m.terminate_thr;
    
    for i = 1:length(EP)
        % --- BATCH LOAD: Pull the entire i-th row into RAM at once ---
        % This hits the disk 1 time per outer loop instead of 'num_mu0' times.
        batch_l2     = m.L2_HIST(i, :);
        batch_amp    = m.AMP_HISTS(i, :);
        batch_dmode  = m.DMODE_HIST(i, :);
        
        for j = 1:length(MU0)
            ep = EP(i);
            mu0 = MU0(j);
            dt = 0.02 * 1e-3 / ep; 
            
            L2_HIST    = batch_l2{j}(:); % Make sure its a column vector
            AMP_HISTS  = batch_amp{j};
            DMODE_HIST = batch_dmode{j};
                
            out_filename = sprintf('val_sample_ep=1e%.5f_mu0=%.7f_%s.mat', log10(ep), mu0,  datatype);
            save(out_filename, 'L2_HIST', 'DMODE_HIST', ...
                'AMP_HISTS', 'ep', 'mu0', 'Lx', 'Nx', 'dt', ...
                'terminate_thr', 'datatype')
        end
    end
end

% process('grid_survey_64x4_middle_strip.mat', 'Float64x4');
% process('grid_survey_64x4_left_strip.mat', 'Float64x4');
process('grid_survey_64_middle_strip.mat', 'Float64');
% process('grid_sample.mat', 'Float64');
