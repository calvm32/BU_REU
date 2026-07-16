function process(filename, datatype)
    % File to process big survey because of difference with Julia
    % implementation
    m = matfile(filename);
    
    EP = m.EP;
    MU0 = m.MU0;
    Lx = m.Lx;
    Nx = m.Nx;
    terminate_thr = m.terminate_thr;

    NUM_FILES = length(EP) * length(MU0);
    bar = waitbar(0, sprintf('Loading %d files...\n', NUM_FILES), 'Name', 'Process Monitor');
    
    % Flip MU0 and EP b/c of meshgrid
    for i = 1:length(MU0) 
        batch_l2     = m.L2_HIST(i, :);
        batch_amp    = m.AMP_HISTS(i, :);
        batch_dmode  = m.DMODE_HIST(i, :);
        
        for j = 1:length(EP) 
            mu0 = MU0(i);
            ep  = EP(j);
            dt = 0.02 * 1e-3 / ep; 
    
            L2_HIST    = batch_l2{j}(:); % Make sure its a column vector
            AMP_HISTS  = batch_amp{j};
            DMODE_HIST = batch_dmode{j};
    
            out_filename = sprintf('sample_ep=1e%.5f_mu0=%.7f_%s.mat', log10(ep), mu0,  datatype);
            save(out_filename, 'L2_HIST', 'DMODE_HIST', ...
                'AMP_HISTS', 'ep', 'mu0', 'Lx', 'Nx', 'dt', ...
                'terminate_thr', 'datatype')

            currFile = (i-1) * length(MU0) + j;
            currentProgress = currFile / NUM_FILES;
            statusText = sprintf('Processing file %d of %d...', currFile, NUM_FILES);
            waitbar(currentProgress, bar, statusText);
        end
    end
end

process('grid_sample.mat', 'Float64');
