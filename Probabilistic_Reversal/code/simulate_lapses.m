clear all

%% Set up simulation parameters for individual participants with extended lapses
% Set the number of subjects to simulate
num_subjects = 20;
% Create a cell array to store simulated data for each subject
Alldata = cell(num_subjects, 1);
% Create a matrix to store simulated parameters for each subject
sim_params = zeros(num_subjects, 3);

for i = 1:num_subjects
    % alpha, stick, eps
    while true
        % Sample alpha and eps from normal distributions
        alpha = normrnd(0.3, 0.1);
        eps = normrnd(0.05, 0.02);
        % Sample the number of trials for the lapse period from a normal distribution
        n_trials_lapse = floor(normrnd(300, 100));
        % Check if the sampled parameters and lapse duration are within the desired range
        if alpha > 0 && alpha < 1 && eps > 0 && eps < 1 && ...
                n_trials_lapse > 0 && n_trials_lapse < 500
            break
        end
    end
    % Sample stick from a normal distribution
    stick = normrnd(0, 0.05);
    % Sample the start trial for the lapse period
    start_trial_lapse = ceil(rand * 300);

    % Store the start trial and lapse duration for each subject
    starts(i) = start_trial_lapse;
    durations(i) = n_trials_lapse;

    % Store the simulated parameters for each subject
    sim_params(i, :) = [alpha, stick, eps];

    % Simulate data for this subject using the static_model function
    Alldata{i} = static_model([alpha, stick, eps], start_trial_lapse, n_trials_lapse);
end

%% fit models
% Create a cell array to store the models
Ms = [];

% Define the static model and its parameter bounds
curr_model = [];
curr_model.name = 'static_model';
curr_model.pMin = [1e-6, -1, 1e-6];
curr_model.pMax = [1, 1, 1];
curr_model.pnames = {'alpha', 'stick', 'epsilon'};

% Add the static model to the list of models
Ms{1} = curr_model;

% Define the dynamic model and its parameter bounds
curr_model = [];
curr_model.name = 'dynamic_model';
curr_model.pMin = [1e-6, -1, 1e-6, 1e-6];
curr_model.pMax = [1, 1, 1, 1];
curr_model.pnames = {'alpha', 'stick', 'lapse', 'recover'};

% Add the dynamic model to the list of models
Ms{2} = curr_model;

% Create cell arrays to store the fitted parameters and fit measures for each model
All_Params = cell(length(Ms), 1);
All_fits = cell(length(Ms), 1);

for m = 1:length(Ms)
    % Select the current model
    fit_model = Ms{m};
    % Define the parameter limits for fitting
    pmin = fit_model.pMin;
    pmax = fit_model.pMax;

    % Create cell arrays to store the fit measures and fitted parameters for each subject
    fitmeasures = cell(num_subjects, 1);
    fitparams = cell(num_subjects, 1);

    % Perform fitting for each subject in parallel
    parfor s = 1:num_subjects
        % Get the data for the current subject
        data = Alldata{s};
        % Sample initial parameter values within the specified limits
        par = pmin + rand() .*(pmax - pmin);

        % Define the objective function for fitting
        myfitfun = @(p) feval([fit_model.name, '_llh'], p, data);
        % Set optimization options
        rng default % For reproducibility
        opts = optimoptions(@fmincon, 'Algorithm', 'sqp');
        % Create the optimization problem
        problem = createOptimProblem('fmincon', 'objective', ...
            myfitfun, 'x0', par, 'lb', pmin, 'ub', pmax, 'options', opts);
        % Create a global search object
        gs = GlobalSearch;
        % Run the global search to find the best fit
        [param, llh] = run(gs, problem);

        % Compute fit measures
        ntrials = size(data, 1);
        AIC = 2 * llh + 2 * length(param);
        BIC = 2 * llh + log(ntrials) * length(param);
        AIC0 = -2 * log(1 / 3) * ntrials;
        psr2 = (AIC0 - AIC) / AIC0;

        % Store the fit measures and fitted parameters for this subject
        fitmeasures{s} = [s, llh, AIC, BIC, psr2, AIC0];
        fitparams{s} = param;
    end

    % Convert cell arrays to matrices for fitted parameters and fit measures
    All_Params{m} = cell2mat(fitparams);
    All_fits{m} = cell2mat(fitmeasures);
end

% Reformat All_fits matrix
temp = All_fits;
All_fits = zeros(num_subjects, size(temp{1}, 2), length(Ms));
for i = 1:length(Ms)
    All_fits(:, :, i) = temp{i};
end

%% plot individual validation
% Set the number of subjects to plot
n_subj_to_plot = 6;
% Set the number of iterations for smoothing
niters = 50;
% Compute the AIC difference between dynamic and static models
AICs = squeeze(All_fits(:, 3, :));
AIC_diff = AICs(:, 2) - AICs(:, 1);
[~, idx] = sort(AIC_diff);
subjects = 1:num_subjects;
subjects_ranked = subjects(idx);
figure('Position', [200 200 1600 600])
colororder(gca().ColorOrder(4:end, :));
for subj_ind = 1:n_subj_to_plot
    subplot(2, floor(n_subj_to_plot / 2), subj_ind)
    this_ID = subjects_ranked(subj_ind); 

    % plot switches
    for s = 1:9
        xline(s * 50, '--');
        hold on
    end

    this_data_all = zeros(500, 6);
    this_static_sim = zeros(500, 6);
    this_dynamic_sim = zeros(500, 6);
    for it = 1:niters
        % Simulate data for this subject using the static model 
        this_data = static_model(sim_params(this_ID, :), starts(this_ID), durations(this_ID));
        this_data_all = this_data_all + this_data;
        this_static_sim = this_static_sim + static_model(All_Params{1}(this_ID, :));
        latent_st_traj = dynamic_model_latent(All_Params{2}(this_ID, :), this_data);
        this_dynamic_sim = this_dynamic_sim + dynamic_model(All_Params{2}(this_ID, :), latent_st_traj);
    end
    this_data_all = this_data_all / niters;
    this_static_sim = this_static_sim / niters;
    this_dynamic_sim = this_dynamic_sim / niters;

    % Plot the lapse period as a shaded area
    area([starts(this_ID), min(500, starts(this_ID) + durations(this_ID))], [1, 1], ...
        'basevalue', 0, 'EdgeColor', 'none', 'FaceColor', 'black', 'FaceAlpha', 0.05);
    hold on

    window = 50;
    smoothing_window = 10;
    % Apply moving mean to each block of 50 data points
    for i = 0:9
        start = i * window + 1;
        stop = start + window - 1;
        plot(start:stop, movmean(this_data_all(start:stop, 4), smoothing_window), 'k', 'LineWidth', 2)
        hold on
        plot(start:stop, movmean(this_static_sim(start:stop, 4), smoothing_window), 'r', 'LineWidth', 2)
        hold on
        plot(start:stop, movmean(this_dynamic_sim(start:stop, 4), smoothing_window), 'g', 'LineWidth', 2)
    end

    title(['ΔAIC=', sprintf('%.1f', AIC_diff(subjects == this_ID))]);
    hold on

    if subj_ind == 1
        legend({'', '', '', '', '', '', '', '', '', 'switch', 'lapse', 'data', 'static', 'dynamic'}, 'Interpreter', 'none')
    end
end

% Save the figure
saveas(gcf, '../plots/individual_learning_curves.png')
saveas(gcf, '../plots/individual_learning_curves.svg')

%% Simulate large number of participants
clear all

% Set the number of subjects to simulate
num_subjects = 3000;
% Create a cell array to store simulated data for each subject
Alldata = cell(num_subjects, 1);
% Create a matrix to store simulated parameters for each subject
sim_params = zeros(num_subjects, 3);

% Define functions for parameter sampling
alpha_sample = @(x) unifrnd(0, 0.6);
stick_sample = @(x) unifrnd(-0.3, 0.3);
eps_sample = @(x) unifrnd(0, 0.2);

for i = 1:num_subjects
    % alpha, stick, eps
    while true
        % Sample alpha and eps from uniform distributions
        alpha = alpha_sample(0);
        eps = eps_sample(0);
        % Sample the number of trials for the lapse period from a uniform distribution
        n_trials_lapse = unifrnd(0, 500);
        % Check if the sampled parameters and lapse duration are within the desired range
        if alpha > 0 && alpha < 1 && eps > 0 && eps < 1 && ...
                n_trials_lapse > 0 && n_trials_lapse < 500
            break
        end
    end
    % Sample stick from a uniform distribution
    stick = stick_sample(0);
    % Sample the start trial for the lapse period
    start_trial_lapse = ceil(rand * 100);

    % Store the start trial and lapse duration for each subject
    starts(i) = start_trial_lapse;
    durations(i) = min(n_trials_lapse, 500 - start_trial_lapse);

    % Store the simulated parameters for each subject
    sim_params(i, :) = [alpha, stick, eps];

    % Simulate data for this subject using the static_model function
    Alldata{i} = static_model([alpha, stick, eps], start_trial_lapse, n_trials_lapse);
end

%% fit models
% Create a cell array to store the models
Ms = [];

% Define the static model and its parameter bounds
curr_model = [];
curr_model.name = 'static_model';
curr_model.pMin = [1e-6, -1, 1e-6];
curr_model.pMax = [1, 1, 1];
curr_model.pnames = {'alpha', 'stick', 'epsilon'};

% Add the static model to the list of models
Ms{1} = curr_model;

% Define the dynamic model and its parameter bounds
curr_model = [];
curr_model.name = 'dynamic_model';
curr_model.pMin = [1e-6, -1, 1e-6, 1e-6];
curr_model.pMax = [1, 1, 1, 1];
curr_model.pnames = {'alpha', 'stick', 'lapse', 'recover'};

% Add the dynamic model to the list of models
Ms{2} = curr_model;

% Create cell arrays to store the fitted parameters and fit measures for each model
All_Params = cell(length(Ms), 1);
All_fits = cell(length(Ms), 1);

for m = 1:length(Ms)
    % Select the current model
    fit_model = Ms{m};
    % Define the parameter limits for fitting
    pmin = fit_model.pMin;
    pmax = fit_model.pMax;

    % Create cell arrays to store the fit measures and fitted parameters for each subject
    fitmeasures = cell(num_subjects, 1);
    fitparams = cell(num_subjects, 1);

    % Perform fitting for each subject in parallel
    parfor s = 1:num_subjects
        % Get the data for the current subject
        data = Alldata{s};
        % Sample initial parameter values within the specified limits
        par = pmin + rand() .*(pmax - pmin);

        % Set starting values of dynamic model parameters to the best fit
        % static model parameters
        if contains(fit_model.name, 'dynamic')
            model_dynamic = fit_model;
            model_static_ind = find(contains(cellfun(@(x) x.name, Ms, 'UniformOutput', false), 'static'));
            model_static = Ms{model_static_ind};
            for z = 1:length(model_dynamic.pnames)
                this_p = model_dynamic.pnames{z};
                if sum(strcmp(this_p, model_static.pnames)) > 0
                    par(strcmp(this_p, model_dynamic.pnames)) = All_Params{model_static_ind}(s, strcmp(this_p, Ms{model_static_ind}.pnames));
                end
            end
            par(strcmp('lapse', model_dynamic.pnames)) = All_Params{model_static_ind}(s, strcmp('epsilon', model_static.pnames));
            par(strcmp('recover', model_dynamic.pnames)) = 1 - All_Params{model_static_ind}(s, strcmp('epsilon', model_static.pnames));
        end

        % Define the objective function for fitting
        myfitfun = @(p) feval([fit_model.name, '_llh'], p, data);
        % Set optimization options
        rng default % For reproducibility
        opts = optimoptions(@fmincon, 'Algorithm', 'sqp');
        % Create the optimization problem
        problem = createOptimProblem('fmincon', 'objective', ...
            myfitfun, 'x0', par, 'lb', pmin, 'ub', pmax, 'options', opts);
        % Create a global search object
        gs = GlobalSearch;
        % Run the global search to find the best fit
        [param, llh] = run(gs, problem);

        % Compute fit measures
        ntrials = size(data, 1);
        AIC = 2 * llh + 2 * length(param);
        BIC = 2 * llh + log(ntrials) * length(param);
        AIC0 = -2 * log(1 / 3) * ntrials;
        psr2 = (AIC0 - AIC) / AIC0;

        % Store the fit measures and fitted parameters for this subject
        fitmeasures{s} = [s, llh, AIC, BIC, psr2, AIC0];
        fitparams{s} = param;
    end

    % Convert cell arrays to matrices for fitted parameters and fit measures
    All_Params{m} = cell2mat(fitparams);
    All_fits{m} = cell2mat(fitmeasures);
end

temp = All_fits;
All_fits = zeros(num_subjects, size(temp{1}, 2), length(Ms));
for i = 1:length(Ms)
    All_fits(:, :, i) = temp{i};
end

%% plot validation
% Set the number of repeats for validation
niters = 50;

% Create a matrix to store the mean squared error (MSE) for each subject and model
MSE = zeros(num_subjects, 2);

parfor this_ID = 1:num_subjects
    % Create variables to accumulate simulated data for each iteration
    this_data_all = zeros(500, 6);
    this_static_sim = zeros(500, 6);
    this_dynamic_sim = zeros(500, 6);
    for it = 1:niters
        % Simulate data for this subject using the static model
        this_data = static_model(sim_params(this_ID, :), starts(this_ID), durations(this_ID));
        this_data_all = this_data_all + this_data;
        this_static_sim = this_static_sim + static_model(All_Params{1}(this_ID, :));
        latent_st_traj = dynamic_model_latent(All_Params{2}(this_ID, :), this_data);
        this_dynamic_sim = this_dynamic_sim + dynamic_model(All_Params{2}(this_ID, :), latent_st_traj);
    end
    % Compute the average of simulated data over iterations
    this_data_all = this_data_all / niters;
    this_static_sim = this_static_sim / niters;
    this_dynamic_sim = this_dynamic_sim / niters;

    window = 20;
    % Apply moving mean to each block of 20 data points
    this_data_all = movmean(this_data_all(:, 4), window);
    this_static_sim = movmean(this_static_sim(:, 4), window);
    this_dynamic_sim = movmean(this_dynamic_sim(:, 4), window);

    % Compute the MSE for static and dynamic models
    MSE(this_ID, :) = [sum((this_data_all - this_static_sim).^2) / length(this_data_all), ...
        sum((this_data_all - this_dynamic_sim).^2) / length(this_data_all)];
end

%% plot MSE against length of lapse
% Plot the MSE for static and dynamic models against the length of the lapse period
plot(durations, MSE(:, 1), '.', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r')
hold on
p = polyfit(durations, MSE(:, 1), 2);
x1 = linspace(0, 500);
y1 = polyval(p, x1);
plot(x1, y1, 'k', 'LineWidth', 2)
hold on
plot(durations, MSE(:, 2), '.', 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'g')
hold on
p = polyfit(durations, MSE(:, 2), 2);
y1 = polyval(p, x1);
plot(x1, y1, 'k', 'LineWidth', 2)
ylim([0, 0.05])

xlabel('Duration of lapse (Trials)')
ylabel('MSE per trial')

legend({'static', '', 'dynamic', ''}, 'Interpreter', 'none')
title('Difference in learning curves with sim data')

% Save the figure
saveas(gcf, '../plots/MSE.svg')
saveas(gcf, '../plots/MSE.png')

%% plot |inferred - true| over true binned
figure('Position', [300 300 1000 400])
for p = 1:2
    subplot(1, 2, p)

    x = sim_params(:, p);
    y = [All_Params{1}(:, p), All_Params{2}(:, p)];
    y = abs(y - sim_params(:, p)); 

    % Define the number of bins for each column of y
    numBins = 3;
    
    % Initialize a matrix to store the bin counts and bin means for each column of y
    binCounts = zeros(numBins, size(y, 2));
    binMeans = zeros(numBins, size(y, 2));
    
    % Loop over each column of y
    for i = 1:size(y, 2)
        % Compute the bin edges based on the corresponding values in x
        binEdges = linspace(min(x), max(x), numBins+1);
        
        % Compute the bin counts and means for this column of y
        [binCounts(:, i), ~, binIdx] = histcounts(x, binEdges);
        binMeans(:, i) = accumarray(binIdx, y(:, i), [], @mean);
    end
    
    % Compute the standard error of the mean for each bin
    sem = std(y) ./ sqrt(binCounts);
    
    % Define the custom colors for the bars
    colors = [0.8588 0.5176 0.5765; 0.5098 0.6784 0.7922];
    
    % Define the offset for the error bars
    offset = (binEdges(2) - binEdges(1)) / 2;

    % Create the bar plot
    barPlot = bar(binMeans, 'grouped');
    
    % Set the custom colors for the bars
    for i = 1:size(y, 2)
        set(gca, 'ColorOrderIndex', i);
        set(barPlot(i), 'FaceColor', colors(i, :));
    end
    
    % Add error bars
    hold on;
    for i = 1:size(y, 2)
        this_x = 1:numBins;
        this_x = this_x + (i-1.5) * 0.28;
        errorbar(this_x, binMeans(:, i), sem(:, i), '.', 'Color', 'k', 'LineWidth', 1.5);
    end
    hold off;
    
    % Label the x-axis with the bin edges
    xticklabels(sprintfc('%0.2f to %0.2f', ...
        [binEdges(1:end-1); binEdges(2:end)]'));

    xlabel('True')
    ylabel('|Inferred - True|')

    title(Ms{1}.pnames{p})
    
    % Add a legend for the two columns of y
    legend({'static', 'dynamic'}, 'Interpreter', 'none', 'Location', 'northwest');
end

% Save the figure
saveas(gcf, '../plots/param_binned_over_true.svg')
saveas(gcf, '../plots/param_binned_over_true.png')

%% plot |inferred - true| over number of trials binned
figure('Position', [300 300 1000 400])
for p = 1:2
    subplot(1, 2, p)

    x = durations';
    y = [All_Params{1}(:, p), All_Params{2}(:, p)];
    y = abs(y - sim_params(:, p)); 

    % Compute the bin edges based on the corresponding values in x
    binEdges = [0 100 200 300 400 500];

    % Define the number of bins for each column of y
    numBins = length(binEdges) - 1;
    
    % Initialize a matrix to store the bin counts and bin means for each column of y
    binCounts = zeros(numBins, size(y, 2));
    binMeans = zeros(numBins, size(y, 2));
    
    % Loop over each column of y
    for i = 1:size(y, 2)
        % Compute the bin counts and means for this column of y
        [binCounts(:, i), ~, binIdx] = histcounts(x, binEdges);
        binMeans(:, i) = accumarray(binIdx, y(:, i), [], @mean);
    end
    
    % Compute the standard error of the mean for each bin
    sem = std(y) ./ sqrt(binCounts);
    
    % Define the custom colors for the bars
    colors = [0.8588 0.5176 0.5765; 0.5098 0.6784 0.7922];
    
    % Define the offset for the error bars
    offset = (binEdges(2) - binEdges(1)) / 2;

    % Create the bar plot
    barPlot = bar(binMeans, 'grouped');
    
    % Set the custom colors for the bars
    for i = 1:size(y, 2)
        set(gca, 'ColorOrderIndex', i);
        set(barPlot(i), 'FaceColor', colors(i, :));
    end
    
    % Add error bars
    hold on;
    for i = 1:size(y, 2)
        this_x = 1:numBins;
        this_x = this_x + (i-1.5) * 0.28;
        errorbar(this_x, binMeans(:, i), sem(:, i), '.', 'Color', 'k', 'LineWidth', 1.5);
    end
    hold off;
    
    % Label the x-axis with the bin edges
    xticklabels(sprintfc('%d to %d', ...
        [binEdges(1:end-1); binEdges(2:end)]'));

    xlabel('Lapse duration (Trials)')
    ylabel('|Inferred - True|')

    title(Ms{1}.pnames{p})
    
    % Add a legend for the two columns of y
    legend({'static', 'dynamic'}, 'Interpreter', 'none', 'Location', 'northwest');
end

% Save the figure
saveas(gcf, '../plots/param_binned_over_lapse_duration.svg')
saveas(gcf, '../plots/param_binned_over_lapse_duration.png')
