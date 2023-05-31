% This script performs model fitting to the 2-step (Nussenbaum et al., 2020) behavioral data using maximum likelihood estimation. 
% 
% Description:
% 1. Data Loading: The script loads the behavioral data from the 'data.mat' file.
% 2. Model Definition: Two models, static_model and dynamic_model, are defined along with their parameter priors. These models capture different aspects of the behavioral data.
% 3. Model Fitting: The script fits the defined models to each subject's data in parallel. It uses maximum likelihood estimation, employing the fmincon function and the GlobalSearch optimization algorithm.
% 4. Fit Measures: Fit measures such as AIC and BIC are calculated to assess the quality of model fit.
% 5. Model Comparison: The models are compared based on their AIC differences, providing insights into which model better explains the behavioral data.
% 6. Plotting: The script generates plots of AIC differences, allowing visual comparison between the models.
% 7. Results: The generated plots are saved as 'fit.png' and 'fit.svg' for further analysis and documentation.
% 
% Usage:
% 1. Ensure that the 'data.mat' file is available in the appropriate directory.
% 2. Run the script to perform model fitting and analysis.
% 3. Review the generated plots to compare the models based on AIC differences.
% 4. The script also saves the fit results for further analysis.
% 
% Dependencies:
% - Optimization Toolbox (fmincon)
% - Global Optimization Toolbox
% - Parallel Computing Toolbox (for parallel execution)
%
% Author: Jing-Jing Li (jl3676@berkeley.edu)
% Last Modified: 6/1/2023

%% Set up
% Clear workspace variables
clear all

% Load data
load("../data/Alldata.mat")

% Get list of subjects
subjects = 1:size(data,2);

%% Define priors of model parameters
% Define functions to sample model parameters from uniform distributions
alpha_sample = @(x) betarnd(1.1, 1.1);
beta_sample = @(x) gamrnd(3,1);
beta_mb_sample = @(x) gamrnd(3,1);
beta_mf_sample = @(x) gamrnd(3,1);
lambda_sample = @(x) betarnd(1.1, 1.1);
stick_sample = @(x) normrnd(0, 10);
lapse_sample = @(x) unifrnd(0, 1);
rec_sample = @(x) unifrnd(0, 1);
eps_sample = @(x) unifrnd(0, 1);

%% Define models
Ms = [];

% RL meta model
curr_model = [];
curr_model.name = 'static_model';
curr_model.pMin = [1e-6 1e-6 1e-6 1e-6 1e-6 -20 1e-6];
curr_model.pMax = [1 20 20 20 1 20 1];
curr_model.pdfs = {alpha_sample, beta_mb_sample, beta_mf_sample, beta_sample, lambda_sample, stick_sample, eps_sample};
curr_model.pnames = {'alpha','beta_mb','beta_mf','beta','lambda','stick','epsilon'};
Ms{1} = curr_model; % Add the model to the model list

% RL meta lapse model
curr_model = [];
curr_model.name = 'dynamic_model'; % Name of the model
curr_model.pMin = [1e-6 1e-6 1e-6 1e-6 1e-6 -20 1e-6 1e-6];
curr_model.pMax = [1 20 20 20 1 20 1 1];
curr_model.pdfs = {alpha_sample, beta_mb_sample, beta_mf_sample, beta_sample, lambda_sample, stick_sample, lapse_sample, rec_sample};
curr_model.pnames = {'alpha','beta_mb','beta_mf','beta','lambda','stick','lapse','recover'};
Ms{2} = curr_model; % Add the model to the model list

%% Fit models
All_Params = cell(length(Ms), 1);
All_fits = cell(length(Ms), 1);

for m = 1:length(Ms)
    fit_model = Ms{m};
    pmin = fit_model.pMin;
    pmax = fit_model.pMax;
    pdfs = fit_model.pdfs;

    fitmeasures = cell(length(subjects), 1);
    fitparams = cell(length(subjects), 1);

%     for k = 1:length(subjects) % no parallel processing
    parfor k = 1:length(subjects) % parallel processing
        s = subjects(k);
        this_data = data(k).data;

        % Sample parameter starting values
        par = zeros(length(pmin), 1);
        for p_ind = 1:length(pmin)
            par(p_ind) = pdfs{p_ind}(0); % 0 is the random seed
        end

        % Set starting values of dynamic model parameters to the best fit
        % static model parameters
        if contains(fit_model.name, 'dynamic')
            model_dynamic = fit_model;
            model_static_ind = find(contains(cellfun(@(x) x.name, Ms, 'UniformOutput', false), 'static'));
            model_static = Ms{model_static_ind};
            for z = 1:length(model_dynamic.pnames)
                this_p = model_dynamic.pnames{z};
                if sum(strcmp(this_p, model_static.pnames)) > 0
                    par(strcmp(this_p, model_dynamic.pnames)) = All_Params{model_static_ind}(k, strcmp(this_p, Ms{model_static_ind}.pnames));
                end
            end
            par(strcmp('lapse', model_dynamic.pnames)) = All_Params{model_static_ind}(k, strcmp('epsilon', model_static.pnames));
            par(strcmp('recover', model_dynamic.pnames)) = 1 - All_Params{model_static_ind}(k, strcmp('epsilon', model_static.pnames));
        end

        % Define the objective function for optimization
        myfitfun = @(p) feval([fit_model.name, '_llh'], p, this_data);
        rng default % For reproducibility
        fmincon_opts = optimoptions(@fmincon, 'Algorithm', 'sqp');
        problem = createOptimProblem('fmincon', 'objective', myfitfun, 'x0', par, 'lb', pmin, 'ub', pmax, 'options', fmincon_opts);
        gs = GlobalSearch;
        [param, llh] = run(gs, problem);
    
        % Calculate fit measures (AIC, BIC, etc.)
        ntrials = sum(cellfun(@(x) numel(x), struct2cell(this_data)));
        AIC = 2 * llh + 2 * length(param);
        BIC = 2 * llh + log(ntrials) * length(param);
        AIC0 = -2 * log(1/3) * ntrials;
        psr2 = (AIC0 - AIC) / AIC0;
    
        % Store fit measures and parameters for each subject
        fitmeasures{k} = [k llh AIC BIC psr2 AIC0];
        fitparams{k} = param';
    end

    % Store fit measures and parameters for each model
    All_Params{m} = cell2mat(fitparams);
    All_fits{m} = cell2mat(fitmeasures);
end

% Reformat All_fits matrix
temp = All_fits;
All_fits = zeros(length(subjects), size(temp{1}, 2), length(Ms));
for i = 1:length(Ms)
    All_fits(:, :, i) = temp{i};
end

%% Plot 
AICs = squeeze(All_fits(:, 3, :));
mAICs = AICs - repmat(mean(AICs, 2), 1, size(AICs, 2));

figure('Position', [300 300 900 400])
subplot(1, 2, 1)
hold on
bar(mean(mAICs))
errorbar(mean(mAICs), std(mAICs) / sqrt(size(mAICs, 1)))
xticks(1:length(Ms));
xticklabels(cellfun(@(x) x.name, Ms, 'UniformOutput', false));
set(gca, 'TickLabelInterpreter', 'none')
ylabel('\Delta AIC')
title('Models')

static_model_ind = find(contains(cellfun(@(x) x.name, Ms, 'UniformOutput', false), 'static'));
dynamic_model_ind = find(contains(cellfun(@(x) x.name, Ms, 'UniformOutput', false), 'dynamic'));
p = signrank(AICs(:, dynamic_model_ind), AICs(:, static_model_ind), 'tail', 'left');

subplot(1, 2, 2)
yline(0, '--')
hold on
plot(sort(AICs(:, dynamic_model_ind) - AICs(:, static_model_ind), 'descend'), '.', 'MarkerSize', 15)
title('attention vs. not')
ylabel('\Delta AIC')
xlabel('sorted participant')
set(gca, 'fontsize', 14)
sgtitle(['2-step (p=' num2str(p) ')'])

% Save figures (Fig 4)
saveas(gcf, '../plots/fit.png')
saveas(gcf, '../plots/fit.svg')

%% Save
save ModelFit
