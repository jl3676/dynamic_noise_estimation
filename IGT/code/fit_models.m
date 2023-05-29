% IGT Model Fitting Script
%
% This script performs model fitting for the Iowa Gambling Task (IGT) data using two models: static_model and dynamic_model.
% The script loads the IGT data for a specified number of trials, defines the priors for the model parameters,
% and fits the models to the data using global optimization (GlobalSearch) and maximum likelihood estimation.
% The script then calculates fit measures (AIC, BIC, etc.) for each model and plots the model comparison results.
%
% Dependencies:
% - Optimization Toolbox (fmincon)
% - Global Optimization Toolbox
% - Parallel Computing Toolbox (for parallel execution)
%
% Author: Jing-Jing Li (jl3676@berkeley.edu)
% Last Modified: 5/29/2023

%% Set up
% Clear workspace variables
clear all

% Load data
ntrials = 100;
data_choice = readtable(strjoin(["../data/choice_" num2str(ntrials) ".txt"],""));
data_gain = readtable(strjoin(["../data/wi_" num2str(ntrials) ".txt"],""));
data_loss = readtable(strjoin(["../data/lo_" num2str(ntrials) ".txt"],""));

% Get list of subjects
subjects = unique(data_choice.Var1);

%% Define priors of model parameters
% Define functions to sample model parameters from uniform distributions
beta_sample = @(x) unifrnd(0, 20); % Sampling function for beta parameter
alpha_sample = @(x) unifrnd(0, 1); % Sampling function for alpha parameter
sensitivity_sample = @(x) unifrnd(0, 1); 
decay_sample = @(x) unifrnd(0, 1); % Sampling function for decay parameter
phi_sample = @(x) unifrnd(0, 200); % Sampling function for phi parameter
eps_sample = @(x) unifrnd(0, 1); % Sampling function for epsilon parameter
lapse_sample = @(x) unifrnd(0, 1); % Sampling function for lapse parameter (T_1^0)
rec_sample = @(x) unifrnd(0, 1); % Sampling function for recover parameter (T_0^1)

%% Define models
Ms = [];

curr_model = [];
curr_model.name = 'static_model';
curr_model.pMin = [1e-6 1e-6 1e-6 1e-6 1e-6 1e-6];
curr_model.pMax = [1 20 1 1 200 1];
curr_model.pdfs = {alpha_sample, beta_sample, sensitivity_sample, decay_sample, phi_sample, eps_sample};
curr_model.pnames = {'alpha','beta','sensitivity','decay','phi','epsilon'};

Ms{1} = curr_model;

curr_model = [];
curr_model.name = 'dynamic_model';
curr_model.pMin = [1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6];
curr_model.pMax = [1 20 1 1 200 1 1];
curr_model.pdfs = {alpha_sample, beta_sample, sensitivity_sample, decay_sample, phi_sample, lapse_sample, rec_sample};
curr_model.pnames = {'alpha','beta','sensitivity','decay','phi','lapse','recover'};

Ms{2} = curr_model;

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
        this_data = [table2array(data_choice(k,2:end))' table2array(data_gain(k,2:end))' table2array(data_loss(k,2:end))'];

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
sgtitle(['IGT (p=' num2str(p) ')'])

% Save figures (Fig 4)
saveas(gcf, '../plots/fit.png')
saveas(gcf, '../plots/fit.svg')

%% Save
save ModelFit
