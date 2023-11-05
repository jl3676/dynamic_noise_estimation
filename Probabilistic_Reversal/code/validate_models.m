% This script performs simulations, fits models to the simulated data,
% and plots the validation results based on the latent state.

clear all;

%% Set Up
% Define sampling functions for parameter generation
alpha_sample = @(x) betarnd(3, 10);
stick_sample = @(x) normrnd(0, .1);
eps_sample = @(x) betarnd(1, 15);
lapse_sample = @(x) betarnd(1, 15);
rec_sample = @(x) betarnd(1, 15);

% Store the sampling functions in a cell array
sampling_funcs = {alpha_sample, stick_sample, lapse_sample, rec_sample};

%% Simulate Participants
num_subjects = 1000;
Alldata = cell(num_subjects,1);
Alllatent = cell(num_subjects,1);
All_latent_st_traj = cell(num_subjects,1);
sim_params = zeros(num_subjects, 4);

% Iterate over participants to simulate data
for i = 1:num_subjects
    % Generate random parameters for each participant
    theta = zeros(length(sampling_funcs),1);
    for p = 1:length(theta)
        theta(p) = sampling_funcs{p}(0);
    end

    sim_params(i,:) = theta;

    % Simulate data for this participant
    this_data = dynamic_model(theta);
    this_latent = [this_data(:,1) this_data(:,end)];
    Alldata{i} = this_data;
    Alllatent{i} = this_latent;
end

%% Fit Models
options = optimoptions('fmincon','Display','off');

% Create a cell array to store the models
Ms = [];

% Define the static model and its parameter limits
curr_model = [];
curr_model.name = 'static_model';
curr_model.pMin = [1e-6, -1, 1e-6];
curr_model.pMax = [1, 1, 1];
curr_model.pnames = {'alpha', 'stick', 'epsilon'};

% Add the static model to the list of models
Ms{1} = curr_model;

% Define the dynamic model and its parameter limits
curr_model = [];
curr_model.name = 'dynamic_model';
curr_model.pMin = [1e-6, -1, 1e-6, 1e-6];
curr_model.pMax = [1, 1, 1, 1];
curr_model.pnames = {'alpha', 'stick', 'lapse', 'recover'};

% Add the dynamic model to the list of models
Ms{2} = curr_model;

All_Params = cell(length(Ms),1);
All_fits = cell(length(Ms),1);

% Fit models for each participant
for m = 1:length(Ms)
    fit_model = Ms{m};
    pmin = fit_model.pMin;
    pmax = fit_model.pMax;

    fitmeasures = cell(num_subjects,1);
    fitparams = cell(num_subjects,1);

    parfor s = 1:num_subjects
        data = Alldata{s};
        par = pmin + rand().*(pmax-pmin);

        myfitfun = @(p) feval([fit_model.name,'_llh'],p,data);
        rng default % For reproducibility
        opts = optimoptions(@fmincon,'Algorithm','sqp');
        problem = createOptimProblem('fmincon','objective',...
            myfitfun,'x0',par,'lb',pmin,'ub',pmax,'options',opts);
        gs = GlobalSearch;
        [param,llh] = run(gs,problem);

        ntrials = size(data,1);

        % Calculate fit measures
        AIC = 2*llh + 2*length(param);
        BIC = 2*llh + log(ntrials)*length(param);
        AIC0 = -2*log(1/3)*ntrials;
        psr2 = (AIC0-AIC)/AIC0;

        fitmeasures{s} = [s llh AIC BIC psr2 AIC0];
        fitparams{s} = param;
    end

    All_Params{m} = cell2mat(fitparams);
    All_fits{m} = cell2mat(fitmeasures);
end

temp = All_fits;
All_fits = zeros(num_subjects,size(temp{1},2),length(Ms));
for i = 1:length(Ms)
    All_fits(:,:,i) = temp{i};
end

%% Plot Validation Around Switches
niters = 10;

sim_data = zeros(num_subjects, 50, 2);
static_data = zeros(num_subjects, 50, 2);
dynamic_data = zeros(num_subjects, 50, 2);

sim_choice_prob = zeros(num_subjects, 500);
static_choice_prob = zeros(num_subjects, 500);
dynamic_choice_prob = zeros(num_subjects, 500);

latent_mask = zeros(num_subjects, 500);

% Iterate over participants to generate simulation data and calculate choice probabilities
for this_ID = 1:num_subjects
    this_static_sim_engaged = zeros(500,6);
    this_static_sim_random = zeros(500,6);
    this_dynamic_sim_engaged = zeros(500,6);
    this_dynamic_sim_random = zeros(500,6);
    this_data_all_engaged = zeros(500,6);
    this_data_all_random = zeros(500,6);

    this_data = Alldata{this_ID};
    this_latent = Alllatent{this_ID}(:,1);

    latent_mask(this_ID,:) = this_latent;
    sim_choice_prob(this_ID,:) = Alllatent{this_ID}(:,2);

    for it = 1:niters
        % Simulate data using static model
        data_temp = static_model(All_Params{1}(this_ID,:));
        latent = [data_temp(:,1) data_temp(:,end)];
        this_static_sim_engaged(this_latent==1,:) = this_static_sim_engaged(this_latent==1,:) + data_temp(this_latent==1,:);
        this_static_sim_random(this_latent==0,:) = this_static_sim_random(this_latent==0,:) + data_temp(this_latent==0,:);

        static_choice_prob(this_ID,:) = static_choice_prob(this_ID,:) + latent(:,2)';

        % Simulate data using dynamic model
        latent_st_traj = zeros(1,500);
        for i = 1:20
            latent = dynamic_model_latent(All_Params{2}(this_ID,:), this_data);
            latent_st_traj = latent_st_traj + latent(:,1);
        end
        latent_st_traj = latent_st_traj / 20;
        data_temp = dynamic_model(All_Params{2}(this_ID,:), latent_st_traj);
        latent = [data_temp(:,1) data_temp(:,end)];
        this_dynamic_sim_engaged(this_latent==1,:) = this_dynamic_sim_engaged(this_latent==1,:) + data_temp(this_latent==1,:);
        this_dynamic_sim_random(this_latent==0,:) = this_dynamic_sim_random(this_latent==0,:) + data_temp(this_latent==0,:);

        dynamic_choice_prob(this_ID,:) = dynamic_choice_prob(this_ID,:) + latent(:,2)';
    end

    this_data_all_engaged(this_latent==1,:) = this_data_all_engaged(this_latent==1,:) + this_data(this_latent==1,:);
    this_data_all_random(this_latent==0,:) = this_data_all_random(this_latent==0,:) + this_data(this_latent==0,:);

    this_data_all_engaged(this_latent==0,4) = nan;
    this_data_all_random(this_latent==1,4) = nan;
    this_static_sim_engaged(this_latent==0,4) = nan;
    this_static_sim_random(this_latent==1,4) = nan;
    this_dynamic_sim_engaged(this_latent==0,4) = nan;
    this_dynamic_sim_random(this_latent==1,4) = nan;

    rewards_engaged = reshape(this_data_all_engaged(:,4), [50,10]);
    rewards_engaged = nanmean(rewards_engaged, 2);
    rewards_random = reshape(this_data_all_random(:,4), [50,10]);
    rewards_random = nanmean(rewards_random, 2);

    rewards_static_engaged = reshape(this_static_sim_engaged(:,4), [50,10]);
    rewards_static_engaged = nanmean(rewards_static_engaged,2);
    rewards_static_random = reshape(this_static_sim_random(:,4), [50,10]);
    rewards_static_random = nanmean(rewards_static_random,2);

    rewards_dynamic_engaged = reshape(this_dynamic_sim_engaged(:,4), [50,10]);
    rewards_dynamic_engaged = nanmean(rewards_dynamic_engaged,2);
    rewards_dynamic_random = reshape(this_dynamic_sim_random(:,4), [50,10]);
    rewards_dynamic_random = nanmean(rewards_dynamic_random,2);

    sim_data(this_ID,:,1) = rewards_engaged;
    sim_data(this_ID,:,2) = rewards_random;
    static_data(this_ID,:,1) = rewards_static_engaged / niters;
    static_data(this_ID,:,2) = rewards_static_random / niters;
    dynamic_data(this_ID,:,1) = rewards_dynamic_engaged / niters;
    dynamic_data(this_ID,:,2) = rewards_dynamic_random / niters;
end

static_choice_prob = static_choice_prob / niters;
dynamic_choice_prob = dynamic_choice_prob / niters;

%% Plot
window = 1;
trials_to_plot = 15;

xs = -trials_to_plot:3:trials_to_plot;
figure('Position',[300,300,600,300])

data_to_plot = {static_data, dynamic_data};

for i = 1:2
    subplot(1,2,i);
    rewards_engaged = nanmean(sim_data(:,:,1),1)';
    rewards_random = nanmean(sim_data(:,:,2),1)';

    this_data = data_to_plot{i};

    rewards_static_engaged = nanmean(this_data(:,:,1),1)';
    rewards_static_random = nanmean(this_data(:,:,2),1)';

    rewards_engaged = movmean(rewards_engaged,window);
    rewards_random = movmean(rewards_random,window);
    rewards_static_engaged = movmean(rewards_static_engaged,window);
    rewards_static_random = movmean(rewards_static_random,window);

    rewards_engaged = [rewards_engaged(end-trials_to_plot:end); rewards_engaged(1:trials_to_plot)];
    rewards_random = [rewards_random(end-trials_to_plot:end); rewards_random(1:trials_to_plot)];
    rewards_static_engaged = [rewards_static_engaged(end-trials_to_plot:end); rewards_static_engaged(1:trials_to_plot)];
    rewards_static_random = [rewards_static_random(end-trials_to_plot:end); rewards_static_random(1:trials_to_plot)];

    plot([trials_to_plot+1 trials_to_plot+1], [0 1], '--', 'Color',[.5 .5 .5])
    hold on
    plot(rewards_engaged,'k','LineWidth',2)
    hold on
    plot(rewards_random,'k--','LineWidth',2)
    hold on
    plot(rewards_static_engaged,'r','LineWidth',2)
    hold on
    plot(rewards_static_random,'r--','LineWidth',2)

    xticks(1:3:2*trials_to_plot+1)
    xticklabels(xs)
    xlabel('Trial from switch')
    ylabel('Accuracy')

    legend({'', 'data engaged', 'data random', 'model engaged', 'model random'}, 'Interpreter','none','Location','best')
end

saveas(gcf,'../plots/validation_by_latent_st.png')
saveas(gcf,'../plots/validation_by_latent_st.svg')

%% Simulate Participants with static model
num_subjects = 1000;
Alldata = cell(num_subjects,1);
Alllatent = cell(num_subjects,1);
All_latent_st_traj = cell(num_subjects,1);
sim_params = zeros(num_subjects, 3);

% Store the sampling functions in a cell array
sampling_funcs = {alpha_sample, stick_sample, eps_sample};

% Iterate over participants to simulate data
for i = 1:num_subjects
    % Generate random parameters for each participant
    theta = zeros(length(sampling_funcs),1);
    for p = 1:length(theta)
        theta(p) = sampling_funcs{p}(0);
    end

    sim_params(i,:) = theta;

    % Simulate data for this participant
    this_data = static_model(theta);
    this_latent = [this_data(:,1) this_data(:,end)];
    Alldata{i} = this_data;
    Alllatent{i} = this_latent;
end

%% Fit Models
options = optimoptions('fmincon','Display','off');

% Create a cell array to store the models
Ms = [];

% Define the static model and its parameter limits
curr_model = [];
curr_model.name = 'static_model';
curr_model.pMin = [1e-6, -1, 1e-6];
curr_model.pMax = [1, 1, 1];
curr_model.pnames = {'alpha', 'stick', 'epsilon'};

% Add the static model to the list of models
Ms{1} = curr_model;

% Define the dynamic model and its parameter limits
curr_model = [];
curr_model.name = 'dynamic_model';
curr_model.pMin = [1e-6, -1, 1e-6, 1e-6];
curr_model.pMax = [1, 1, 1, 1];
curr_model.pnames = {'alpha', 'stick', 'lapse', 'recover'};

% Add the dynamic model to the list of models
Ms{2} = curr_model;

All_Params = cell(length(Ms),1);
All_fits = cell(length(Ms),1);

% Fit models for each participant
for m = 1:length(Ms)
    fit_model = Ms{m};
    pmin = fit_model.pMin;
    pmax = fit_model.pMax;

    fitmeasures = cell(num_subjects,1);
    fitparams = cell(num_subjects,1);

    parfor s = 1:num_subjects
        data = Alldata{s};
        par = pmin + rand().*(pmax-pmin);

        myfitfun = @(p) feval([fit_model.name,'_llh'],p,data);
        rng default % For reproducibility
        opts = optimoptions(@fmincon,'Algorithm','sqp');
        problem = createOptimProblem('fmincon','objective',...
            myfitfun,'x0',par,'lb',pmin,'ub',pmax,'options',opts);
        gs = GlobalSearch;
        [param,llh] = run(gs,problem);

        ntrials = size(data,1);

        % Calculate fit measures
        AIC = 2*llh + 2*length(param);
        BIC = 2*llh + log(ntrials)*length(param);
        AIC0 = -2*log(1/3)*ntrials;
        psr2 = (AIC0-AIC)/AIC0;

        fitmeasures{s} = [s llh AIC BIC psr2 AIC0];
        fitparams{s} = param;
    end

    All_Params{m} = cell2mat(fitparams);
    All_fits{m} = cell2mat(fitmeasures);
end

temp = All_fits;
All_fits = zeros(num_subjects,size(temp{1},2),length(Ms));
for i = 1:length(Ms)
    All_fits(:,:,i) = temp{i};
end

%% Plot Validation Around Switches
niters = 10;

sim_data = zeros(num_subjects, 50, 2);
static_data = zeros(num_subjects, 50, 2);
dynamic_data = zeros(num_subjects, 50, 2);

sim_choice_prob = zeros(num_subjects, 500);
static_choice_prob = zeros(num_subjects, 500);
dynamic_choice_prob = zeros(num_subjects, 500);

latent_mask = zeros(num_subjects, 500);

% Iterate over participants to generate simulation data and calculate choice probabilities
for this_ID = 1:num_subjects
    this_static_sim_engaged = zeros(500,6);
    this_static_sim_random = zeros(500,6);
    this_dynamic_sim_engaged = zeros(500,6);
    this_dynamic_sim_random = zeros(500,6);
    this_data_all_engaged = zeros(500,6);
    this_data_all_random = zeros(500,6);

    this_data = Alldata{this_ID};
    this_latent = Alllatent{this_ID}(:,1);

    latent_mask(this_ID,:) = this_latent;
    sim_choice_prob(this_ID,:) = Alllatent{this_ID}(:,2);

    for it = 1:niters
        % Simulate data using static model
        data_temp = static_model(All_Params{1}(this_ID,:));
        latent = [data_temp(:,1) data_temp(:,end)];
        this_static_sim_engaged(this_latent==1,:) = this_static_sim_engaged(this_latent==1,:) + data_temp(this_latent==1,:);
        this_static_sim_random(this_latent==0,:) = this_static_sim_random(this_latent==0,:) + data_temp(this_latent==0,:);

        static_choice_prob(this_ID,:) = static_choice_prob(this_ID,:) + latent(:,2)';

        % Simulate data using dynamic model
        latent = dynamic_model_latent(All_Params{2}(this_ID,:), this_data);
        latent_st_traj = latent(:,1);
        data_temp = dynamic_model(All_Params{2}(this_ID,:), latent_st_traj);
        latent = [data_temp(:,1) data_temp(:,end)];
        this_dynamic_sim_engaged(this_latent==1,:) = this_dynamic_sim_engaged(this_latent==1,:) + data_temp(this_latent==1,:);
        this_dynamic_sim_random(this_latent==0,:) = this_dynamic_sim_random(this_latent==0,:) + data_temp(this_latent==0,:);

        dynamic_choice_prob(this_ID,:) = dynamic_choice_prob(this_ID,:) + latent(:,2)';
    end

    this_data_all_engaged(this_latent==1,:) = this_data_all_engaged(this_latent==1,:) + this_data(this_latent==1,:);
    this_data_all_random(this_latent==0,:) = this_data_all_random(this_latent==0,:) + this_data(this_latent==0,:);

    this_data_all_engaged(this_latent==0,4) = nan;
    this_data_all_random(this_latent==1,4) = nan;
    this_static_sim_engaged(this_latent==0,4) = nan;
    this_static_sim_random(this_latent==1,4) = nan;
    this_dynamic_sim_engaged(this_latent==0,4) = nan;
    this_dynamic_sim_random(this_latent==1,4) = nan;

    rewards_engaged = reshape(this_data_all_engaged(:,4), [50,10]);
    rewards_engaged = nanmean(rewards_engaged, 2);
    rewards_random = reshape(this_data_all_random(:,4), [50,10]);
    rewards_random = nanmean(rewards_random, 2);

    rewards_static_engaged = reshape(this_static_sim_engaged(:,4), [50,10]);
    rewards_static_engaged = nanmean(rewards_static_engaged,2);
    rewards_static_random = reshape(this_static_sim_random(:,4), [50,10]);
    rewards_static_random = nanmean(rewards_static_random,2);

    rewards_dynamic_engaged = reshape(this_dynamic_sim_engaged(:,4), [50,10]);
    rewards_dynamic_engaged = nanmean(rewards_dynamic_engaged,2);
    rewards_dynamic_random = reshape(this_dynamic_sim_random(:,4), [50,10]);
    rewards_dynamic_random = nanmean(rewards_dynamic_random,2);

    sim_data(this_ID,:,1) = rewards_engaged;
    sim_data(this_ID,:,2) = rewards_random;
    static_data(this_ID,:,1) = rewards_static_engaged / niters;
    static_data(this_ID,:,2) = rewards_static_random / niters;
    dynamic_data(this_ID,:,1) = rewards_dynamic_engaged / niters;
    dynamic_data(this_ID,:,2) = rewards_dynamic_random / niters;
end

static_choice_prob = static_choice_prob / niters;
dynamic_choice_prob = dynamic_choice_prob / niters;

%% Plot
window = 1;
trials_to_plot = 15;

xs = -trials_to_plot:3:trials_to_plot;
figure('Position',[300,300,600,300])

data_to_plot = {static_data, dynamic_data};

for i = 1:2
    subplot(1,2,i);
    rewards_engaged = nanmean(sim_data(:,:,1),1)';
    rewards_random = nanmean(sim_data(:,:,2),1)';

    this_data = data_to_plot{i};

    rewards_static_engaged = nanmean(this_data(:,:,1),1)';
    rewards_static_random = nanmean(this_data(:,:,2),1)';

    rewards_engaged = movmean(rewards_engaged,window);
    rewards_random = movmean(rewards_random,window);
    rewards_static_engaged = movmean(rewards_static_engaged,window);
    rewards_static_random = movmean(rewards_static_random,window);

    rewards_engaged = [rewards_engaged(end-trials_to_plot:end); rewards_engaged(1:trials_to_plot)];
    rewards_random = [rewards_random(end-trials_to_plot:end); rewards_random(1:trials_to_plot)];
    rewards_static_engaged = [rewards_static_engaged(end-trials_to_plot:end); rewards_static_engaged(1:trials_to_plot)];
    rewards_static_random = [rewards_static_random(end-trials_to_plot:end); rewards_static_random(1:trials_to_plot)];

    plot([trials_to_plot+1 trials_to_plot+1], [0 1], '--', 'Color',[.5 .5 .5])
    hold on
    plot(rewards_engaged,'k','LineWidth',2)
    hold on
    plot(rewards_static_engaged,'r','LineWidth',2,'LineStyle',':')

    xticks(1:3:2*trials_to_plot+1)
    xticklabels(xs)
    xlabel('Trial from switch')
    ylabel('Accuracy')

    legend({'', 'data', 'model'}, 'Interpreter','none','Location','best')
end

saveas(gcf,'../plots/validation_static.png')
saveas(gcf,'../plots/validation_static.svg')


%% Plot AICs
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
ylabel('\Delta AIC')
xlabel('sorted participant')
set(gca, 'fontsize', 14)
sgtitle(['Probabilistic reversal - static (p=' num2str(p) ')'])

% Save figures (Fig 4)
saveas(gcf, '../plots/static_fit.png')
saveas(gcf, '../plots/static_fit.svg')