%% Recovers Latent State Probability Trajectory

% Clear all variables in the workspace
clear all

%% Define Models

% Set options for optimization
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

%% Set Up Parameters

% Find the index of the dynamic model
model_ind = find(contains(cellfun(@(x) x.name, Ms, 'UniformOutput', false), 'dynamic'));
% Retrieve the dynamic model from the model list
model = Ms{model_ind};

names = Ms{model_ind}.pnames;
pmin = Ms{model_ind}.pMin;
pmax = Ms{model_ind}.pMax;

% Define sampling functions for model parameters
alpha_sample = @(x) betarnd(3, 10);
stick_sample = @(x) normrnd(0, .1);
lapse_sample = @(x) betarnd(1, 15);
rec_sample = @(x) betarnd(1, 15);

sampling_funcs = {alpha_sample, stick_sample, lapse_sample, rec_sample};

n_iters = 100;  % Number of iterations
ntrials = 500;  % Number of trials

%% Iterate

% Initialize variables for storing results
f = waitbar(0, 'Starting');  % Start progress bar for iterations
total_iters = n_iters;
this_iter = 0;
p_engaged = cell(n_iters, 1);
latent_st = cell(n_iters, 1);

n_subj_its = 100;  % Number of subject iterations
n_recs = 100;  % Number of recordings

for it = 1:n_iters
    this_iter = this_iter + 1;
    waitbar(this_iter/total_iters, f, sprintf('Progress: %d %%', floor(this_iter/total_iters*100)));  % Update progress bar
    
    % Generate random parameter values for the model
    theta = zeros(length(pmin), 1);
    for p = 1:length(pmin)
        theta(p) = sampling_funcs{p}(0);
    end

    subj_p_engaged = [];
    subj_latent_st = [];
    
    all_data = cell(n_subj_its, 1);
    all_latent_sim = cell(n_subj_its, 1);
    all_latent_rec = zeros(n_subj_its * n_recs, ntrials);

    % Generate data and latent states for each subject iteration
    parfor subj_it = 1:n_subj_its
        data = feval(Ms{model_ind}.name, theta);
        latent_sim = data(:, 1);
        all_data{subj_it} = data;
        all_latent_sim{subj_it} = latent_sim;
    end
        
    parfor this_it = 1:n_subj_its * n_recs
        subj_it = ceil(this_it / n_recs);
        rec = rem(this_it - 1, n_recs) + 1;
        this_data = all_data{subj_it};
        this_latent = feval([Ms{model_ind}.name '_latent'], theta, this_data);
        all_latent_rec(this_it, :) = this_latent(:, 1);
    end

    for subj_it = 1:n_subj_its
        latent_sim = all_latent_sim{subj_it};
        latent_rec = nanmean(all_latent_rec((subj_it - 1) * n_recs + 1:subj_it * n_recs, :));

        subj_latent_st = [subj_latent_st latent_sim'];
        subj_p_engaged = [subj_p_engaged latent_rec];
    end

    latent_st{it} = subj_latent_st;
    p_engaged{it} = subj_p_engaged;
end
close(f);

%% Plot True Attention Over Recovered p(att)

nbins = 10;
bins_left = linspace(0, 1 - 1/nbins, nbins);
bins_right = bins_left + 1/nbins;

latent_st_data = [];
p_engaged_count = [];

for subj = 1:size(latent_st, 1)
    this_p_engaged = p_engaged{subj};
    this_latent_st = latent_st{subj};

    this_latent_st_data = [];
    this_p_engaged_count = [];
    for bin = 1:nbins
        this_latent_st_data(end+1) = nanmean(this_latent_st(this_p_engaged >= bins_left(bin) & this_p_engaged <= bins_right(bin)));
        this_p_engaged_count(end+1) = nansum(this_p_engaged >= bins_left(bin) & this_p_engaged <= bins_right(bin));
    end

    latent_st_data = [latent_st_data; this_latent_st_data];
    p_engaged_count = [p_engaged_count; this_p_engaged_count];
end

x = 1:nbins;
data = nanmean(latent_st_data, 1);
err = nanstd(latent_st_data, 1) / sqrt(size(latent_st_data, 1));
figure('Position', [300 300 1000 400])
subplot(1, 2, 1)
% bar(x, data, 'FaceColor', [0 .5 .5], 'EdgeColor', [0 .9 .9], 'LineWidth', 1.5)
% hold on

plot([0.5 10.5], [0 1], 'k');
hold on
er = errorbar(x, data, err, 'ro');
% er.Color = [0 0 0];
% er.LineStyle = 'none';

ticks = 1:(nbins + 1);
ticks = ticks - 0.5;
xticks(ticks);
xticklabels(linspace(0, 1, nbins + 1))
xlabel('Recovered p(att)')
xlim([0.5, 10.5])
ylim([0, 1])
ylabel('True latent state')

title(['Attention state by recovered p(engaged) (' Ms{model_ind}.name ')'], 'Interpreter', 'none')

% plot p(att) count
data = nanmean(p_engaged_count, 1);
err = nanstd(p_engaged_count, 1) / sqrt(size(p_engaged_count, 1));
subplot(1, 2, 2)
bar(x, data, 'FaceColor', [.5 .4 .8], 'EdgeColor', [.8 .6 .9], 'LineWidth', 1.5)
hold on

er = errorbar(x, data, err, err);
er.Color = [0 0 0];
er.LineStyle = 'none';

ticks = 1:(nbins + 1);
ticks = ticks - 0.5;
xticks(ticks);
xticklabels(linspace(0, 1, nbins + 1))
ticks_y = round(yticks / sum(data) * 100);
ticks_y = strcat(string(ticks_y), ' %');
yticklabels(ticks_y);
xlabel('Recovered p(engaged)')
ylabel('Frequency')

title(['Count of recovered p(engaged) (' Ms{model_ind}.name ')'], 'Interpreter', 'none')

saveas(gcf, '../plots/latent_st_rec.png')
saveas(gcf, '../plots/latent_st_rec.svg')
