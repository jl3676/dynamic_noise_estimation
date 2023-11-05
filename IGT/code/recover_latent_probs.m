% This script performs an analysis of the true latent state compared to the recovered probability of engagement (p(engaged)) for a given dynamic model.
% It uses the modelFit.mat file which contains model fitting results and the behavioral data.
% The script iterates over subjects and generates simulated data based on the fitted parameters of the dynamic model.
% It then calculates the recovered p(engaged) using the dynamic model's latent state estimation.
% Finally, it plots the relationship between the true latent state and the recovered p(engaged) and also the count of recovered p(engaged) in different bins.

clear all
load('ModelFit.mat')
ntrials = 100;
indices = readtable(strjoin({'../data/index_', num2str(ntrials)}, ""));
payoff_lookup = readtable("../data/payoff_lookup.csv");

%% Data Initialization
model_ind = find(contains(cellfun(@(x) x.name, Ms, 'UniformOutput', false), 'dynamic')); % Find the index of the dynamic model
model = Ms{model_ind}; % Retrieve the dynamic model from the model list
fitted_params = All_Params{strcmp(cellfun(@(x) x.name, Ms(:), 'UniformOutput', false), model.name)}; % Get the fitted parameters for the dynamic model

%% Perform Iterations
num_subject_iterations = 30; % The number of repeats per subject

engaged_state = cell(length(subjects),1); % Cell array to store the true latent state for each subject
p_engaged = cell(length(subjects),1); % Cell array to store the recovered p(engaged) for each subject
num_iterations = size(subjects,1);

f = waitbar(0, 'Starting'); % Start progress bar for iterations
total_iterations = size(subjects,1);
current_iteration = 0;

for iteration=1:num_iterations
    current_iteration = current_iteration + 1;
    waitbar(current_iteration/total_iterations, f, sprintf('Progress: %d %%', floor(current_iteration/total_iterations*100))); % Update progress bar

    theta = fitted_params(iteration,:); % Get the fitted parameters for the current iteration
    study = indices.Study(indices.Subj == sscanf(subjects{iteration},'Subj_%d'));
    payoff = payoff_lookup.Payoff(strcmp(payoff_lookup.Dataset,study{1}));
    this_data = feval(model.name, theta, payoff, ntrials); % Generate simulated data based on the dynamic model and reward structure
    
    latent_sim = this_data(:,4:5); 
    ntrials = size(latent_sim,1);

    subj_engaged_state_0 = zeros(num_subject_iterations,ntrials); % Matrix to store the true latent state for each subject iteration
    subj_engaged_state_1 = zeros(num_subject_iterations,ntrials); % Matrix to store the true latent state for each subject iteration
    subj_engaged_state = []; % Vector to store the true latent state for each subject
    subj_p_engaged = []; % Vector to store the recovered p(engaged) for each subject

    all_data = cell(num_subject_iterations,1); % Cell array to store the simulated data for each subject iteration
    all_latent_sim = cell(num_subject_iterations,1); % Cell array to store the latent state for each subject iteration
    all_latent_rec = zeros(num_subject_iterations,ntrials); % Matrix to store the recovered latent state for each subject iteration and record

    parfor subj_it=1:num_subject_iterations
        this_data = feval(model.name, theta, payoff, ntrials); % Generate simulated data based on the dynamic model and reward structure
        all_data{subj_it} = this_data;
        all_latent_sim{subj_it} = this_data(:,4); 
    end

    parfor this_it=1:num_subject_iterations
        data = all_data{this_it}; % Get the simulated data for the current subject iteration
        latent_sim = all_latent_sim{this_it}; % Get the latent state for the current subject iteration
        
        this_latent_rec = feval([model.name '_latent'], theta, data); % Estimate the latent state using the dynamic model
        all_latent_rec(this_it,:) = this_latent_rec(:,1); % Store the recovered latent state
    end

    for subj_it=1:num_subject_iterations
        latent_sim = all_latent_sim{subj_it}; % Get the latent state for the current subject iteration
        latent_rec = nanmean(all_latent_rec(subj_it,:)); % Calculate the average recovered latent state

        subj_engaged_state = [subj_engaged_state latent_sim']; % Concatenate the true latent state for the current subject
        subj_p_engaged = [subj_p_engaged latent_rec]; % Concatenate the recovered p(engaged) for the current subject
    end

    engaged_state{iteration} = subj_engaged_state; % Store the true latent state for the current iteration
    p_engaged{iteration} = subj_p_engaged; % Store the recovered p(engaged) for the current iteration
end
close(f);

%% Plotting

% Plot the true latent state over the recovered p(engaged)
num_bins = 10;
bin_edges_left = linspace(0,1-1/num_bins,num_bins);
bin_edges_right = bin_edges_left + 1/num_bins;

engaged_state_data = [];
p_engaged_count = [];

for subj=1:size(engaged_state,1)
    this_p_engaged = p_engaged{subj}; % Get the recovered p(engaged) for the current subject
    this_engaged_state = engaged_state{subj}; % Get the true latent state for the current subject

    this_engaged_state_data = [];
    this_p_engaged_count = [];
    for bin=1:num_bins
        this_engaged_state_data(end+1) = nanmean(this_engaged_state(this_p_engaged>=bin_edges_left(bin) & this_p_engaged<=bin_edges_right(bin))); % Calculate the average true latent state in each bin
        this_p_engaged_count(end+1) = nansum(this_p_engaged>=bin_edges_left(bin) & this_p_engaged<=bin_edges_right(bin)); % Count the occurrences of recovered p(engaged) in each bin
    end

    engaged_state_data = [engaged_state_data; this_engaged_state_data]; % Concatenate the true latent state data for all subjects
    p_engaged_count = [p_engaged_count; this_p_engaged_count]; % Concatenate the count of recovered p(engaged) for all subjects
end

x = 1:num_bins;
engaged_state_mean = nanmean(engaged_state_data,1); % Calculate the mean true latent state for each bin
engaged_state_std = nanstd(engaged_state_data,1) / sqrt(size(engaged_state_data,1)); % Calculate the standard error of the mean true latent state for each bin
figure('Position',[300 300 1000 400])
subplot(1,2,1)
errorbar(x, engaged_state_mean, engaged_state_std, 'ro') % Plot the mean true latent state with error bars
hold on
plot([0.5 num_bins+0.5], [0 1], 'k'); % Plot a line representing the range of true latent state
ticks = 1:(num_bins+1);
ticks = ticks - 0.5;
xticks(ticks);
xticklabels(linspace(0,1,num_bins+1))
xlabel('Recovered p(engaged)')
ylim([0,1])
ylabel('True p(engaged)')
title('True vs. Recovered p(engaged)','Interpreter','none')

% Plot the count of recovered p(engaged) in different bins
p_engaged_count_mean = nanmean(p_engaged_count,1); % Calculate the mean count of recovered p(engaged) for each bin
p_engaged_count_std = nanstd(p_engaged_count,1) / sqrt(size(p_engaged_count,1)); % Calculate the standard error of the mean count of recovered p(engaged) for each bin
subplot(1,2,2)
bar(x, p_engaged_count_mean, 'FaceColor',[.5 .4 .8],'EdgeColor',[.8 .6 .9],'LineWidth',1.5) % Plot the mean count of recovered p(engaged) as a bar plot
hold on
errorbar(x, p_engaged_count_mean, p_engaged_count_std, p_engaged_count_std, 'k', 'LineStyle', 'none') % Plot the error bars
ticks = 1:(num_bins+1);
ticks = ticks - 0.5;
xticks(ticks);
xticklabels(linspace(0,1,num_bins+1))
ticks_y = round(yticks / sum(p_engaged_count_mean) * 100);
ticks_y = strcat(string(ticks_y),' %');
yticklabels(ticks_y);
xlabel('Recovered p(engaged)')
ylabel('Frequency')
title('Distribution of Recovered p(engaged)','Interpreter','none')

saveas(gcf,'../plots/latent_probs_rec.png')
saveas(gcf,'../plots/latent_probs_rec.svg')
