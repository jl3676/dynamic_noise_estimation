%% Recovers latent state probability trajectory for the dynamic model

% Clear workspace
clear all

% Load model fit data
load('ModelFit.mat')

% Specify the model
model = 'dynamic_model';
model_ind = find(strcmp(cellfun(@(x) x.name, Ms(:), 'UniformOutput', false), model));
M = Ms{model_ind};
fitted_params = All_Params{model_ind};

%% Iterate for all models

% Initialize progress bar
f = waitbar(0, 'Starting'); % Start progress bar for iterations
total_iters = length(subjects);
this_iter = 0;

% Initialize variables
p_engaged_0 = [];
p_engaged_1 = [];
p_engaged = cell(length(subjects),1);
latent_st = cell(length(subjects),1);
n_iters = length(subjects);

% Iterate over subjects
for it = 1:n_iters
    this_iter = this_iter + 1;
    waitbar(this_iter/total_iters, f, sprintf('Progress: %d %%', floor(this_iter/total_iters*100))); % Update progress bar
    
    % Retrieve fitted parameters for the subject
    theta = fitted_params(it,:);
    
    % Simulate data using the model and fitted parameters
    this_data = feval(M.name, theta);
    latent_sim = this_data.latent;
    ntrials = size(latent_sim,1);

    n_subj_its = 100;

    subj_p_engaged_0 = zeros(n_subj_its,ntrials);
    subj_p_engaged_1 = zeros(n_subj_its,ntrials);
    subj_p_engaged = [];
    subj_latent_st = [];

    all_data = cell(n_subj_its,1);
    all_latent_sim = cell(n_subj_its,1);
    all_latent_rec = zeros(n_subj_its,ntrials);

    % Parallelize simulation for different iterations
    parfor subj_it = 1:n_subj_its
        this_data = feval(M.name, theta);
        latent_sim = this_data.latent;
        all_data{subj_it} = this_data;
        all_latent_sim{subj_it} = latent_sim(:,2);
    end
        
    % Parallelize recovery process for different iterations and recoveries
    parfor this_it = 1:n_subj_its
        this_data = all_data{this_it};
        latent_sim = all_latent_sim{this_it};
        
        this_latent_rec = feval([M.name '_latent'], theta, this_data);
        all_latent_rec(this_it,:) = [nan(9,1); this_latent_rec(:,1)];
    end

    % Aggregate recovered attention trajectories and latent states for each iteration
    for subj_it = 1:n_subj_its
        latent_sim = all_latent_sim{subj_it};
        latent_rec = nanmean(all_latent_rec(subj_it,:));

        subj_latent_st = [subj_latent_st latent_sim'];
        subj_p_engaged = [subj_p_engaged latent_rec];
    end

    latent_st{it} = subj_latent_st;
    p_engaged{it} = subj_p_engaged;
end

% Close progress bar
close(f);

%% Plot true attention over recovered p(att)

% Set up plotting parameters
nbins = 10;
bins_left = linspace(0,1-1/nbins,nbins);
bins_right = bins_left + 1/nbins;

% Initialize variables for plotting
latent_st_data = [];
p_engaged_count = [];

% Iterate over subjects
for subj = 1:size(latent_st,1)
    this_p_engaged = p_engaged{subj};
    this_latent_st = latent_st{subj};

    this_latent_st_data = [];
    this_p_engaged_count = [];
    
    % Calculate mean latent state and frequency of recovered attention for each bin
    for bin = 1:nbins
        this_latent_st_data(end+1) = nanmean(this_latent_st(this_p_engaged>=bins_left(bin) & this_p_engaged<=bins_right(bin)));
        this_p_engaged_count(end+1) = nansum(this_p_engaged>=bins_left(bin) & this_p_engaged<=bins_right(bin));
    end

    latent_st_data = [latent_st_data; this_latent_st_data];
    p_engaged_count = [p_engaged_count; this_p_engaged_count];
end

% Plot true attention over recovered p(att)
x = 1:nbins;
data = nanmean(latent_st_data,1);
err = nanstd(latent_st_data,1) / sqrt(size(latent_st_data,1));
figure('Position',[300 300 1000 400])
subplot(1,2,1)
plot([0.5 10.5], [0 1], 'k');
hold on
er = errorbar(x,data,err,'ro');    
ticks = 1:(nbins+1);
ticks = ticks - 0.5;
xticks(ticks);
xticklabels(linspace(0,1,nbins+1))
xlabel('Recovered p(engaged)')
ylim([0,1])
ylabel('True latent state')
title(['Latent state by recovered p(engaged) (' M.name ')'],'Interpreter','none')

% Plot p(att) count
data = nanmean(p_engaged_count,1);
err = nanstd(p_engaged_count,1) / sqrt(size(p_engaged_count,1));
subplot(1,2,2)
bar(x,data,'FaceColor',[.5 .4 .8],'EdgeColor',[.8 .6 .9],'LineWidth',1.5)
hold on
er = errorbar(x,data,err,err);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
ticks = 1:(nbins+1);
ticks = ticks - 0.5;
xticks(ticks);
xticklabels(linspace(0,1,nbins+1))
ticks_y = round(yticks / sum(data) * 100);
ticks_y = strcat(string(ticks_y),' %');
yticklabels(ticks_y);
xlabel('Recovered p(engaged)')
ylabel('Frequency')
title(['Count of recovered p(engaged) (' M.name ')'],'Interpreter','none')

% Save plots
saveas(gcf,'../plots/latent_probs_rec.png')
saveas(gcf,'../plots/latent_probs_rec.svg')
