%% Model Validation Script

% This script performs model validation for the Iowa Gambling Task (IGT).
% It loads precomputed model fits and behavioral data, generates simulated data for each
% fitted model, and compares the models with the actual human behavior. The
% validation is performed using bar plots that show the frequencies of deck
% choices for both human participants and the simulated models.

% The script consists of the following main steps:
%
% 1. Load necessary data:
%    - Load precomputed model fits from 'ModelFit.mat'.
%    - Load behavioral data for the IGT task.
%
% 2. Prepare data for behavior plotting:
%    - Calculate the frequencies of deck choices for each human participant.
%    - Generate a bar plot to visualize the human behavior.
%
% 3. Perform model validation:
%    - Iterate over the static and dynamic models.
%    - Generate simulated data for each subject using the fitted parameters.
%    - Calculate the frequencies of deck choices for the simulated data.
%    - Calculate the prediction accuracy by comparing the simulated data with the actual data.
%
% 4. Plot model validation results:
%    - Generate bar plots to compare the frequencies of deck choices between the models and human behavior.
%    - Plot the prediction accuracy of each model.
%    - Save the resulting plots as 'model_validation.png' and 'model_validation.svg'.
%
% Required Files:
%   - 'ModelFit.mat': Precomputed model fits.
%   - '../data/index_<ntrials>.csv': Index data for the IGT task.
%   - '../data/payoff_lookup.csv': Payoff lookup table for the IGT task.
%
% Output:
%   - '../plots/model_validation.png' and '../plots/model_validation.svg': Bar plots for model validation.
%
% Dependencies:
%   - Optimization Toolbox (fmincon)
%   - Global Optimization Toolbox
%   - Parallel Computing Toolbox (for parallel execution)
%
% Author: Jing-Jing Li (jl3676@berkeley.edu)
% Last Modified: 5/29/23

%% Load data
clear all

ntrials = 100;
load('ModelFit.mat') % Load precomputed model fits
indices = readtable(strjoin({'../data/index_', num2str(ntrials)}, "")); % Load index data
payoff_lookup = readtable("../data/payoff_lookup.csv"); % Load payoff lookup table

niters = 30; % Number of iterations for model validation
pred_acc = zeros(length(subjects),2,niters); % Array to store prediction accuracy

%% Plot behavior
choices = zeros(size(subjects,1),4); % Array to store deck choice frequencies
for k=1:length(subjects)
    this_data = table2array(data_choice(k,2:end));
    for c=1:4
        choices(k,c) = sum(this_data==c);
    end
    choices(k,:) = choices(k,:) / sum(choices(k,:));
end

figure('Position',[300,300,400,300])
b = bar(nanmean(choices,1), 'FaceColor', [.5 .4 .8], 'EdgeColor', [.8 .6 .9], 'LineWidth',1.5,'FaceAlpha',0.3);
hold on
er = errorbar(nanmean(choices,1),nanstd(choices,1)/sqrt(length(subjects)-1));
er.Color = [0 0 0];
er.LineStyle = 'none';

%% Plot static model
model = 'static_model';

choices = zeros(size(subjects,1),4,niters);
for k=1:length(subjects)
    theta = All_Params{1}(k,:);
    this_data = [table2array(data_choice(k,2:end))' table2array(data_gain(k,2:end))' table2array(data_loss(k,2:end))'];
    study = indices.Study(indices.Subj == sscanf(subjects{k},'Subj_%d'));
    payoff = payoff_lookup.Payoff(strcmp(payoff_lookup.Dataset,study{1}));

    parfor it=1:niters
        sim_data = feval(model, theta, payoff, ntrials);
        this_choices = zeros(4,1);
        for c=1:4
            this_choices(c) = sum(sim_data(:,1)==c);
        end
        choices(k,:,it) = this_choices;
        pred_acc(k,1,it) = mean(this_data(:,1)==sim_data(:,1));
    end
end

choices = sum(choices,3);
for k=1:length(subjects)
    choices(k,:) = choices(k,:) / sum(choices(k,:));
end

plot(nanmean(choices,1),'.','MarkerSize',20,'MarkerEdgeColor',	"#EDB120");
hold on

%% Plot dynamic model
model = 'dynamic_model';

choices = zeros(size(subjects,1),4,niters);
for k=1:length(subjects)
    theta = All_Params{2}(k,:);
    this_data = [table2array(data_choice(k,2:end))' table2array(data_gain(k,2:end))' table2array(data_loss(k,2:end))'];
    study = indices.Study(indices.Subj == sscanf(subjects{k},'Subj_%d'));
    payoff = payoff_lookup.Payoff(strcmp(payoff_lookup.Dataset,study{1}));

    parfor it=1:niters
        p_att = feval([model '_latent'], theta, this_data);
        sim_data = feval(model, theta, payoff, ntrials, p_att);
        this_choices = zeros(4,1);
        for c=1:4
            this_choices(c) = sum(sim_data(:,1)==c);
        end
        choices(k,:,it) = this_choices;
        pred_acc(k,2,it) = mean(this_data(:,1)==sim_data(:,1));
    end
end

choices = sum(choices,3);
for k=1:length(subjects)
    choices(k,:) = choices(k,:) / sum(choices(k,:));
end

plot(nanmean(choices,1),'.','MarkerSize',20,'MarkerEdgeColor',"#77AC30");

%% Format and save the plot
ylim([.1 .5])
xlabel('Deck')
ylabel('Frequency')
legend('Human','SEM','static','dynamic')
title('Model validation')
saveas(gcf, '../plots/model_validation.png');
saveas(gcf, '../plots/model_validation.svg');
