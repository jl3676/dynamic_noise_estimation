clear all

load("ModelFit.mat")

model1 = 'static_model';
model2 = 'dynamic_model';

model_ind1 = find(strcmp(cellfun(@(x) x.name, Ms(:), 'UniformOutput', false), model1));
model_ind2 = find(strcmp(cellfun(@(x) x.name, Ms(:), 'UniformOutput', false), model2));

%% plot behavior
figure('Position',[300 300 600 400])

stay_probs = zeros(4,length(data));
for s=1:length(data)
    this_data = data(s).data;

    money = this_data.money;
    money_prev = [-1; money(1:end-1)];

    transition = double(strcmp(this_data.transition,'common'));
    transition_prev = [-1; transition(1:end-1)];

    choice1 = this_data.choice1;
    choice1_prev = [0; choice1(1:end-1)];

    stay = choice1_prev == choice1;

    stay_probs(1,s) = sum(stay(transition_prev == 1 & money_prev == 1)) / sum(transition_prev == 1 & money_prev == 1);
    stay_probs(2,s) = sum(stay(transition_prev == 0 & money_prev == 1)) / sum(transition_prev == 0 & money_prev == 1);
    stay_probs(3,s) = sum(stay(transition_prev == 1 & money_prev == 0)) / sum(transition_prev == 1 & money_prev == 0);
    stay_probs(4,s) = sum(stay(transition_prev == 0 & money_prev == 0)) / sum(transition_prev == 0 & money_prev == 0);
end

stay_probs_mean = squeeze(nanmean(stay_probs, 2));
stay_probs_sem = squeeze(nanstd(stay_probs, 1, 2)) / sqrt(length(data));
y = [stay_probs_mean(1) stay_probs_mean(2); 
     stay_probs_mean(3) stay_probs_mean(4)];
err = [stay_probs_sem(1) stay_probs_sem(2);
       stay_probs_sem(3) stay_probs_sem(4)];

b = bar(y, 'grouped', 'LineWidth',1.5);
b(1).FaceColor = [0 .5 .5];
b(1).EdgeColor = [0 .9 .9];
b(2).FaceColor = [.5 .4 .8];
b(2).EdgeColor = [.8 .6 .9];
ylim([0.5 0.85])
hold on

[ngroups,nbars] = size(y);
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
end
xticklabels({'Reward', 'No reward'})
xlabel('Outcome of previoius trial')
ylabel('Proportion of first-stage stays')

errorbar(x', y, err, 'k', 'LineStyle','none','LineWidth',1)
hold on

%% plot simulations of model1
niters = 100;
stay_probs = zeros(4,length(data),niters);
fitted_params = All_Params{model_ind1};
for s=1:length(data)
    for it=1:niters
    theta = fitted_params(s,:);
    this_data = feval(model1, theta);

    money = this_data.money;
    money_prev = [-1; money(1:end-1)];

    transition = double(strcmp(this_data.transition,'common'));
    transition_prev = [-1; transition(1:end-1)];

    choice1 = this_data.choice1;
    choice1_prev = [0; choice1(1:end-1)];

    stay = choice1_prev == choice1;

    stay_probs(1,s,it) = sum(stay(transition_prev == 1 & money_prev == 1)) / sum(transition_prev == 1 & money_prev == 1);
    stay_probs(2,s,it) = sum(stay(transition_prev == 0 & money_prev == 1)) / sum(transition_prev == 0 & money_prev == 1);
    stay_probs(3,s,it) = sum(stay(transition_prev == 1 & money_prev == 0)) / sum(transition_prev == 1 & money_prev == 0);
    stay_probs(4,s,it) = sum(stay(transition_prev == 0 & money_prev == 0)) / sum(transition_prev == 0 & money_prev == 0);
    end
end

stay_probs = squeeze(nanmean(stay_probs, 3));

stay_probs_mean = squeeze(nanmean(stay_probs, 2));
stay_probs_sem = squeeze(nanstd(stay_probs, 1, 2)) / sqrt(length(data));
y = [stay_probs_mean(1) stay_probs_mean(2); 
     stay_probs_mean(3) stay_probs_mean(4)];
err = [stay_probs_sem(1) stay_probs_sem(2);
       stay_probs_sem(3) stay_probs_sem(4)];

[ngroups,nbars] = size(y);
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints - 0.05;
end

errorbar(x', y, err, 'k', 'LineStyle','none','LineWidth',1,'Marker','.','MarkerEdgeColor',[245/255 130/255 32/255],'MarkerSize',30)
hold on


%% plot simulations of model2
stay_probs = zeros(4,length(data),niters);
fitted_params = All_Params{model_ind2};
for s=1:length(data)
    for it=1:niters
    theta = fitted_params(s,:);
    latent_st_traj = feval(strjoin([model2,"_latent"],""), theta, data(s).data);
    latent_st_traj = latent_st_traj(:,1);
    this_data = feval(model2, theta, latent_st_traj);

    money = this_data.money;
    money_prev = [-1; money(1:end-1)];

    transition = double(strcmp(this_data.transition,'common'));
    transition_prev = [-1; transition(1:end-1)];

    choice1 = this_data.choice1;
    choice1_prev = [0; choice1(1:end-1)];

    stay = choice1_prev == choice1;

    stay_probs(1,s,it) = sum(stay(transition_prev == 1 & money_prev == 1)) / sum(transition_prev == 1 & money_prev == 1);
    stay_probs(2,s,it) = sum(stay(transition_prev == 0 & money_prev == 1)) / sum(transition_prev == 0 & money_prev == 1);
    stay_probs(3,s,it) = sum(stay(transition_prev == 1 & money_prev == 0)) / sum(transition_prev == 1 & money_prev == 0);
    stay_probs(4,s,it) = sum(stay(transition_prev == 0 & money_prev == 0)) / sum(transition_prev == 0 & money_prev == 0);
    end
end

stay_probs = squeeze(nanmean(stay_probs, 3));

stay_probs_mean = squeeze(nanmean(stay_probs, 2));
stay_probs_sem = squeeze(nanstd(stay_probs, 1, 2)) / sqrt(length(data));
y = [stay_probs_mean(1) stay_probs_mean(2); 
     stay_probs_mean(3) stay_probs_mean(4)];
err = [stay_probs_sem(1) stay_probs_sem(2);
       stay_probs_sem(3) stay_probs_sem(4)];

[ngroups,nbars] = size(y);
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints + 0.05;
end

errorbar(x', y, err, 'k', 'LineStyle','none','LineWidth',1,'Marker','.','MarkerEdgeColor',[112/255 172/255 66/255],'MarkerSize',30)
hold on

hleg = legend({'Common','Rare','','','static','','dynamic',''},'Location','best');
htitle = get(hleg,'Title');
set(htitle,'String','Previous trial transition')
title('2-step model validation')

saveas(gcf,'../plots/validation.svg')