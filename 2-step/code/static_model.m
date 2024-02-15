function data = static_model(theta)
% Simulates data for the 2-step environment using the static noise estimation model.
%
% Inputs:
%   - theta: Model parameters
%         theta(1): alpha - learning rate
%         theta(2): beta_mb - inverse softmax temperature parameter for
%         model-based learning
%         theta(3): beta_mf - inverse softmax temperature parameter for
%         model-free learning
%         theta(4): beta - inverse softmax temperature parameter for
%         the second stage
%         theta(5): lambda - discount factor between stages
%         theta(6): stickiness - choice stickiness
%         theta(7): epsilon - uniform level of noise
%   - latent_st_traj (optional): a vector containing the trial-by-trial 
%         probability trajectory of the latent engaged state 
%
% Output:
%   - data (struct): Simulated data
%
% Author: Jing-Jing Li (jl3676@berkeley.edu)
% Last Modified: 5/28/2023

% Parameters:
alpha = theta(1);           % learning rate
beta_mb = theta(2);         % softmax inverse temperature for model-based
beta_mf = theta(3);         % softmax inverse temperature for model-free
beta = theta(4);            % softmax inverse temperature for second stage
lambda = theta(5);          % discount factor
stickiness = theta(6);      % choice stickiness
epsilon = theta(7);         % uniform level of noise
lapse = 0;                  % lapse - not used in the static model
recover = 1;                % recover - not used in the static model

N = 200;

choice1 = zeros(N, 1);
choice2 = zeros(N, 1);
money = zeros(N, 1);
state = zeros(N, 1);
transition = cell(N, 1);

reward_table = readmatrix('../data/masterprob4.csv');

Qd = zeros(3, 2);           % Q values for each state-action pair
tr = 0.7;                   % common transition probability
Tm = [.7, .3; .3, .7];      % transition matrix for first-stage states
priorchoice = [0, 0];

engaged = 1;

for i = 1:N
    MaxQ = max(Qd, [], 2);
    MaxQS2 = MaxQ(2:end);
    
    Qmb = [sum(MaxQS2 .* Tm(:, 1)), sum(MaxQS2 .* Tm(:, 2))];
    Qmf = Qd(1, :);
    
    tr_prob = rand;
    if tr_prob < tr
        tr_type = 'common';
    else
        tr_type = 'rare';
    end
    
    numerator1 = exp(beta_mf * Qmf(1, 1) + beta_mb * Qmb(1, 1) + stickiness * priorchoice(1));
    denominator1 = sum(exp(beta_mf * Qmf(1, :) + beta_mb * Qmb(1, :) + stickiness * priorchoice));

    if engaged == 1 && rand > epsilon
        % Choose first stage action
        if rand < numerator1 / denominator1
            a1 = 1;
            s = round(double(tr_prob > tr)) + 2;
        else 
            a1 = 2;
            s = round(double(tr_prob < tr)) + 2;
        end

        numerator2 = exp(beta * (Qd(s, 1)));
        denominator2 = sum(exp(beta * (Qd(s, :))));

        if rand < numerator2 / denominator2
            a2 = 1;
        else
            a2 = 2;
        end

        if rand < lapse
            engaged = 0;
        end
    else
        a1 = randsample([1, 2], 1);
        a2 = randsample([1, 2], 1);

        if a1 == 1
            s = round(double(tr_prob > tr)) + 2;
        else
            s = round(double(tr_prob < tr)) + 2;
        end

        if rand < recover
            engaged = 1;
        end
    end

    numerator1 = exp(beta_mf*(Qmf(1,a1)) + beta_mb*(Qmb(1,a1)) + stickiness*priorchoice(1));
    numerator2 = exp(beta*(Qd(s,a2)));

    alien = rem(s, 2);

    if s <= 2
        reward = rand < reward_table(i + 1, 1 + alien);
    else
        reward = rand < reward_table(i + 1, 3 + alien);
    end
    reward = int8(reward);

    priorchoice = [0, 0];
    priorchoice(a1) = 1;
    
    tdQ = [0, 0];
    tdQ(1) = Qd(s, a2) - Qd(1, a1);
    tdQ(2) = reward - Qd(s, a2);
    
    Qd(1, a1) = Qd(1, a1) + alpha * tdQ(1) + lambda * alpha * tdQ(2);
    Qd(s, a2) = Qd(s, a2) + alpha * tdQ(2);

    choice1(i) = a1;
    choice2(i) = a2;
    money(i) = reward;
    state(i) = s;
    transition{i} = tr_type;
end

data = struct();
data.choice1 = choice1;
data.choice2 = choice2;
data.money = money;
data.state = state;
data.transition = transition;
end
