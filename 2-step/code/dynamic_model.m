function data = dynamic_model(theta, latent_st_traj)
% dynamic_model simulates data based on the dynamic model given the parameters (theta) and latent state trajectory (latent_st_traj).

% Parameters:
alpha = theta(1);           % softmax inverse temperature
beta_mb = theta(2);         % learning rate
beta_mf = theta(3);         % eligibility trace decay
beta = theta(4);            % mixing weight
lambda = theta(5);          % stimulus stickiness
stickiness = theta(6);      % response stickiness
lapse = theta(7);           % lapse
recover = theta(8);         % recover

N = 200;
if nargin > 1
    N = length(latent_st_traj);
end

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

engaged  = 1;
latent = zeros(N, 2);

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
    denominator1 = sum(exp(beta_mf * Qmf(1, :) + beta_mb * Qmb(1, :)) + (stickiness * priorchoice));

    latent(i, 2) = engaged ;

    if engaged  == 1
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
            engaged  = 0;
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
            engaged  = 1;
        end
    end

    if nargin > 1
        engaged  = rand < latent_st_traj(i);
    end

    numerator1 = exp(beta_mf * Qmf(1, a1) + beta_mb * Qmb(1, a1) + stickiness * priorchoice(1));
    numerator2 = exp(beta * (Qd(s, a2)));

    latent(i, 1) = (numerator1 / denominator1) * (numerator2 / denominator2);

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
data.latent = latent;
end
