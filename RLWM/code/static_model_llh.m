function nllh = static_model_llh(theta,K,data)
% Computes the negative log-likelihood of the data given model parameters. 
%
% Inputs:
%   - theta: A vector of model parameters [alpha, bias, stick, rho, forget, lapse, recover].
%   - K (int): A hyper-parameter defining working memory capacity.
%   - data: A matrix representing real experimental data, used to inform set sizes and block structure.

% Outputs:
%   - nllh: Negative log-likelihood of the static model.
%
% Author: Jing-Jing Li (jl3676@berkeley.edu)
% Last Modified: 5/28/2023

% Fixed model parameter for softmax action selection, controls sharpness of action probabilities
beta = 20;

% Unpack model parameters from input vector theta
alpha = theta(1);  % Learning rate for the RL process
bias = theta(2);   % Modifies learning rate and WM update based on outcome
stick = theta(3);  % Stickiness bias
rho = theta(4);    % Working memory weight
forget = theta(5); % Forgetting rate for working memory
epsilon = theta(6);% Uniform level of noise
lapse = 0;         % Probability of transitioning from engaged to random - not used in static model
recover = 1;       % Probability of transitioning from random to engaged - not used in static model

T = [1-recover lapse;recover 1-lapse]; % Transition matrix

Allstimuli = data(:,1); % Stimuli in the observed data
Allchoices = data(:,2); % Choices in the observed data
Allrewards = data(:,3); % Rewards in the observed data
Allsetsize = data(:,4); % Set sizes in the observed data
Allblocks = data(:,5); % Blocks in the observed data

blocks = unique(Allblocks)'; % Unique block values

nA = 3; % Number of actions
llh = 0; % Log-likelihood
tt = 0; % Time step counter

for block = blocks
    Tb = find(Allblocks == block); % Indices of the current block in the observed data
    stimuli = Allstimuli(Tb); % Stimuli for the current block
    choices = Allchoices(Tb); % Choices for the current block
    rewards = Allrewards(Tb); % Rewards for the current block
    ns = Allsetsize(Tb(1)); % Set size for the current block (assuming it is constant within the block)

    % Working memory weight
    w = rho*min(1,K/ns);
    Q = (1/nA)*ones(ns,nA); % Q-values
    WM = (1/nA)*ones(ns,nA); % Working memory weights
    side = zeros(1,nA); % Side to stick to
    p = [lapse 1-lapse]; % Probability vector for latent states

    for k = 1:length(choices)
        tt = tt + 1;

        s = stimuli(k); % Current stimulus
        choice = choices(k); % Current choice
        r = rewards(k); % Current reward

        W = Q(s,:) + stick*side; % Action values with stickiness
        bRL = exp(beta*W); % Softmax probabilities for reinforcement learning
        bRL = epsilon/nA + (1-epsilon)*bRL/sum(bRL);
        W = WM(s,:) + stick*side; % Working memory values with stickiness
        bWM = exp(beta*W); % Softmax probabilities for working memory
        bWM = epsilon/nA + (1-epsilon)*bWM/sum(bWM);
        b = w*bWM + (1-w)*bRL; % Combined probabilities for choice

        lt = log(p(1)/nA + b(choice)*p(2)); % Log-transformed probability of the choice
        llh = llh + lt; % Accumulate log-likelihood

        p = (p(1) * T(:,1) / nA + b(choice) * p(2) * T(:,2)) / exp(lt); % Update probabilities using transition matrix

        latent(tt,:) = [p(2) b(choice)]; % Store latent probabilities and choice probabilities

        WM = WM + forget*(1/nA - WM); % Update working memory weights with forgetting
        if r == 1
            alphaRL = alpha;
            alphaWM = 1;
        else
            alphaRL = bias*alpha;
            alphaWM = bias;
        end
        Q(s,choice) = Q(s,choice) + alphaRL*(r - Q(s,choice)); % Update Q-value using reinforcement learning
        WM(s,choice) = WM(s,choice) + alphaWM*(r - WM(s,choice)); % Update working memory weight

        side = zeros(1,nA); % Reset side vector
        side(choice) = 1; % Set chosen action to 1 in side vector
    end
end

nllh = -llh; % Compute negative log-likelihood

end
