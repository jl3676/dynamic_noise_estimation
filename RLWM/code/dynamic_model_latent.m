function latent = dynamic_model_latent(theta,K,data)
% Computes the negative log-likelihood (nllh) of the static model for the RLWM task given the model parameters and the observed data.

% Inputs:
%   - theta: Model parameters for the static model.
%   - K: Working memory capacity (integer). 
%   - data: Observed data containing stimuli, choices, rewards, set sizes, and blocks.

% Outputs:
%   - latent: latent state and choice probability trajectories

alpha = theta(1);
beta = 20; % Fixed value for beta parameter
bias = theta(2);
stick = theta(3);
rho = theta(4);
forget = theta(5);
epsilon = 0;
lapse = theta(6);
recover = theta(7);

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
    p = [(1+lapse-recover)/2 (1-lapse+recover)/2]; % Probability vector for latent states

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
