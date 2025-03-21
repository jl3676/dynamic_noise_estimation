function latent = dynamic_model_latent(theta, K, data)
% Computes the latent states and choice probability trajectories based on the dynamic RLWM model.
% This function calculates the latent states throughout the RLWM task, given a set of parameters 
% and observed data.
%
% Inputs:
%   - theta: A vector of model parameters [alpha, bias, stick, rho, forget, lapse, recover].
%   - K (int): A hyper-parameter defining working memory capacity.
%   - data: A matrix representing real experimental data, used to inform set sizes and block structure.
%
% Outputs:
%   - latent: A matrix containing the latent state probabilities and choice probabilities for each
%     trial.
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
epsilon = 0;       % Uniform level of noise - not used in dynamic model
lapse = theta(6);  % Probability of transitioning from engaged to random
recover = theta(7);% Probability of transitioning from random to engaged

% Transition matrix for engaged/random states
T = [1-recover lapse; recover 1-lapse];

% Extract trial information from the data matrix
Allstimuli = data(:,1); % Stimuli presented in each trial
Allchoices = data(:,2); % Choices made in each trial
Allrewards = data(:,3); % Rewards received in each trial
Allsetsize = data(:,4); % Set sizes for each trial
Allblocks = data(:,5);  % Block identifiers for each trial

% Process unique blocks for segmented analysis
blocks = unique(Allblocks)'; % Unique block identifiers

nA = 3; % Number of possible actions
tt = 0;  % Counter for total number of trials processed

% Iterate over each block to calculate latent states and choice probabilities
for block = blocks
    Tb = find(Allblocks == block); % Indices of trials belonging to the current block
    stimuli = Allstimuli(Tb); % Stimuli for the current block
    choices = Allchoices(Tb); % Choices for the current block
    rewards = Allrewards(Tb); % Rewards for the current block
    ns = Allsetsize(Tb(1)); % Assuming constant set size within the block

    % Compute working memory (WM) weight
    w = rho * min(1, K / ns);

    % Initialize Q-values and WM with equal probabilities
    Q = (1 / nA) * ones(ns, nA);
    WM = (1 / nA) * ones(ns, nA);
    side = zeros(1, nA); % Initialize side preference as none
    p = [lapse 1-lapse]; % Initial probabilities for latent states (random, engaged)

    % Process each trial in the current block
    for k = 1:length(choices)
        tt = tt + 1; % Increment total trial counter

        % Extract current trial information
        s = stimuli(k); % Current stimulus
        choice = choices(k); % Made choice
        r = rewards(k); % Received reward

        % Calculate action values with stickiness and compute softmax probabilities
        W = Q(s,:) + stick * side; % For reinforcement learning
        bRL = exp(beta * W); % Softmax probabilities for RL
        bRL = epsilon / nA + (1 - epsilon) * bRL / sum(bRL);
        W = WM(s,:) + stick * side; % For working memory
        bWM = exp(beta * W); % Softmax probabilities for WM
        bWM = epsilon / nA + (1 - epsilon) * bWM / sum(bWM);
        b = w * bWM + (1 - w) * bRL; % Combined choice probabilities

        % Calculate log-transformed probability of the chosen action
        lt = log(p(1) / nA + b(choice) * p(2)); % Based on latent state probabilities

        % Update latent state probabilities using the transition matrix
        p = (p(1) * T(:,1) / nA + b(choice) * p(2) * T(:,2)) / exp(lt);

        % Store latent state and choice probability for the current trial
        latent(tt,:) = [p(2) b(choice)];

        % Update WM and Q-values with forgetting and learning
        WM = WM + forget * (1 / nA - WM); % Apply forgetting to all WM entries
        if r == 1
            alphaRL = alpha;       % Standard learning rate for RL
            alphaWM = 1;           % Full update for WM on correct choice
        else
            alphaRL = bias * alpha; % Adjusted learning rate for RL
            alphaWM = bias;         % Adjusted update rate for WM
        end
        Q(s,choice) = Q(s,choice) + alphaRL * (r - Q(s,choice)); % Update Q-value
        WM(s,choice) = WM(s,choice) + alphaWM * (r - WM(s,choice)); % Update WM

        % Reset side and update for the chosen action
        side = zeros(1, nA);
        side(choice) = 1;
    end
end

end
