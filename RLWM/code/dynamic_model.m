function data = dynamic_model(theta, K, realData)
% Simulates data using the dynamic reinforcement learning with working memory (RLWM) model.
% The function generates simulated trial data based on model parameters and the structure of provided real data.
%
% Inputs:
%   - theta: A vector of model parameters [alpha, bias, stick, rho, forget, lapse, recover].
%   - K (int): A hyper-parameter defining working memory capacity.
%   - realData: A matrix representing real experimental data, used to inform set sizes and block structure.
%
% Output:
%   - data: A matrix of simulated data. Each row represents a trial with columns detailing trial-specific information.
%
% Author: Jing-Jing Li (jl3676@berkeley.edu)
% Last Modified: 5/28/2023

% Fixed model parameter setting the inverse temperature for the softmax action selection
beta = 20;

% Unpack model parameters from the input vector theta
alpha = theta(1);  % Learning rate for the RL process
bias = theta(2);   % Modifies learning rate and WM update based on outcome
stick = theta(3);  % Stickiness bias
rho = theta(4);    % Working memory weight
forget = theta(5); % Forgetting rate for working memory
epsilon = 0;       % Uniform level of noise - not used in dynamic model
lapse = theta(6);  % Probability of transitioning from engaged to random
recover = theta(7);% Probability of transitioning from random to engaged

% Number of actions and initial engaged state
nA = 3;            % Number of available actions
engaged = 1;       % Initial state of engagement
actions = [1 2 3]; % Action space

% Initialize counter and extract unique blocks from real data
k = 0;
all_blocks = unique(realData(:,5))'; % Unique block identifiers from real data

% Iterate over each block of trials
for block = all_blocks
    % Extract data for the current block
    this_data = realData(realData(:,5) == block, :);
    ns = this_data(1,4); % Set size for the block
    num_trials = size(this_data, 1); % Number of trials in the block

    % Compute the working memory (WM) weight
    w = rho * min(1, K / ns);

    % Initialize Q-values and working memory matrix with equal probabilities
    side = zeros(1, nA);
    Q = (1 / nA) * ones(ns, nA); % Initial Q-values for RL
    WM = (1 / nA) * ones(ns, nA); % Initial values for working memory

    % Iterate over each trial within the block
    for trial = 1:num_trials
        k = k + 1; % Trial counter across all blocks
        
        % Extract state and correct action from this trial's data
        s = this_data(trial, 1);   % State
        cor = this_data(trial, 3); % Correct action

        data(k, 6) = engaged; % Record engagement state

        % Compute action probabilities based on current engagement state
        if engaged == 1
            % If engaged, compute probabilities from RL and WM perspectives
            pRL = exp(beta * (Q(s, :) + side * stick)); % RL-based probabilities
            pRL = epsilon / nA + (1 - epsilon) * pRL / sum(pRL); % Normalized
            pWM = exp(beta * (WM(s, :) + side * stick)); % WM-based probabilities
            pWM = epsilon / nA + (1 - epsilon) * pWM / sum(pWM); % Normalized
            pr = w * pWM + (1 - w) * pRL; % Combined probability using WM weight
            if rand < lapse, engaged = 0; end % Disengage with certain probability
        else
            % If disengaged, actions are chosen uniformly at random
            pr = ones(1, nA) / nA;
            if rand < recover, engaged = 1; end % Re-engage with certain probability
        end

        % Sample a choice based on computed probabilities
        choice = actions(find(rand < cumsum(pr), 1, 'first'));
        % For initial trials, force choice to match real data (learning phase)
        if trial <= ns
            choice = this_data(trial, 2);
        end
        correct = choice == cor; % Determine if the choice was correct
        r = correct; % Reward is 1 for correct choice, 0 otherwise
        
        % Update learning rates based on trial outcome
        if r == 1
            alphaRL = alpha;     % Standard learning rate for RL
            alphaWM = 1;         % Full update for WM on correct choice
        else
            alphaRL = bias * alpha; % Adjusted learning rate for RL
            alphaWM = bias;         % Adjusted update rate for WM
        end

        % Update WM and Q values with forgetting and learning
        WM = WM + forget * (1 / nA - WM); % Apply forgetting to all WM entries
        WM(s, choice) = WM(s, choice) + alphaWM * (r - WM(s, choice)); % Update WM for chosen action
        Q(s, choice) = Q(s, choice) + alphaRL * (r - Q(s, choice)); % Update Q-value for chosen action

        % Reset side and update for chosen action
        side = zeros(1, nA);
        side(choice) = 1;

        % Record trial data: state, choice, reward, set size, block, and chosen probability
        data(k, 1:5) = [s, choice, r, ns, block];
        data(k, 7) = pr(choice);
    end
    
    % Fill in trials beyond the actual number with NaNs to maintain structure
    for trials = num_trials + 1 : 13 * ns
        k = k + 1;
        data(k, :) = [nan, nan, nan, ns, block, nan, nan];
    end
end
end
