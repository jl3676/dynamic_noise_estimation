function data = RL_meta_static(theta, rewardStruct)
% Simulates data for the probabilistic reversal environment using a static RL model.
%
% Inputs:
%   - theta: Model parameters (1x8 vector)
%     - theta(1:2): Alpha parameters (alpha- and alpha+)
%     - theta(3): Beta parameter
%     - theta(4): Forget parameter
%     - theta(5): Bias parameter
%     - theta(6): Alpha_v parameter
%     - theta(7): Phi parameter
%     - theta(8): Epsilon parameter
%   - rewardStruct: Struct containing reward probabilities for each session
%
% Output:
%   - data: Simulated behavioral data
%
% Author: Jing-Jing Li (jl3676@berkeley.edu)
% Last Modified: 5/28/2023

%% Initialize
alpha = theta(1:2);
beta = theta(3);
forget = theta(4);
bias = theta(5);
alpha_v = theta(6);
phi = theta(7);
epsilon = theta(8);
lapse = 0;
recover = 1;

engaged = 1; % State variable indicating whether the agent is engaged or not
nA = 2; % Number of actions

%% Simulate data for each session
sessions = fieldnames(rewardStruct)';

for s = sessions
    Q = ones(1,nA) / nA; % randomly initialize Q-values for choices Left and Right
    err = 0; % Prediction error for reward prediction
    
    this_rewardProbL = rewardStruct.(s{1}).rewardProbL; % Reward probabilities for Left choices in the current session
    this_rewardProbR = rewardStruct.(s{1}).rewardProbR; % Reward probabilities for Right choices in the current session

    N = length(this_rewardProbR); % Number of trials in the session
    
    for t = 1:N
        data.(s{1})(t).rewardProbL = this_rewardProbL(t); % Store the reward probability for Left choice in the data
        data.(s{1})(t).rewardProbR = this_rewardProbR(t); % Store the reward probability for Right choice in the data
        data.(s{1})(t).latent_att = engaged; % Store the current state of engagement (latent) in the data

        % Calculate action probabilities
        b(2) = 1 / (1 + exp(beta * (Q(1) - Q(2) + bias))); % Probability of choosing Right based on Q-values and bias
        b(1) = 1 - b(2); % Probability of choosing Left
        b = epsilon / 2 + (1 - epsilon) * b; % Add static noise to the action probabilities
        
        if engaged == 1
            if rand < b(1) % Make choice using softmax for engaged state
                choice = 1; % Choose Left
            else
                choice = 2; % Choose Right
            end
            if rand < lapse, engaged = 0; end % Lapse from engaged to random state with a probability
        else
            choice = randsample([1, 2], 1); % Choose randomly when in random state
            if rand < recover, engaged = 1; end % Return to engaged state with a probability
        end

        data.(s{1})(t).latent_prob = b(choice); % Store the probability of the chosen action in the data

        % Calculate reward
        if choice == 1
            r = rand < this_rewardProbL(t) / 100; % Generate reward for Left choice based on the reward probability
            data.(s{1})(t).rewardL = r; % Store the reward outcome for Left choice in the data
            data.(s{1})(t).rewardR = nan; % No reward outcome for Right choice
        else
            r = rand < this_rewardProbR(t) / 100; % Generate reward for Right choice based on the reward probability
            data.(s{1})(t).rewardR = r; % Store the reward outcome for Right choice in the data
            data.(s{1})(t).rewardL = nan; % No reward outcome for Left choice
        end

        % Update Q-values and prediction error
        v = abs(r - Q(choice)) - err; % Calculate the value prediction error
        err = err + alpha_v * v; % Update the prediction error
        
        % Update alpha- (alpha for negative prediction errors)
        if r - Q(choice) < 0
            alpha(1) = phi * (v + theta(1)) + (1 - phi) * alpha(1); % Update alpha- using decay and phi
            alpha(1) = max(alpha(1), 0); % Ensure alpha- is not negative
            alpha(1) = min(alpha(1), 1); % Ensure alpha- is not greater than 1
        end

        % Update Q-values using the reward and prediction error
        Q(choice) = Q(choice) + alpha(r + 1) * (r - Q(choice)) * (1 - err);

        % Forget the value of the unchosen action
        Q(3 - choice) = forget * Q(3 - choice);
    end
end

end
