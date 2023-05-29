function data = dynamic_model(theta, latent_st_traj)
% Simulates data for the probabilistic reversal environment.
%
% Inputs:
%   - theta: Model parameters
%         theta(1): alpha - learning rate
%         theta(2): stickiness - stickiness parameter
%         theta(3): lapse - transition probability from engaged to random
%         theta(4): recover - transition probability from random to engaged
%   - latent_st_traj: a vector containing the probability trajectory of the
%   latent engaged state trial by trial
%
% Output:
%   - data: Simulated data matrix
%         Column 1: Engaged state (1 = engaged state, 0 = random state)
%         Column 2: Correct choice (1 or 2)
%         Column 3: Participant's choice
%         Column 4: Correctness of the choice (1 = correct, 0 = incorrect)
%         Column 5: Reward received (1 = rewarded, 0 = not rewarded)
%         Column 6: Probability of choosing the performed action
%
% Author: Jing-Jing Li (jl3676@berkeley.edu)
% Last Modified: 5/28/2023

%% Initialize
beta = 8;
alpha = theta(1);
stick = theta(2);
epsilon = 0;
lapse = theta(3);
recover = theta(4);

nA = 2;            % Number of actions
noise = 0.8;       % Probability that correct choice is rewarded
num_episodes = 10; % Number of episodes
num_trials = 50;   % Number of trials per episode

engaged = 1;          % Initial attentional state (1 = attentive state)
Q = ones(1, nA) / nA; % Initialize action values
data = zeros(num_episodes * num_trials, 6); % Data matrix to store simulated data

k = 0; % Counter for data matrix

%% Simulate
for ep = 1:num_episodes % Episode
    cor = 1 + rem(ep, 2); % Correct choice (alternates between 1 and 2)
    side = 0; % Side of stickiness; 1 = stick to A1, -1 = stick to A2
    for t = 1:num_trials
        k = k + 1;
        data(k, 1) = engaged; % Store current attentional state in data matrix
        
        if engaged == 1
            pr = 1 / (1 + exp(beta * (Q(1) - Q(2) + side * stick))); % Probability of choosing action A1
            if rand < lapse
                engaged = 0; % Lapse with a probability
            end
        else
            pr = 0.5; % Probability of choosing action A1 during lapse state
            if rand < recover
                engaged = 1; % Return to engaged state with a probability
            end
        end

        if nargin > 1 
            engaged = rand < latent_st_traj(k);
        end

        choice = 1 + (rand < pr); % Choose action A1 with probability pr
        if rand < epsilon
            choice = randsample([1 2], 1); % Exploration: Choose randomly between actions A1 and A2
        end
        correct = choice == cor; % Check correctness of choice
        if rand < noise
            r = correct; % Reward correct choice with probability equal to noise
        else
            r = 1 - correct; % Reward incorrect choice
        end

        choice_prob = (3 - 2 * choice) * pr + choice - 1; % Compute p(choice)
        data(k, 2:6) = [cor, choice, correct, r, choice_prob];
        
        if choice == 1
            side = 1; % Stick to the side of the previous action
        else
            side = -1;
        end
        
        Q(choice) = Q(choice) + alpha * (r - Q(choice)); % Update Q values
    end
end

end
