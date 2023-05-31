function data = dynamic_model(theta, K, realData)
% Simulates data using the dynamic RLWM model based on the provided parameters.
% Inputs:
%   - theta: A vector of model parameters [alpha, bias, stick, rho, forget, lapse, recover].
%   - K: A parameter used to compute the working memory weight.
%   - realData: The real data used to determine the set size and block information.
%
% Output:
%   - data: Simulated data based on the dynamic RLWM model.

beta = 20;
alpha = theta(1);
bias = theta(2);
stick = theta(3);
rho = theta(4);
forget = theta(5);
epsilon = 0;
lapse = theta(6);
recover = theta(7);

nA = 3;
engaged = 1; 
actions = [1 2 3];

k = 0;
all_blocks = unique(realData(:,5))';

for block = all_blocks
    this_data = realData(realData(:,5)==block,:);
    ns = this_data(1,4);
    num_trials = size(this_data,1);

    % WM weight
    w = rho*min(1,K/ns);

    side=zeros(1,nA);
    Q = (1/nA)*ones(ns,nA);
    WM = (1/nA)*ones(ns,nA);

    for trial = 1:num_trials
        k = k + 1;
        
        s = this_data(trial,1);
        cor = this_data(trial,3);

        data(k,6) = engaged;

        % compute action sampling probabilities
        if engaged == 1
            pRL = exp(beta * (Q(s,:) + side * stick));
            pRL = epsilon/nA + (1-epsilon)*pRL/sum(pRL);
            pWM = exp(beta * (WM(s,:) + side * stick));
            pWM = epsilon/nA + (1-epsilon)*pWM/sum(pWM); 
            pr = w * pWM + (1 - w) * pRL;
            if rand < lapse, engaged = 0; end % lapse with a probability
        else
            pr = ones(1,nA) / nA;
            if rand < recover, engaged = 1; end % recoverurn to attentive state with a probability
        end

        % sample choice
        choice = actions(find(rand<cumsum(pr),1,'first'));
        if trial <= ns
            choice = this_data(trial,2);
        end
        correct = choice == cor;
        r = correct; % deterministic reward
        
        if r==1
            alphaRL = alpha;
            alphaWM = 1;
        else
            alphaRL = bias*alpha;
            alphaWM = bias;
        end

        WM = WM + forget*(1/nA - WM);
        WM(s,choice) = WM(s,choice) + alphaWM*(r-WM(s,choice));
        Q(s,choice) = Q(s,choice) + alphaRL*(r-Q(s,choice));

        side=zeros(1,nA);
        side(choice)=1;

        data(k,1:5) = [s choice r ns block];
        data(k,7) = pr(choice);
    end
    
    for trials=num_trials:13*ns
        k = k + 1;
        data(k,:) = [nan nan nan ns block nan nan];
    end
end
end
