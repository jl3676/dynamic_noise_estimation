The data used in this manuscript are contained within the MATLAB structure 'data' in the 'data.mat' file.

The data are first split by behavior task, either Dynamic Foraging ('dynamicForaging') or Probabilistic Pavlovian ('probabilisticPavlovian').

Within each task, the data are sorted according to the analyses for which they were used.

For the chemogenetics experiments, the data are further subdivided by group (DREADDs or mCherry) and condition (Agonist 21 or vehicle injection).

Within each experiment, data are grouped by mouse identifiers comprising two letters and two numbers. 

All relevant sessions for each mouse are nested under that identifier. Each session follows the format of 'mXXXXdYYYYYYYY' where X represents mouse identifier and Y the date in year, month, day order.

Some sessions are used for multiple analyses (e.g., behavior and neural) and so appear multiple times in the appropriate branches.

Each session consists of a structure containing column-wise fields related to behavior, neural, and task events. Each row contains data for each unique trial.

In Dynamic Foraging sessions, there are 10 fields related to behavior and task events:
	- trialType: a string indicating which cue was delivered, a go-cue ('CSplus') or rare no-go cue ('CSminus')
	- trialEnd: a numeric value indicating the timestamp at which the trial was completed (end of the inter-trial interval)
	- CSon: timestamp at which the odor cue was delivered
	- licksL: an array of timestamps for any licks to the left spout during the entire trial, can be empty if no licks are made to that side
	- licksR: an array of timestamps for any licks to the right spout during the entire trial, can be empty if no licks are made to that side
	- rewardL: an indicator variable indicating an outcome on the left spout, NaN if no choice was made to this spout, 0 if a choice was not rewarded, and 1 if a choice was rewarded 
	- rewardR: an indicator variable indicating an outcome on the right spout, NaN if no choice was made to this spout, 0 if a choice was not rewarded, and 1 if a choice was rewarded 
	- rewardTime: time at which outcome was delivered, consequent to the first decision lick
	- rewardProbL: reward probability assigned to the left spout
	- rewardProbR: reward probability assigned to the right spout

In Probabilistic Pavlovian sessions, there are 9 fields related to behavior and task events:
	- trial: numeric indicator of trial number in session
	- CSon: timestamp at which the odor cue begun to be delivered
	- CSoff: timestamp at which odor cue turned off
	- USon: time at which outcome begun to be delivered
	- USoff: time at which outcome delivery ended
	- trialEnd: timestamp at which the trial was completed (end of the inter-trial interval)
	- licks: timestamps any licks of the spout during the entire trial
	- trialType: a string indicating which cue was delivered, a go-cue ('CSplus') or rare no-go cue ('CSminus')
	- rewarded: an indicator variable indicating outcome, 0 for no reward and 1 for reward

For both tasks, in the experiments in which neural data was collected, any identified serotonin neurons recorded during that session appear in separate fields with an identifier. The cell name follows the format 'TTX_SS_YY', where X is the tetrode number and YY is the cell number on that tetrode. Each row contains timestamps of action potentials for the relevant neuron.

For behavior experiments, timestamps were collected from the Arduino Uno microcontroller that was programmed to run the task and detect licks. For electrophysiology experiments, timestamps were collected from the Intan recording system, that logged neural activity times along with copies of task signals sent from the Arduino Uno. 

