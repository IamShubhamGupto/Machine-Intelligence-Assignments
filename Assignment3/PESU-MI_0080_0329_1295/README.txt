<TEAM NAME> Assignment 3
Requirements before running:
	numpy version 1.19.1
	pandas version 0.24.2
	scikit version 0.21.3

All code was developed using jupyter notebook python 3.6.5
Model Description:

	Number of layers 4:
		(Virtual) Input layer - Neurons = number of columns in processed dataset
		Hidden 1 layer - 512 neurons - tanh
		Hidden 2 layer - 256 neurons - tanh
		Hidden 3 layer - 512 neurons - tanh
		Output layer - 1 Neuron - sigmoid

	Hyper parameters used:
		Learning rate - 8e-6
		Number of neurons - [512,256,512,1]
		Number of layers - 4
		NAG Momentum - 0.999
		Testing size - 0.3
		Epochs - 100
		Learning rate decay - Learning rate/Number of epochs
		random_state = 1000000007 - splitting test train data
	
	Key points about the model:
		Started with simple forward and backward propagation
		Added more layers due to fluctuating results
		Initial layer weights are initialized using Xavier's initialization method
		Implemented momentum
		Improved momentum by using Nesterov's Adaptive Gradient momentum
		Tested different learning rate decay methods:
			Constant learning rate
			Step decay
			Exponential decay
			Time based decay
		Implemented a custom decay rate formula based on time based and exponential decay
		Failed implementations:
		Using ReLU and Swish activation functions
		Tried different types of regularization to avoid overfitting
			Dropout: Failed as dropout masks generated were not accurate
			L2 Regularization: Not in submitted version as it did not seem to help
		Early stopping: Removed from final version due to improper implementation

	Key features and beyond the basics:
		Large number of neurons, small learning rate, high momentum
		Implemented a wide neural network
		Implementation of Nesterov's momentum
		Time based decay:
			lr = lr*(1/(1+decay*t)) 
		Exponential decay:
			lr = lr0*exp^(-decay*t)
		Ours:
			lr = lr0*(1/(1+decay*t)) 
		lr -> learning rate, lr0 ->initial learning rate, decay -> rate, t -> iteration 
	
	Steps to run the assignment:
		python3 MI_A3.py

Maximum accuracy achieved on test dataset:

Y_accc     [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]
Y_test_obs [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]
Confusion Matrix : 
[[4, 1], [3, 20]]


Precision : 0.9523809523809523
Recall : 0.8695652173913043
F1 SCORE : 0.909090909090909
Accuracy :  0.8571428571428571
		
Maximum accuracy achieved on train dataset:
Accuracy :  0.8153846153846154

However we have achived higher accuracies for keeping y_hat threshold > 0.55 (84% on train and 85% on test) and y_hat > 0.5 (85% on train and 89% on test). We have achived 89% accuracy with 0.6 as a threshold but that seems like a occasional event - depends on random weight initialization. Hence we put our observed highest accuracy as 85%
To get maximum accuracy, run only the python code with no other background programs (not sure if its coincidence)	