import numpy as np
def build_neural_network(number_of_columns, hidden1):
    neural_arch = {
        'hidden_layer': np.array([np.random.randn() 
            for i in range(number_of_columns*hidden1)]).reshape(number_of_columns,hidden1),
        'output_layer':np.array([np.random.randn() 
            for i in range(hidden1*1)]).reshape(hidden1,1)
    }
    return neural_arch

neural_weights = build_neural_network(number_of_columns= 15,hidden1=20)
#print(neural_weights)
def feed_forward(row):
    hidden_layer = neural_weights['hidden_layer']
    print("hidden layers",hidden_layer)
    for i in range(1):
        input1 = np.multiply(hidden_layer[i],row.transpose())
        print("Input 1 = ", input1, input1.shape)

feed_forward(np.array([np.random.randn() for i in range(15)]).reshape(1,15))
