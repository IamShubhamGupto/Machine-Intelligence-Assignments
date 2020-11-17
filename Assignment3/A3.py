import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
def new_predict(history): #this function returns the predicted results of Y_test and original values
    '''
    It runs all the samples from the Y_test through the network , updates weights and biases at each layer 
    Uses these weights and biases to predict the output, and that is stored in the variable result
    '''
    result = []
    result.append(neural_net.predict_test(history[1]))
    return result,history[2][0]
def normalize(df, columns):
    result = df.copy()
    for feature_name in columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result
def cleaning(df):
    df['Education'].fillna(5.0, inplace=True)
    df['Residence']=df['Residence'].fillna(method='ffill')
    df['Weight']=df['Weight'].fillna(df['Weight'].mean())
    df['BP']=df['BP'].fillna(df['BP'].mean())
    df['HB']=df['HB'].fillna(df['HB'].mean())
    df['Delivery phase']=df['Delivery phase'].fillna(method='ffill')
    df['Age']=df['Age'].fillna(df['Age'].mean())
    #normalized_df = normalize(df, ['Age','Weight','HB','BP',])
    #dum_df = pd.get_dummies(df, columns=["Community"])
    return df
def activation(z, derivative=False):
    """
    Sigmoid activation function:
    It handles two modes: normal and derivative mode.
    Applies a pointwize operation on vectors
    
    Parameters:
    ---
    z: pre-activation vector at layer l
        shape (n[l], batch_size)

    Returns: 
    pontwize activation on each element of the input z
    """
    if derivative:
        # return 1 - activation(z)**2
        return activation(z)*(1-activation(z))
    else:
        return 1 / (1 + np.exp(-z))
        # return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def cost_function(y_true, y_pred):
    """
    Computes the Mean Square Error between truth values and a prediction values
    Parameters:
    ---
    y_true: ground-truth vector
    y_pred: prediction vector
    Returns:
    ---
    cost: a scalar value representing the loss
    """
    n = y_pred.shape[1]
    cost = (1./(2*n)) * np.sum((y_true - y_pred) ** 2)
    return cost

def cost_function_prime(y_true, y_pred):
    """
    Computes the derivative of the loss function w.r.t the activation of the output layer
    Parameters:
    ---

    Returns:
    ---
    cost_prime: derivative of the loss w.r.t. the activation of the output
    shape: (n[L], batch_size)    
    """
    cost_prime = y_pred - y_true
    return cost_prime


class NN(object):
    '''' X and Y are dataframes '''
    def __init__(self, size, seed=42):
        """
        Instantiate the weights and biases of the network
        weights and biases are attributes of the NeuralNetwork class
        They are updated during the training
        """
        self.seed = seed
        np.random.seed(self.seed)
        self.size = size
        self.weights = [np.random.randn(self.size[i], self.size[i-1]) * np.sqrt(1 / self.size[i-1]) for i in range(1, len(self.size))]
        self.biases = [np.random.rand(n, 1) for n in self.size[1:]]
    def forward(self, input):
        '''
        Perform a feed forward computation 

        Parameters
        ---
        input: data to be fed to the network with
        shape: (input_shape, batch_size)

        Returns
        ---
        a: ouptut activation (output_shape, batch_size)
        pre_activations: list of pre-activations per layer
        each of shape (n[l], batch_size), where n[l] is the number 
        of neuron at layer l
        activations: list of activations per layer
        each of shape (n[l], batch_size), where n[l] is the number 
        of neuron at layer l

        '''
        a = input
        pre_activations = []
        activations = [a]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a  = activation(z)
            pre_activations.append(z)
            activations.append(a)
        return a, pre_activations, activations

    def compute_deltas(self, pre_activations, y_true, y_pred):
        """
        Computes a list containing the values of delta for each layer using 
        a recursion
        Parameters:
        ---
        pre_activations: list of of pre-activations. each corresponding to a layer
        y_true: ground truth values of the labels
        y_pred: prediction values of the labels
        Returns:
        ---
        deltas: a list of deltas per layer
        
        """
        delta_L = cost_function_prime(y_true, y_pred) * activation(pre_activations[-1], derivative=True)
        deltas = [0] * (len(self.size) - 1)
        deltas[-1] = delta_L
        for l in range(len(deltas) - 2, -1, -1):
            delta = np.dot(self.weights[l + 1].transpose(), deltas[l + 1]) * activation(pre_activations[l], derivative=True) 
            deltas[l] = delta
        return deltas

    def backpropagate(self, deltas, pre_activations, activations):
        """
        Applies back-propagation and computes the gradient of the loss
        w.r.t the weights and biases of the network

        Parameters:
        ---
        deltas: list of deltas computed by compute_deltas
        pre_activations: a list of pre-activations per layer
        activations: a list of activations per layer
        Returns:
        ---
        dW: list of gradients w.r.t. the weight matrices of the network
        db: list of gradients w.r.t. the biases (vectors) of the network
    
        """
        dW = []
        db = []
        deltas = [0] + deltas
        for l in range(1, len(self.size)):
            dW_l = np.dot(deltas[l], activations[l-1].transpose()) 
            db_l = deltas[l]
            dW.append(dW_l)
            db.append(np.expand_dims(db_l.mean(axis=1), 1))
        return dW, db

    

    def fit(self, X, y, batch_size, epochs, learning_rate, validation_split=0.2, print_every=10, tqdm_=True, plot_every=None):
        """
        Trains the network using the gradients computed by back-propagation
        Splits the data in train and validation splits
        Processes the training data by batches and trains the network using batch gradient descent

        Parameters:
        ---
        X: input data
        y: input labels
        batch_size: number of data points to process in each batch
        epochs: number of epochs for the training
        learning_rate: value of the learning rate
        validation_split: percentage of the data for validation
        print_every: the number of epochs by which the network logs the loss and accuracy metrics for train and validations splits
        tqdm_: use tqdm progress-bar
        plot_every: the number of epochs by which the network plots the decision boundary
    
        Returns:
        ---
        history: dictionary of train and validation metrics per epoch
            train_acc: train accuracy
            test_acc: validation accuracy
            train_loss: train loss
            test_loss: validation loss

        This history is used to plot the performance of the model
        """
        history_train_losses = []
        history_train_accuracies = []
        history_test_losses = []
        history_test_accuracies = []

        x_train, x_test, y_train, y_test = train_test_split(X.T, y.T, test_size=validation_split, )
        x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T
        #print(x_train)
        #x_train=X.T
        #y_train=y.T
        if tqdm_:
            epoch_iterator = tqdm(range(epochs))
        else:
            epoch_iterator = range(epochs)

        for e in epoch_iterator:
            if x_train.shape[1] % batch_size == 0:
                n_batches = int(x_train.shape[1] / batch_size)
            else:
                n_batches = int(x_train.shape[1] / batch_size ) - 1

            x_train, y_train = shuffle(x_train.T, y_train.T)
            x_train, y_train = x_train.T, y_train.T

            batches_x = [x_train[:, batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]
            #print(batches_x)
            
            batches_y = [y_train[:, batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]
            #print(batches_y)
            train_losses = []
            train_accuracies = []
            
            test_losses = []
            test_accuracies = []

            dw_per_epoch = [np.zeros(w.shape) for w in self.weights]
            db_per_epoch = [np.zeros(b.shape) for b in self.biases] 
            
            for batch_x, batch_y in zip(batches_x, batches_y):
                batch_y_pred, pre_activations, activations = self.forward(batch_x)
                deltas = self.compute_deltas(pre_activations, batch_y, batch_y_pred)
                dW, db = self.backpropagate(deltas, pre_activations, activations)
                for i, (dw_i, db_i) in enumerate(zip(dW, db)):
                    dw_per_epoch[i] += dw_i / batch_size
                    db_per_epoch[i] += db_i / batch_size

                batch_y_train_pred = self.predict(batch_x)
                
                
                train_loss = cost_function(batch_y, batch_y_train_pred)
                train_losses.append(train_loss)
                train_accuracy = accuracy_score(batch_y.T, batch_y_train_pred.T)
                train_accuracies.append(train_accuracy)

                batch_y_test_pred = self.predict(x_test)

                test_loss = cost_function(y_test, batch_y_test_pred)

                test_losses.append(test_loss)
                test_accuracy = accuracy_score(y_test.T, batch_y_test_pred.T)
                test_accuracies.append(test_accuracy)


            # weight update
            for i, (dw_epoch, db_epoch) in enumerate(zip(dw_per_epoch, db_per_epoch)):
                self.weights[i] = self.weights[i] - learning_rate * dw_epoch
                self.biases[i] = self.biases[i] - learning_rate * db_epoch

            history_train_losses.append(np.mean(train_losses))
            history_train_accuracies.append(np.mean(train_accuracies))
            
            history_test_losses.append(np.mean(test_losses))
            history_test_accuracies.append(np.mean(test_accuracies))



        history = {'epochs': epochs,
                   'train_loss': history_train_losses, 
                   'train_acc': history_train_accuracies,
                   'test_loss': history_test_losses,
                   'test_acc': history_test_accuracies
                   
                   }
        
        return history,x_test,y_test   

    def predict(self, a):
        #print(a)
        '''
        Use the current state of the network to make predictions

        Parameters:
        ---
        a: input data, shape: (input_shape, batch_size)

        Returns:
        ---
        predictions: vector of output predictions
        '''
        for w, b in zip(self.weights, self.biases):

            z = np.dot(w, a) + b
            a = activation(z)
        predictions = (abs(a) > 0.47).astype(int)
        return predictions
    def predict_test(self, a):
        #print(a)
        '''
        Use the current state of the network to make predictions

        Parameters:
        ---
        a: input data, shape: (input_shape, batch_size)

        Returns:
        ---
        predictions: vector of output predictions
        '''
        for w, b in zip(self.weights, self.biases):

            z = np.dot(w, a) + b
            a = activation(z)
    
        return a

    def CM(self,y_test,y_pred):

        for i in range(len(y_pred)):
            if(abs(y_pred[i])>0.735):
                y_pred[i]=1
            else:
                y_pred[i]=0
        
        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0
        
        for i in range(len(y_test)):
            if(y_test[i]==1 and y_pred[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_pred[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_pred[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_pred[i]==1):
                fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp

        p= tp/(tp+fp)
        r=tp/(max(1,tp+fn))
        f1=(2*p*r)/(max(1,p+r))
        a = (tn+tp)/(tn+fp+fn+tp)
        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Precision : {round(p*100)}%")
        print(f"Accuracy : {round(a*100)}%")
        print(f"Recall : {round(r*100)}%")
        print(f"F1 SCORE : {round(f1*100)}%")
        print(f"True Values : {y_test}")
        print(f" Predicted Values : {y_pred}")
    
#-------------------------------------------------------End of Class ------------------------------------------------------------------------

if __name__ == "__main__":
    PATH = r"F:/engi_books/sem5/mi/UE18CS303_Assignment/Assignment3/LBW_Dataset.csv"
    raw_df=pd.read_csv(PATH)
    df = cleaning(raw_df)
    #df = clean_df(r'F:\Engi_Books\Sem5\MI\UE18CS303_Assignment\Assignment3\LBW_Dataset.csv')
    print(df.Result.value_counts()) # Checks the number of samples in each category (1/0)
    X=df.iloc[:,:-1]
    Y=df.iloc[:,-1]
    X=X.to_numpy()
    #y=df.iloc[:,-1]
    #y=y.to_numpy()

    y=np.expand_dims(Y.to_numpy(), 1)
    X_train,X_test,Y_train,Y_test = train_test_split(df.drop(["Result"],axis = 1), Y, stratify = Y,
                                                    test_size = 0.3, random_state = 123)
    neural_net = NN([X_train.shape[1],11,5, 1],seed=0)
    # neural_net = NN([9,50, 1],seed=5)
    history = neural_net.fit(X=X.T, y=y.T, batch_size=16, epochs=100, learning_rate=0.4, print_every=200, validation_split=0.2,tqdm_=False, plot_every=10)
    '''
    History returns the following lists : 
    history[0] : Details such as epochs, train_loss,Train_accuracy, test_loss, etc..
    history[1] : returns the test samples (X_test)
    history[2] : return the test output labels (Y_test) 
    '''
    res,y_test = new_predict(history)
    print(res[0].flatten(),y_test.flatten()) #prints both the lists
    neural_net.CM(y_test.flatten(),res[0].flatten()) #Prints the evaluation metric results such as Precision, Recall , Accuracy.
    
