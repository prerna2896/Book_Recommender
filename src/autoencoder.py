import os
import time
import sys
import numpy as np
import random
from scipy import stats

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.ioff()

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.parallel

import torch.optim as optim
import torch.utils.data

####################################################################################################
# CONSTANTS AND HYPERPARAMETERS
####################################################################################################

# Use GPU if available
DEVICE                      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants for the data set 
DATASET_FILE                = "../data/ratings.csv"

UNKNOWN_RATING              = 0
MIN_RATING                  = 1
MAX_RATING                  = 5

NUM_DEV_TEST_USERS          = 0.5
NUM_DEV_BOOKS               = 0.3
NUM_TEST_BOOKS              = 0.2
NUM_DEV_TEST_BOOKS          = NUM_DEV_BOOKS + NUM_TEST_BOOKS

# Hyperparameters for the model
ACTIVATION                  = 'Sigmoid'
HIDDEN_DIM                  = 32
NUM_STACKS                  = 4
LEARNING_RATE               = 0.04
WEIGHT_DECAY                = 0.0
LOSS_FUNCTION               = 'RMSE'
NUM_ITERATIONS              = 200
BATCH_SIZE                  = 1113
OPTIMIZER                   = 'Adam'
MODEL_NAME                  = 'model.stackedAutoencoder'

print("\n")
print("Initializing...")

####################################################################################################
# DATASET
####################################################################################################

# Load the data from the file
raw_data = np.loadtxt(DATASET_FILE, dtype=np.int, delimiter=",")
num_users = int(stats.describe(raw_data[:,0]).minmax[1])
num_books = int(stats.describe(raw_data[:,1]).minmax[1])
data = np.zeros(shape=(num_users, num_books))
data.fill(UNKNOWN_RATING)
data[raw_data[:,0] - 1, raw_data[:,1] - 1] = raw_data[:,2]

# Normalize the data
np.random.shuffle(data)
num_users, num_books = data.shape
print(data.shape)

# Divide the data into train, dev and test
train_data  = np.copy(data)

dev_data    = np.zeros(shape=data.shape)
dev_data.fill(UNKNOWN_RATING)

test_data   = np.zeros(shape=data.shape)
test_data.fill(UNKNOWN_RATING)

num_dev_test_users  = int(NUM_DEV_TEST_USERS    * num_users)
num_dev_books       = int(NUM_DEV_BOOKS         * num_books)
num_test_books      = int(NUM_TEST_BOOKS        * num_books)
num_dev_test_books  = int(NUM_DEV_TEST_BOOKS    * num_books)

train_data  [num_dev_test_users : , num_dev_test_books  :               ]   = UNKNOWN_RATING
dev_data    [num_dev_test_users : , -num_dev_test_books : -num_dev_books]   = data[num_dev_test_users : , -num_dev_test_books   : -num_dev_books]
test_data   [num_dev_test_users : , -num_dev_books      :               ]   = data[num_dev_test_users : , -num_dev_books        :               ]

# train_data  = torch.tensor(train_data,  device = DEVICE, dtype=torch.float)
# dev_data    = torch.tensor(dev_data,    device = DEVICE, dtype=torch.float)
# test_data   = torch.tensor(test_data,   device = DEVICE, dtype=torch.float)

####################################################################################################
# STACKED AUTOENCODER MODEL
####################################################################################################

# The stacked auto encoder model
class StackedAutoEncoder(nn.Module):
    def __init__(self, input_dim = num_books, hidden_dim = 5, output_dim = num_books, activation = 'ReLU', num_stacks = 6):
        super(StackedAutoEncoder, self).__init__()

        if activation.lower() == 'relu':
            F = nn.ReLU()
        if activation.lower() == 'tanh':
            F = nn.Tanh()
        if activation.lower() == 'sigmoid':
            F = nn.Sigmoid()

        self.ae = nn.ModuleList([
                    nn.Sequential(nn.Linear(input_dim, hidden_dim), F, nn.Linear(hidden_dim, output_dim))
                    for i in range(num_stacks)])

    def forward(self, x, n):
        for i in range(n - 1):
            self.ae[i].requires_grad = False
        for i in range(n):
            x = self.ae[i](x)
        return x

####################################################################################################
# LOSS FUNCTIONS
####################################################################################################

def Precision_Recall_TopK(predicted, actual, K = 10):
    actual      = actual.cpu().detach().numpy()
    predicted   = predicted.cpu().detach().numpy()

    n, d        = actual.shape
    
    mask_actual = (actual    != UNKNOWN_RATING) * (actual     >= (0.6 * MAX_RATING))
    mask_pred   = (actual    != UNKNOWN_RATING) * (predicted  >= (0.6 * MAX_RATING))

    actual      = actual    * mask_actual
    predicted   = predicted * mask_pred

    precision   = 0
    recall      = 0
    for i in range(n):
        relevant_items  = set(filter(lambda item: actual[i][item] != 0, range(d)))
        topK_pred       = np.argsort(-predicted[i])[:K]
        topK_pred       = set(filter(lambda item: predicted[i][item] != 0, topK_pred))
    
        num_common  = len(relevant_items.intersection(topK_pred))
        precision   += num_common / len(topK_pred)      if len(topK_pred)       != 0    else 1
        recall      += num_common / len(relevant_items) if len(relevant_items)  != 0    else 1

    precision   = precision / n
    recall      = recall / n
    F1          = 2 * (precision * recall) / (precision + recall)
    return precision, recall, F1


def Absolute_Square_Error(predicted, actual):
    # Get the mask
    mask        = actual != UNKNOWN_RATING
    mask        = mask.float()  

    # Mask the columns in the output where the input is unrated
    actual      = actual    * mask
    predicted   = predicted * mask

    # Total number of ratings
    num_ratings = torch.sum(mask)

    # Calculate the square of the errors
    error       = torch.sum((actual - predicted) ** 2)
    return error, num_ratings

def getLoss(error, num_ratings, loss_function='MMSE'):
    num_ratings += 1e-5
    if (loss_function == 'MMSE'):
        return error / num_ratings
    elif (loss_function == 'RMSE'):
        return (error / num_ratings) ** 0.5

####################################################################################################
# TRAIN AND TEST
####################################################################################################

def train(hidden_dim, activation, num_stacks, learing_rate, weight_decay, loss_function, num_iterations, optimizer, save_model = False):
    # Training on train data
    stackedAutoEncoder  = StackedAutoEncoder(hidden_dim = hidden_dim, activation = activation, num_stacks = num_stacks).to(DEVICE)

    if optimizer.lower() == 'adam':
        opt = optim.Adam(stackedAutoEncoder.parameters(), lr = learing_rate, weight_decay = weight_decay)
    if optimizer.lower() == 'sgd':
        opt = optim.SGD(stackedAutoEncoder.parameters(), lr = learing_rate, weight_decay = weight_decay)
    if optimizer.lower() == 'rmsprop':
        opt = optim.RMSprop(stackedAutoEncoder.parameters(), lr = learing_rate, weight_decay = weight_decay)


    print("Training...")
    # Train the model
    epoch_train_loss    = []
    epoch_dev_loss      = []
    time_train_loss     = []
    time_dev_loss       = []
    start_time          = time.time()
    for i in range(num_iterations):
        n = int(i / (num_iterations / num_stacks)) + 1

        opt.zero_grad()
        
        total_error = 0
        total_ratings = 0
        num_batches = num_users // BATCH_SIZE
        for batch in range(num_batches):
            next_batch = torch.tensor(train_data[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE],  device = DEVICE, dtype=torch.float)
            predicted_ratings = stackedAutoEncoder(next_batch, n)
            error_ratings = Absolute_Square_Error(predicted_ratings, next_batch)
            loss = getLoss(error_ratings[0], error_ratings[1], loss_function)
            total_error += float(error_ratings[0])
            total_ratings += float(error_ratings[1])
            loss.backward()
        total_loss = getLoss(total_error, total_ratings, loss_function)
        
        opt.step()

        end_time = time.time() - start_time

        epoch_train_loss.append((i + 1, total_loss))
        time_train_loss.append((end_time, total_loss))

        dev_loss = dev(stackedAutoEncoder, loss_function, n)
        epoch_dev_loss.append((i + 1, dev_loss))
        time_dev_loss.append((end_time, dev_loss))

        print("Epoch #", (i + 1), ":\t Training loss: ", round(total_loss, 8), "\t Dev loss: ", round(dev_loss, 8))


    print("Training finished.\n")

    if (save_model):
        print("Saving model...")
        torch.save(stackedAutoEncoder, MODEL_NAME)
        print("Saved model.\n")
   
    train_metrics   = (epoch_train_loss, time_train_loss)
    dev_metrics     = (epoch_dev_loss, time_dev_loss)
    return (train_metrics, dev_metrics)

def dev(model, loss_function, num_stacks):
    total_error = 0
    total_ratings = 0
    num_batches = num_users // BATCH_SIZE
    for batch in range(num_batches):
        next_batch = torch.tensor(dev_data[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE],  device = DEVICE, dtype=torch.float)
        predicted_ratings = model(next_batch, num_stacks)
        error_ratings = Absolute_Square_Error(predicted_ratings, next_batch)
        total_error += float(error_ratings[0])
        total_ratings += float(error_ratings[1])
    dev_loss = getLoss(total_error, total_ratings, loss_function)
    return dev_loss

def test(model, loss_function, num_stacks):
    print("Testing...")
    total_error = 0
    total_ratings = 0
    num_batches = num_users // BATCH_SIZE
    for batch in range(num_batches):
        next_batch = torch.tensor(test_data[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE],  device = DEVICE, dtype=torch.float)
        predicted_ratings = model(next_batch, num_stacks)
        error_ratings = Absolute_Square_Error(predicted_ratings, next_batch)
        total_error += float(error_ratings[0])
        total_ratings += float(error_ratings[1])
    test_loss = getLoss(total_error, total_ratings, loss_function)
    print("Loss on test data: ", test_loss)

    print("\n")

    return test_loss

####################################################################################################
# EXPERIMENTATION
####################################################################################################

def plot_images(plot_data, labels, xlabel, ylabel, filename):
    refined_data = []
    for data in plot_data:
        refined_data.append(list(filter(lambda x: x[1] < 25, data)))

    plt.clf()
    for data, label in zip(refined_data, labels):
        xs = [x[0] for x in data]
        ys = [y[1] for y in data]
        plt.plot(xs, ys, label=label)
    plt.legend(loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.clf()

def experiment_learning_rate():
    print("Experimenting with learning rate...")
    learning_rates = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08]

    plot_data_train = []
    plot_data_dev = []
    labels = []
    for learning_rate in learning_rates:
        print("Trying learning rate: " + str(learning_rate))
        train_metrics, dev_metrics = train(HIDDEN_DIM, ACTIVATION, NUM_STACKS, learning_rate, WEIGHT_DECAY, "RMSE", NUM_ITERATIONS, OPTIMIZER, save_model = False)
        plot_data_train.append(train_metrics[0])
        plot_data_dev.append(dev_metrics[0])
        label = "Learning rate: " + str(learning_rate)
        labels.append(label)

    plot_images(plot_data_train, labels, "Epoch", "Root Mean squared error", "../images/VaryingLearningRate_RMSE_Train.png")
    plot_images(plot_data_dev, labels, "Epoch", "Root Mean squared error", "../images/VaryingLearningRate_RMSE_Dev.png")


def experiment_hidden_dim():
    print("Experimenting with hidden dimensions...")
    hidden_dims = [8, 16, 32, 64]

    plot_data_train = []
    plot_data_dev = []
    labels = []
    for hidden_dim in hidden_dims:
        print("Trying hidden dimension: " + str(hidden_dim))
        train_metrics, dev_metrics = train(hidden_dim, ACTIVATION, NUM_STACKS, LEARNING_RATE, WEIGHT_DECAY, "RMSE", NUM_ITERATIONS, OPTIMIZER, save_model = False)
        plot_data_train.append(train_metrics[0])
        plot_data_dev.append(dev_metrics[0])
        label = "Hidden dimension: " + str(hidden_dim)
        labels.append(label)

    plot_images(plot_data_train, labels, "Epoch", "Root Mean squared error", "../images/VaryingHiddenDim_RMSE_Train.png")
    plot_images(plot_data_dev, labels, "Epoch", "Root Mean squared error", "../images/VaryingHiddenDim_RMSE_Dev.png")


def experiment_num_stack():
    print("Experimenting with number of stacks...")
    num_stacks = [4, 5, 8, 10, 20]

    plot_data_train = []
    plot_data_dev = []
    labels = []
    for num_stack in num_stacks:
        print("Trying number of stacks: " + str(num_stack))
        train_metrics, dev_metrics = train(HIDDEN_DIM, ACTIVATION, num_stack, LEARNING_RATE, WEIGHT_DECAY, "RMSE", NUM_ITERATIONS, OPTIMIZER, save_model = False)
        plot_data_train.append(train_metrics[0])
        plot_data_dev.append(dev_metrics[0])
        label = "Number of stacks: " + str(num_stack)
        labels.append(label)

    plot_images(plot_data_train, labels, "Epoch", "Root Mean squared error", "../images/VaryingNumStack_RMSE_Train.png")
    plot_images(plot_data_dev, labels, "Epoch", "Root Mean squared error", "../images/VaryingNumStack_RMSE_Dev.png")


def experiment_optimizer():
    print("Experimenting with optimizer...")
    optimizers = ['Adam', 'SGD', 'RMSProp']

    plot_data_train = []
    plot_data_dev = []
    labels = []
    for optimizer in optimizers:
        print("Trying optimizer: " + str(optimizer))
        train_metrics, dev_metrics = train(HIDDEN_DIM, ACTIVATION, NUM_STACKS, LEARNING_RATE, WEIGHT_DECAY, "RMSE", NUM_ITERATIONS, optimizer, save_model = False)
        plot_data_train.append(train_metrics[0])
        plot_data_dev.append(dev_metrics[0])
        label = "Optimizer: " + str(optimizer)
        labels.append(label)

    plot_images(plot_data_train, labels, "Epoch", "Root Mean squared error", "../images/VaryingOptimizer_RMSE_Train.png")
    plot_images(plot_data_dev, labels, "Epoch", "Root Mean squared error", "../images/VaryingOptimizer_RMSE_Dev.png")

def experiment_activation():
    print("Experimenting with activation function...")
    activations = ['ReLU', 'Tanh', 'Sigmoid']

    plot_data_train = []
    plot_data_dev = []
    labels = []
    for activation in activations:
        print("Trying activation: " + str(activation))
        train_metrics, dev_metrics = train(HIDDEN_DIM, activation, NUM_STACKS, LEARNING_RATE, WEIGHT_DECAY, "RMSE", NUM_ITERATIONS, OPTIMIZER, save_model = False)
        plot_data_train.append(train_metrics[0])
        plot_data_dev.append(dev_metrics[0])
        label = "Activation: " + str(activation)
        labels.append(label)

    plot_images(plot_data_train, labels, "Epoch", "Root Mean squared error", "../images/VaryingActivation_RMSE_Train.png")
    plot_images(plot_data_dev, labels, "Epoch", "Root Mean squared error", "../images/VaryingActivation_RMSE_Dev.png")

def run_experiments():
    experiment_learning_rate()
    experiment_hidden_dim()
    experiment_num_stack()
    experiment_optimizer()
    experiment_activation()

####################################################################################################
# USER INTERACTION FOR TRAINING AND TESTING MODELS
####################################################################################################

mode = sys.argv[1]

if (mode == 'train'):
    # Training on train data
    plot_data_train     = []
    plot_data_dev       = []
    time_data_train     = []
    time_data_dev       = []
    labels = []

    train_metrics, dev_metrics = train(HIDDEN_DIM, ACTIVATION, NUM_STACKS, LEARNING_RATE, WEIGHT_DECAY, LOSS_FUNCTION, NUM_ITERATIONS, OPTIMIZER, save_model=True)

    plot_data_train.append(train_metrics[0])
    plot_data_dev.append(dev_metrics[0])
    time_data_train.append(train_metrics[1])
    time_data_dev.append(dev_metrics[1])
    label = "Deep Autoencoder"
    labels.append(label)

    plot_images(plot_data_train, labels, "Epoch", "Root Mean squared error", "../images/StackedAutoencoder_RMSE_Train.png")
    plot_images(plot_data_dev, labels, "Epoch", "Root Mean squared error", "../images/StackedAutoencoder_RMSE_Dev.png")
    plot_images(time_data_train, labels, "Time", "Root Mean squared error", "../images/StackedAutoencoder_RMSE_Train_Timed.png")
    plot_images(time_data_dev, labels, "Time", "Root Mean squared error", "../images/StackedAutoencoder_RMSE_Dev_Timed.png")

    with open("../images/Time_Error.txt", "a") as f:
        f.write(','.join(str(data[0]) for data in train_metrics[1]))
        f.write('\n')
        f.write(','.join(str(data[1]) for data in train_metrics[1]))
        f.write('\n')
        f.write(','.join(str(data[0]) for data in dev_metrics[1]))
        f.write('\n')
        f.write(','.join(str(data[1]) for data in dev_metrics[1]))
        f.write('\n')


elif (mode == 'test'):
    # Testing on test data
    print("Loading model...")
    stackedAutoEncoder = torch.load(MODEL_NAME, map_location = torch.device(DEVICE))
    print("Loaded model.")

    test(stackedAutoEncoder, LOSS_FUNCTION, NUM_STACKS)

elif (mode == 'exp'):
    # Run the experiments
    run_experiments()

else:
    print("Usage: python3 stackedAutoencoders.py <train | test | exp>")

print('\n')