import gc
import os
import pickle
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from skopt import load
import pandas as pd

from EEG_Model.EEG_gpt2 import EEG_GPT2
from EEG_Model.EventDataset import EventDataset

""""
Global variables are necessary due to the Bayesian Optimization used
"""
train_dataset, valid_dataset, test_dataset = None, None, None
counter = 0
CUR_BEST_MODEL_PATH = None
CHECKPOINT_MODEL_PATH = None
RES_FILE_PATH = None

""""
Functions for Bayesian optimization
"""
# Callback that saves the current best model
def customCallback(res):
    global counter
    global CUR_BEST_MODEL_PATH
    global CHECKPOINT_MODEL_PATH
    global RES_FILE_PATH

    x0 = res.x_iters  # List of input points
    y0 = res.func_vals  # Evaluation of input points
    print('Last eval: ', x0[-1],
          ' - Score ', y0[-1])
    print('Current iter: ', counter,
          ' - Score ', res.fun,
          ' - Args: ', res.x)

    if y0[-1] == res.fun:
        if os.path.exists(CUR_BEST_MODEL_PATH):
            os.remove(CUR_BEST_MODEL_PATH)

        os.rename(CHECKPOINT_MODEL_PATH, CUR_BEST_MODEL_PATH)

        res_file = open(RES_FILE_PATH, 'wb')
        pickle.dump(res, res_file)
        res_file.close()

    counter += 1


# Function passed to the bayesian opt algorithm that initializes and runs the (EEG) model
def f(x):
    print("Testing values", x)

    batch_size = x[0]
    epochs = x[1]
    l_r = x[2]
    warmup = x[3]
    patiente = x[4]
    model_size = x[5]

    global train_dataset
    global valid_dataset

    # inicializar modelo
    model = EEG_GPT2()
    model2, list_train, list_valid, loss = model.train(train_dataset, valid_dataset, model_size, batch_size=batch_size, epochs=epochs, warmup=warmup,
                                                       patience=patiente, lr=l_r)

    del model
    del model2
    del list_train
    del list_valid
    gc.collect()
    return loss

# Bayesian Optimization using skopt from scikit learning
def BayesOptimization(opt_checkpoint_path, space, calls=10):
    checkpoint_saver = CheckpointSaver(opt_checkpoint_path,
                                       compress=9)  # keyword arguments will be passed to `skopt.dump`
    if os.path.exists(opt_checkpoint_path):
        res = load(opt_checkpoint_path)
        x0 = res.x_iters
        y0 = res.func_vals

        if calls - len(x0) <= 0:
            print("Last checkpoint was complete.")
            return res

        res = gp_minimize(f, space, n_calls=calls - len(x0), callback=[checkpoint_saver, customCallback],
                          random_state=777, n_jobs=1, model_queue_size=1
                          , x0=x0, y0=y0)
    else:
        res = gp_minimize(f, space, n_calls=10, callback=[checkpoint_saver, customCallback],
                          # a list of callbacks including the checkpoint saver
                          random_state=777, n_jobs=1, model_queue_size=1)
    return res


"""
Splits the corpus into train, test and validation sets
"""
def splitCorpus(emosentencesevents_path, train_path, valid_path, test_path):

    EmoSentencesEvents = pd.read_pickle(emosentencesevents_path)

    if not os.path.exists(train_path) or not os.path.exists(valid_path) or not not os.path.exists(test_path):
        ## ========= Split into train and test =========
        train = EmoSentencesEvents.sample(frac=0.8, random_state=200)  # random state is a seed value
        test = EmoSentencesEvents.drop(train.index)
        valid = test.sample(frac=0.1, random_state=200)
        test = test.drop(valid.index)

        ## ========= Save train and test into files
        train.to_pickle(train_path)
        test.to_pickle(test_path)
        valid.to_pickle(valid_path)

    train_dataset = EventDataset(train_path)
    test_dataset = EventDataset(test_path)
    valid_dataset = EventDataset(valid_path)
    return train_dataset, valid_dataset, test_dataset

"""
Trains the model
"""
def train_model(model_name, space):
    opt_checkpoint_path = "EEG_Model/EEG_gpt2_files/trained_models/BayesOpt_Checkpoint_" + model_name + ".pkl"
    s_res = BayesOptimization(opt_checkpoint_path, space)
    return s_res

"""
Trains two models using Bayesian Opt
EEG-S -> gpt2 (smallest model available in hugging face)
EEG-L -> gpt2-large (largest gpt2 model that fitted in our GPU)
"""
def main():
    global CUR_BEST_MODEL_PATH
    global CHECKPOINT_MODEL_PATH
    global RES_FILE_PATH
    global train_dataset
    global valid_dataset
    global test_dataset

    # Split the dataset into train, test and valid sets and save them into pickles
    corpus_folder = "Dataset/SSE_dataset/EmoSentencesEvents/"
    corpus_file_path = "Dataset/SSE_dataset/EmoSentencesEvents/empdial_emo_senteces_events.p"
    train_path, valid_path, test_path = corpus_folder + 'train.p', corpus_folder + 'valid.p', corpus_folder + 'test.p'
    train_dataset, valid_dataset, test_dataset = splitCorpus(corpus_file_path, train_path, valid_path, test_path)

    # Define the folder where the files resultant from training (model checkpoints) go
    train_folder = 'EEG_Model/EEG_gpt2_files/trained_models/'


    # Define the model size and checkpoint name
    model_size = 'gpt2-large' # Options: gpt2, gpt2-medium, gpt2-large, gpt2-xl
    best_model_name = 'best_eeg_' + model_size

    # Define the search space for Bayesian Optimization
    space = [
        [32, 64, 128],     # Batch size
        [100],             # Number of epochs
        (1.0e-5, 1.0e-4),  # Learning rate
        (5000, 10000),     # Warmup
        (1, 5),            # Early Stopping Patience
        [model_size]       # Model size
    ]

    # Train the EEG-L
    print("********* Training and Hyperparameter Optimization (Bayesian Optimization) *********")
    print("Training model EEG-L")
    CUR_BEST_MODEL_PATH = train_folder + best_model_name + '.pt'
    CHECKPOINT_MODEL_PATH = train_folder + 'checkpoint.pt'
    # The res file contains the result object from bayesian optimization
    # and is used so we can stop and resume the optimization process
    RES_FILE_PATH = train_folder + 'res_file.p'

    # Start the training
    train_model(model_size, space)

    # Train EEG-S
    print("Training model EEG-S")
    # update the file paths
    model_size = 'gpt2' #Now we use the smaller model
    best_model_name = 'best_eeg_' + model_size
    CUR_BEST_MODEL_PATH = train_folder + best_model_name + '.pt'
    CHECKPOINT_MODEL_PATH = train_folder + 'checkpoint.pt'
    RES_FILE_PATH = train_folder + 'res_file.p'

    # Update the space
    space = [
        [32, 64, 128],  # Batch size
        [100],  # Number of epochs
        (1.0e-5, 1.0e-4),  # Learning rate
        (5000, 10000),  # Warmup
        (1, 5),  # Early Stopping Patience
        [model_size]  # Model size
    ]

    # Start the training
    train_model(model_size, space)


if __name__ == '__main__':
    main()
