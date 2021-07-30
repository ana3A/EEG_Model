import pickle
from EEG_Model.EEG_gpt2 import EEG_GPT2
from EEG_Model.EventDataset import EventDataset

def main():
    # Load dataset sets
    corpus_folder = "Dataset/SSE_dataset/EmoSentencesEvents/"
    train_path, test_path = corpus_folder + 'train.p', corpus_folder + 'test.p'
    train_set = EventDataset(train_path)
    test_set = EventDataset(test_path)

    # Define the folder where the files resultant from training (model checkpoints) go
    train_folder = 'EEG_Model/EEG_gpt2_files/trained_models/'

    """
    Evaluate the EEG-L model.
    This generates txt files containing the generated output
    """
    # Define the model size and checkpoint name
    model_size = 'gpt2-large'  # Options: gpt2, gpt2-medium, gpt2-large, gpt2-xl
    best_model_name = 'best_eeg_' + model_size
    # Path of the chcekpoint
    CUR_BEST_MODEL_PATH = train_folder + best_model_name + '.pt'
    # Define where the generated output goes
    out_test_file = "EEG_Model/EEG_gpt2_files/output/" + best_model_name + "_out.txt"

    print("********* Evaluate Model *********")
    model = EEG_GPT2()
    model.evaluate(CUR_BEST_MODEL_PATH, model_size, train_set, test_set, out_test_file)

    """
    Evaluate the EEG-S model.
    This generates txt files containing the generated output
    """
    # Define the model size and checkpoint name
    model_size = 'gpt2'  # Options: gpt2, gpt2-medium, gpt2-large, gpt2-xl
    best_model_name = 'best_eeg_' + model_size
    # Path of the checkpoint
    CUR_BEST_MODEL_PATH = train_folder + best_model_name + '.pt'
    # Define where the generated output goes
    out_test_file = "EEG_Model/EEG_gpt2_files/output/" + best_model_name + "_out.txt"

    print("********* Evaluate Model *********")
    model = EEG_GPT2()
    model.evaluate(CUR_BEST_MODEL_PATH, model_size, train_set, test_set, out_test_file)

if __name__ == '__main__':
    main()
