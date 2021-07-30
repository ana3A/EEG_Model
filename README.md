
# EEG

This is the official code used for our paper "EEG Model: Emotional Episode Generation for Social Sharing of Emotions":

*Social sharing of emotions (SSE) occurs when one communicates their feelings and reactions to a certain event in the course of a social interaction. The phenomenon is part of our social fabric and plays an important role in creating empathetic responses and establishing rapport. Intelligent social agents capable of SSE will have a mechanism to create and build long-term interaction with humans. In this paper, we present the Emotional Episode Generation (EEG) model, a fine-tuned GPT-2 model capable of generating emotional social talk regarding multiple event tuples in a human-like manner.
Human evaluation results show that the model successfully translates one or more event-tuples into emotional episodes, reaching quality levels close to human performance. Furthermore, the model clearly expresses one emotion in each episode as well as humans. To train this model we used a public dataset and built upon it using event extraction techniques.*

## Code Overview
The project was tested in **Python 3.8** and is composed by three main parts:
* Dataset Folder
* EEG_Model Folder
* Eventify Folder

### Dataset Folder
This folder is used to store all dataset/corpora related data. The folder *ED_dataset* contains the *Empathetic Dialogue* data, the folder *SSE_dataset* is used to store the corpora created to train the neural models. Lastly, the *Utils* folder contians a simple text cleaner.

**Note:** The *Empathetic Dialogue* data files are already included in the project. However, the files can be downloaded by following the instructions available on the official dataset github: https://github.com/facebookresearch/EmpatheticDialogues

### EEG_Model Folder
Contains and stores any code or data related to the EEG Neural model (i.e. code, output files and saved models). The model code is defined in *EEG_gpt2.py* file, using pytorch. The file *EventDataset.py* defines a custom dataset class so that we can use our corpus with pytorch and *EarlyStopping.py* contains the class that deals with the Early Stopping process during training. Lastly, all related data us sotred in the *EEG_gpt2_files* folder.

### Eventify Folder
Contains the code (*PredPatt_Eventify.py*) used to extract events from *Empathetic Dialogues* as described in the paper.

## Usage
The project was tested using Python 3.8. All requiered packages are stored in the *requirements.txt* file and must be installed before running the code. The steps to run the code are the following.

#### 1. Install the required packages
```
pip install -r requirements.txt
```
#### 2. Create the corpus
```
python create_corpus.py
```
#### 3. Train the EEG models
```
python train_eeg.py
```
#### 4. Evaluate the models
```
python test_eeg.py
```


