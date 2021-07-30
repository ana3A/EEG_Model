# -*- coding: utf-8 -*-
"""Transformer_test.ipynb
Fine Tuning GTP-2 for event to sentence tranlsation.
This notebook was created as a test to see if it is possible to re-use an already existant transformer model for generating sentences.
Heavly inspired by: https://towardsdatascience.com/teaching-gpt-2-a-sense-of-humor-fine-tuning-large-transformer-models-on-a-single-gpu-in-pytorch-59e8cec40912
The model used is the GPT-2 model. (State-of-the-art for text completion?)
"""

import os
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import torch

from EEG_Model.EarlyStopping import EarlyStopping

logging.getLogger().setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings('ignore')

class EEG_GPT2:
    def __init__(self):
        #Device
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            print('CUDA Available')
            self.device = torch.device('cuda')
        else:
            print('CPU Device')

        #Special tokens
        self.end_of_text_token = "<|endoftext|>"    # end token recognised by GPT-2 - allows the model to work with multiple sequence lengths
        self.begin_of_text_token = "<|beginoftext|>"  # NOT RECOGNIZED BY GPT-2 - although not recognized, it eventually signals the model that a new sequence has started
        self.to_sent_token = "=="                     # NOT RECOGNIZED BY GPT-2 - signals that event sequence is over and the sentence should begin
        self.event_emotion_token = '::'               # signals that the emotion correspondent to the next event

    # Function to first select topN tokens from the probability list and then based on the selected N word distribution
    # get random token ID
    # Chooses the next token predicted by the model
    def choose_from_top(self, probs, n=5):
        ind = np.argpartition(probs, -n)[-n:]
        top_prob = probs[ind]
        top_prob = top_prob / np.sum(top_prob)  # Normalize
        choice = np.random.choice(n, 1, p=top_prob)
        token_id = ind[choice][0]
        return int(token_id)


    def train(self, train_dataset, valid_dataset, model_size, batch_size = 16, epochs = 50, lr = 3e-5, warmup = 5000, patience=20): #largest input from the test set
        """Model Parameters"""
        BATCH_SIZE = batch_size
        EPOCHS = epochs
        LEARNING_RATE = lr
        WARMUP_STEPS = warmup
        
        """Tracking Metrics"""
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        """Regularization: Early Stopping"""
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0.3)

        """Load Model"""
        model, tokenizer = self.loadModel(model_size)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=-1)

        """Load Dataset Loaders"""
        train_dataset_loader = DataLoader(train_dataset)
        valid_dataset_loader = DataLoader(valid_dataset)
        
        
        proc_seq_count = 0
        batch_count = 0

        tmp_data_tens = None
        data_not_added = []
        valid_loss = 0

        """Find Input size
            Input size is the size of the biggest input present in the test set
        """
        MAX_SEQ_LEN = -1
        for idx, data in enumerate(train_dataset_loader):
            data_to_encode = self.structureData(data)
            data_tens = torch.tensor(tokenizer.encode(data_to_encode)).unsqueeze(0).to(self.device)
            size = data_tens.shape[1]
            if size > MAX_SEQ_LEN:
                MAX_SEQ_LEN = size

        """Model Training"""
        print("Training Model. . .")
        for epoch in tqdm(range(EPOCHS)):
            model.train() # model in training mode

            for idx, data in enumerate(train_dataset_loader):
                data_to_encode = self.structureData(data)
                #################### "Fit as many sequences into MAX_SEQ_LEN sequence as possible" logic start ####
                data_tens = torch.tensor(tokenizer.encode(data_to_encode)).unsqueeze(0).to(self.device)
                # Skip sample from dataset if it is longer than MAX_SEQ_LEN
                if data_tens.size()[1] > MAX_SEQ_LEN:
                    print("Not appended:", data_to_encode)
                    data_not_added.append(data_to_encode)
                    continue

                # The first joke sequence in the sequence
                if not torch.is_tensor(tmp_data_tens):
                    tmp_data_tens = data_tens
                    continue
                else:
                    # The next joke does not fit in so we process the sequence and leave the last joke
                    # as the start for next sequence
                    if tmp_data_tens.size()[1] + data_tens.size()[1] > MAX_SEQ_LEN:
                        work_data_tens = tmp_data_tens
                        tmp_data_tens = data_tens
                    else:
                        # Add the joke to sequence, continue and try to add more
                        tmp_data_tens = torch.cat([tmp_data_tens, data_tens[:, 1:]], dim=1)
                        continue
                ################## Sequence ready, process it trough the model ##################

                outputs = model(work_data_tens, labels=work_data_tens)
                loss, logits = outputs[:2]
                loss.backward()

                train_losses.append(loss.item())

                proc_seq_count += 1
                if proc_seq_count == BATCH_SIZE:
                    proc_seq_count = 0
                    batch_count += 1
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    model.zero_grad()


            ######################
            # validate the model #
            ######################
            model.eval()  # prep model for evaluation
            for idx, data in enumerate(valid_dataset_loader):
                data_to_encode = self.structureData(data)
                #################### "Fit as many sequences into MAX_SEQ_LEN sequence as possible" logic start ####
                data_tens = torch.tensor(tokenizer.encode(data_to_encode)).unsqueeze(0).to(self.device)
                # Skip sample from dataset if it is longer than MAX_SEQ_LEN
                if data_tens.size()[1] > MAX_SEQ_LEN:
                    print("Not appended:", data_to_encode)
                    data_not_added.append(data_to_encode)
                    continue

                # The first joke sequence in the sequence
                if not torch.is_tensor(tmp_data_tens):
                    tmp_data_tens = data_tens
                    continue
                else:
                    # The next joke does not fit in so we process the sequence and leave the last joke
                    # as the start for next sequence
                    if tmp_data_tens.size()[1] + data_tens.size()[1] > MAX_SEQ_LEN:
                        work_data_tens = tmp_data_tens
                        tmp_data_tens = data_tens
                    else:
                        # Add the joke to sequence, continue and try to add more
                        tmp_data_tens = torch.cat([tmp_data_tens, data_tens[:, 1:]], dim=1)
                        continue
                ################## Sequence ready, process it trough the model ##################

                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = model(data_tens, labels=data_tens)
                # calculate the loss
                loss, logits = outputs[:2]
                # record validation loss
                valid_losses.append(loss.item())

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            n_epochs = EPOCHS
            epoch_len = len(str(n_epochs))

            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')

            print(print_msg)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                break

        return model, avg_train_losses, avg_valid_losses, valid_loss

    def loadModel(self, model_size):
        # Because we want to perform text generation we'll use GPT2LMHeadModel.
        print('Load tokenizer')
        tokenizer = GPT2Tokenizer.from_pretrained(model_size)  # smallest model because our dataset is small
        print('Load model', model_size)
        model = GPT2LMHeadModel.from_pretrained(model_size)
        print('Map to device', self.device)
        model = model.to(self.device)
        print('Model loaded.')

        return model, tokenizer

    def evaluate(self, saved_model, model_size, train_dataset, test_dataset, out_file): #, model_numbers, max_seq = 1011):
        train_dataset_loader = DataLoader(train_dataset)
        test_dataset_loader = DataLoader(test_dataset)
        model, tokenizer = self.loadModel(model_size)

        """
        Find Input size
        Input size is the size of the biggest input present in the test set
        """
        print("Finding Input size")
        MAX_SEQ_LEN = -1
        for idx, data in enumerate(train_dataset_loader):
            data_to_encode = self.structureData(data)
            data_tens = torch.tensor(tokenizer.encode(data_to_encode)).unsqueeze(0).to(self.device)
            size = data_tens.shape[1]
            if size > MAX_SEQ_LEN:
                MAX_SEQ_LEN = size
        print("Input size found")
        
        # MAX_SEQ_LEN = max_seq
        print("Evaluating")

        #Load model checkpoint
        model_path = os.path.join(saved_model)
        torch_model = torch.load(model_path, map_location=torch.device(self.device))
        model.load_state_dict(torch_model)
        #Put model on eval mode
        model.eval()

        #Generate outputs
        sentence_num = 0
        test_losses = []
        with torch.no_grad():
            for idx, data in tqdm(enumerate(test_dataset_loader)):
                sentence_finished = False
                model_input = self.structureEvalData(data)

                cur_ids = torch.tensor(tokenizer.encode(model_input)).unsqueeze(0).to(self.device)

                if cur_ids.size()[1] > MAX_SEQ_LEN:
                    continue

                while not sentence_finished:
                    outputs = model(cur_ids, labels=cur_ids)
                    loss, logits = outputs[:2]

                    test_losses.append(loss.item())

                    softmax_logits = torch.softmax(logits[0, -1], dim=0)  # Take the first(from only one in this case) batch and the last predicted embedding
                    next_token_id = self.choose_from_top(softmax_logits.to('cpu').numpy(), n=5)  # Randomly(from the topN probability distribution) select the next word
                    cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(self.device) * next_token_id],
                                        dim=1)  # Add the last word to the running sequence

                    if cur_ids.size()[1] > MAX_SEQ_LEN:
                        sentence_finished = True

                    if next_token_id in tokenizer.encode(self.end_of_text_token):
                        sentence_finished = True

                sentence_num += 1

                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)

                test_loss = np.average(test_losses)

                with open(out_file, 'a') as f:
                    f.write(f"{output_text}\n")
                    f.write(f"{test_loss}\n")
            f.close()

    def structureData(self, data):
        emotion = data[0][0]
        events = data[1][0]
        sentence = data[2][0]

        #print(emotion, sentence, events)
        res = self.begin_of_text_token
        res += emotion + self.event_emotion_token
        res += ''.join(events)
        res += self.to_sent_token + sentence
        res += self.end_of_text_token

        return res

    def structureEvalData(self, data):
        emotion = data[0][0]
        events = data[1][0]
        #sentence = data[2][0]

        # print(emotion, sentence, events)
        res = self.begin_of_text_token
        res += emotion + self.event_emotion_token
        res += ''.join(events)
        res += self.to_sent_token #+ sentence
        #res += self.end_of_text_token

        return res
