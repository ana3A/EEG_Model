import sys
import os
import re

import spacy
import pandas as pd
from predpatt.util.ud import dep_v2
from tqdm import tqdm

from Dataset.Utils.DataCleaner import DataCleaner
from Eventify.Predpatt_Eventify import Eventify

"""
Auxiliary initial step. Extracts the situations, speaker utterances and emotions from the original dataset (ED)
"""
def extract_episodes_emotion_tags(ed_path, save_path):
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_lg")

    # Extract Situations, Speaker Utterances and Emotion labels
    emotional_episodes, emotions = read_dataset(ed_path)

    # Filter Speaker Utterances containing "?" or without verbs in the past tense
    emotional_episodes, emotions = filter_sentences(emotional_episodes, emotions, nlp)

    # Mild cleaning
    # Sentences are not lowercased because SRL model were trained com cased data
    # We also solve coreference in this step to help in the event extracting process
    emotional_episodes, emotions, coref_emotional_episodes = clean_sentences(emotional_episodes, emotions, nlp)

    # Save dataframe
    df = pd.DataFrame(list(zip(emotional_episodes, emotions, coref_emotional_episodes)),
                      columns=["Sentences", "Emotion", "CorefSentences"])

    df.to_csv(save_path)
    return df

"""
Uses PredPatt to extract events
"""
def extract_events(emo_sentences_corpus, save_to_path, resolve_relcl, resolve_appos, resolve_amod, resolve_conj,
                 resolve_poss, simple, cut, big_args, strip, ud, mode):
    eventifier = Eventify(resolve_relcl=resolve_relcl, resolve_appos=resolve_appos, resolve_amod=resolve_amod, resolve_conj=resolve_conj,
                 resolve_poss=resolve_poss, simple=simple, cut=cut, big_args=big_args, strip=strip, ud=ud, mode=mode)
    df_final = pd.DataFrame(columns=["Sentences", "Emotion", "Events"])

    sentences = emo_sentences_corpus['Sentences'].to_list()
    emotions = emo_sentences_corpus['Emotion'].to_list()
    coref_sentences = emo_sentences_corpus['CorefSentences'].to_list()

    for i, sentence in tqdm(enumerate(sentences)):
        events = eventifier.eventify(coref_sentences[i])

        if events:
            lst = [[sentences[i], emotions[i], events]]
            df = pd.DataFrame(lst, columns=["Sentences", "Emotion", "Events"])
            df_final = df_final.append(df, ignore_index=True)

            if i % 10 == 0:
                df_final.to_pickle(save_to_path)

    df_final.to_pickle(save_to_path)
    return df_final

""" 
Reader for the Empathetid Dialogues Dataset 
Reads Situations, Speaker Utterances and Emotion labels
"""
def read_dataset(path):
    sentences_set = set()
    sentences = []
    emotions = []
    index = -1
    splitnames = ['train', 'valid', 'test']
    for split in tqdm(splitnames):
        df = open(os.path.join(path, f"{split}.csv"), encoding="utf8").readlines()
        lines = df[1:]
        last_seen = None

        for line in lines:
            items = line.split(",")
            items[3] = re.sub("_comma_", ",", items[3])
            items[5] = re.sub("_comma_", ",", items[5])

            if last_seen is None or last_seen != items[3]:
                last_seen = items[3]
                index = 0

            else:
                index += 1

            # Speaker sentences are pair
            if index % 2 == 0:
                uts = [items[3], items[5]]
            else:
                continue

            emo = items[2].lower()
            for sentence in uts:
                if sentence in sentences_set:
                    continue
                emotions.append(emo)
                sentences.append(sentence)
                sentences_set.add(sentence)

    return sentences, emotions

""" 
Filters non-wanted utterances and converts the emotions to the Ekman model 
"""
def filter_sentences(sentences, emotions, nlp):
    filtered_sentences = []
    ekman_emotions = []
    for i, sentence in tqdm(enumerate(sentences)):
        doc = nlp(sentence)
        past_tense_present = False
        for token in doc:
            if token.text == '?':
                past_tense_present = False
                break
            if token.tag_ in ['VBD', 'VBN']:
                past_tense_present = True

        if past_tense_present:

            if emotions[i] == "surprised":
                ekman_emotions.append("surprise")

            if emotions[i] in ["joyful", "excited", "proud", "grateful", "impressed", "hopeful", "confident",
                               "anticipating", "nostalgic", "prepared", "content", "trusting"]:
                ekman_emotions.append("happiness")

            elif emotions[i] in ["afraid", "terrified", "anxious", "apprehensive"]:
                ekman_emotions.append("fear")

            elif emotions[i] == "angry" or emotions[i] == "furious":
                ekman_emotions.append("anger")

            elif emotions[i] in ["sad", "devastated", "annoyed", "lonely", "guilty", "disappointed", "jealous",
                                 "embarrassed", "ashamed"]:
                ekman_emotions.append("sadness")

            elif emotions[i] == "disgusted":
                ekman_emotions.append("disgust")

            else:
                continue

            filtered_sentences.append(sentence)

    return filtered_sentences, ekman_emotions

""" 
Mild cleaning
"""
def clean_sentences(sentences, emotions, nlp):
    clean_sentences = []
    clean_sentences_emotions = []
    coref_sentences = []
    dc = DataCleaner()

    for i, sent in tqdm(enumerate(sentences)):
        s = sent
        s = dc.removeTextBetweenBrackets(s)
        s = dc.removeNonAscii(s)
        s = dc.removeExtraSpaces(s)
        s = dc.replaceParentheses(s)
        s = dc.expandContractions(s)

        if s == '':
            continue

        coref_s = dc.solveCoRef(s, nlp)
        clean_sentences.append(s)
        coref_sentences.append(coref_s)
        clean_sentences_emotions.append(emotions[i])

    return clean_sentences, clean_sentences_emotions, coref_sentences,

"""
Functions to present Dataset statistic information (number of events, number of elements in the dataset, etc.)
"""
def get_corpus_stats(df):
    labels_list = df['Emotion'].unique().tolist()  # all csv files were created using the isear labels
    n_labels = len(labels_list)
    n_rows = df.shape[0]

    print("Corpus Stats:")
    print("Number of rows:", n_rows)
    print("Number of emotion labels:", n_labels)
    print("Emotion labels:", labels_list)

    n_sentences = 0
    for label in labels_list:
        label_sent = df[df.Emotion == label]
        n_sentences += label_sent.shape[0]
        print("Sentences with emo", label, ":", n_sentences)

    total_events = 0
    total_ne_targets = 0
    total_ne_subs = 0

    if 'Events' in df:
        events = df["Events"].tolist()
        for sentence_events in events:
            n_events, n_non_empty_targets, n_non_empty_subs = count_events(sentence_events)

            total_events += n_events
            total_ne_targets += n_non_empty_targets
            total_ne_subs += n_non_empty_subs

        if total_events > 0:
            print('Total events:', total_events)
            print('Events per Sentence on average:', total_events / n_rows)
            print("Non empty targets: ", total_ne_targets)
            print("Non empty agents: ", total_ne_subs)
            print("######### END #########\n")

def count_events(events):
    n_events = 0
    n_ne_targets = 0
    n_ne_subs = 0

    for event in events:
        t_events, t_targets, t_subs = count_event(event)
        n_events += t_events
        n_ne_targets += t_targets
        n_ne_subs += t_subs

    return n_events, n_ne_targets, n_ne_subs

def count_event(event):
    n_events = 1
    n_ne_targets = 0
    n_ne_subjs = 0

    subj = event[0]
    if '<|EMPTY|>' in subj:
        return n_events, n_ne_targets, n_ne_subjs

    else:
        n_ne_subjs += 1

    elem = event[2]
    if isinstance(elem, str):
        if '<|EMPTY|>' in elem:
            return n_events, n_ne_targets, n_ne_subjs

        n_ne_targets += 1

    elif isinstance(elem, tuple):
        n_e, n_t, n_s = count_event(elem)
        n_events += n_e
        n_ne_targets += n_t
        n_ne_subjs += n_s

    return n_events, n_ne_targets, n_ne_subjs

def main():
    ed_path = "Dataset/ED_dataset"
    empdial_info_save_path = "Dataset/SSE_dataset/EmoSentences/empdial_emo_senteces.csv"
    corpus_save_path = "Dataset/SSE_dataset/EmoSentencesEvents/empdial_emo_senteces_events.p"
    empdial_info_df = extract_episodes_emotion_tags(ed_path, empdial_info_save_path)

    """ The last arguments are PredPatt Options. These are the values used in our work. """
    corpus_df = extract_events(empdial_info_df, corpus_save_path, resolve_relcl=False, resolve_appos=False, resolve_amod=False, resolve_conj=True,
                 resolve_poss=False, simple=True, cut=True, big_args=False, strip=True, ud=dep_v2.VERSION, mode="token")

    get_corpus_stats(corpus_df)

if __name__ == '__main__':
    main()