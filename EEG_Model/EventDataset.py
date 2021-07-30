from torch.utils.data import Dataset
import pandas as pd

class EventDataset(Dataset):
    def __init__(self, dataset_path): #, begin_of_text_token = "<|beginoftext|>", end_of_text_token = "<|endoftext|>"):
        super().__init__()

        self.sentence_list = []
        self.events_list = []
        self.emotion_list = []

        df = pd.read_pickle(dataset_path)
        dataset_list = df.values.tolist()

        for elem in dataset_list:
            self.emotion_list.append(elem[1])
            self.sentence_list.append(elem[0])

            s_event = []
            for event in elem[2]:
                e_str = self.getEventStr(event)
                s_event.append(e_str)

            self.events_list.append(s_event)


    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, item):
        return self.emotion_list[item], self.events_list[item], self.sentence_list[item]

    def getEventStr(self, event):
        agent_str = event[0]
        action_str = event[1]
        target = event[2]

        if isinstance(target, tuple):
            e = self.getEventStr(target)
            t_str = e
        else:
            t_str = target

        return '(' + agent_str + ',' + action_str + ',' + t_str + ')'