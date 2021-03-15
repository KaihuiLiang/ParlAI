import os
import json
import pandas as pd

from parlai.utils.io import PathManager
from parlai.core.teachers import DialogTeacher


class DefaultTeacher(DialogTeacher):

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        # suffix = 'train' if opt['datatype'].startswith('train') else 'dev'
        folder = 'unannotated'

        if 'train' in self.datatype:
            file = 'train.csv'
        else:
            file = 'valid.csv'


        opt['datafile'] = os.path.join(opt['datapath'], 'PhysicalActivity', folder, file)
        self.id = 'trainer'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with PathManager.open(path) as data_file:
            self.sessions_df = pd.read_csv(data_file)

        self.sessions_group = self.sessions_df.groupby('document')

        # for i in range(10):
        #     if i %2 == 0:
        #         new_episode = False
        #     else:
        #         new_episode = True
        #     yield {"text": "Hi", "labels": ["Hello"]}, new_episode

        for group_name, group_df in self.sessions_group:
            print("group_name", group_name)
            # yield {"text": "Hi", "labels": ["Hello"]}, True

            current_document = None
            current_speaker = None
            context = str()
            bot_utterance = ""
            context_speaker_2 = ""

            for row_index, row in group_df.iterrows():
                # print("[row_index]", row_index)

                new_episode = True if row['document'] != current_document else False
                if new_episode:
                    context = str()
                    print("reset context")
                    context_speaker_2 = str()

                # print("new_episode", new_episode)

                current_document = row['document']

                prev_speaker = current_speaker
                current_speaker = row['speaker']

                if current_speaker == 1:
                    bot_utterance += row['utterance'] + " "
                    # print("[bot_utterance]", bot_utterance)


                if current_speaker == 2:
                    context_speaker_2 += row['utterance'] + " "

                # [Start]: first row in document
                if row_index == 0 or self.sessions_df.iloc[row_index - 1]["document"] != group_name:
                    context_speaker_2 = "[START]"

                # Last
                if (current_speaker and row_index == len(self.sessions_df) - 1) or \
                    (current_speaker == 1 and self.sessions_df.iloc[row_index + 1]["document"] != group_name) or \
                    (current_speaker == 1 and self.sessions_df.iloc[row_index + 1]["speaker"] != 1):
                    # just finished reading counseler's side, yield
                    if context_speaker_2 == "[START]":
                        context = context_speaker_2
                    elif context_speaker_2:
                        context += "</s> " + context_speaker_2
                        context = " ".join(context.split()[-128:])
                    # print("[context]", context)

                    print("====yield====\n", {"text": context, "labels": [bot_utterance]}, new_episode)
                    if not bot_utterance or len(bot_utterance) == 0:
                        raise Exception("empty bot utterance!")

                    yield {"text": context, "labels": [bot_utterance]}, True

                    context += " </s> " + bot_utterance
                    # print("[context + bot utterance]", context)

                    bot_utterance = ""
                    context_speaker_2 = ""

