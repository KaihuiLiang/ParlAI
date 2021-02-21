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
        opt['datafile'] = os.path.join(opt['datapath'], 'PhysicalActivity', folder, '107_transcripts_utterance_only.csv')  # todo
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
                if row_index > 50:
                    return

                new_episode = True if row['document'] != current_document else False
                # print("prev document", current_document)
                # print("current document", row['document'])
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
                    print("[bot_utterance]", bot_utterance)

                # print("[current utterance]", row['utterance'])

                if current_speaker == 2:
                    context_speaker_2 += row['utterance'] + " "
                    # print("context_speaker_2", context_speaker_2)
                # print("current_speaker", current_speaker)
                # print("prev_speaker", prev_speaker)

                # context += group_df.loc[row_index - 1]['utterance']

                # if current_speaker == 1 and prev_speaker != 1:
                #     if new_episode:
                #         context = ""
                #     else:
                #         context = group_df.loc[row_index - 1]['utterance']
                #     print("[context]", context)
                #     # bot_utterance += str(row['utterance']) + " "

                # print("last_index_in_group", group_df.iloc[-1].name)
                if (current_speaker and row_index == len(self.sessions_df) - 1) or \
                    (current_speaker == 1 and self.sessions_df.iloc[row_index + 1]["document"] != group_name) or \
                    (current_speaker == 1 and self.sessions_df.iloc[row_index + 1]["speaker"] != 1):
                        # or row_index == self.sessions_df.iloc[-1].index:
                    # just finished reading counseler's side, yield
                    # yield (context, current_utterance), new_episode
                    if context_speaker_2:
                        context += "</s> " + context_speaker_2
                        context = " ".join(context.split()[-128:])
                    # print("[context]", context)
                    print("====yield====\n", {"text": context, "labels": [bot_utterance]}, new_episode)
                    if not bot_utterance:
                        raise Exception("empty bot utterance!")
                    yield {"text": context, "labels": [bot_utterance]}, True

                    context += " </s> " + bot_utterance
                    # print("[context + bot utterance]", context)

                    bot_utterance = ""
                    context_speaker_2 = ""

