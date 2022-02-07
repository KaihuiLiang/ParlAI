import os
import json
import pandas as pd
from tqdm import tqdm

from parlai.core.params import ParlaiParser
from parlai.core.dict import DictionaryAgent, TokenizationMode
from parlai.core.opt import Opt

from parlai.utils.io import PathManager
from parlai.core.teachers import DialogTeacher


class DefaultTeacher(DialogTeacher):
    END = '__end__'

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
        truncate_size = 128

        with PathManager.open(path) as data_file:
            self.sessions_df = pd.read_csv(data_file)

        self.sessions_group = self.sessions_df.groupby('document')


        # Try to read just a portion of data

        # for i in range(10):
        #     if i %2 == 0:
        #         new_episode = False
        #     else:
        #         new_episode = True
        #     yield {"text": "Hi", "labels": ["Hello"]}, new_episode


        group_count = 0
        for group_name, group_df in tqdm(self.sessions_group):
            group_count += 1
            # if group_count > 4:
            #     break
            print("group_name", group_name)
            # yield {"text": "Hi", "labels": ["Hello"]}, True

            current_document = None
            current_speaker = None
            context = str()
            bot_utterance = ""
            context_speaker_2 = ""

            for row_index, row in tqdm(group_df.iterrows()):
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
                        # context += "</s> " + context_speaker_2
                        context += " " + context_speaker_2

                        context = " ".join(context.split()[-truncate_size:])
                    # print("[context]", context)

                    if not bot_utterance or len(bot_utterance) == 0:
                        # no bot utterance at last turn
                        break

                    truncated_text = self.truncate_text_and_prepend_domain(context, row["domain"])
                    # print("Truncated tokens", truncated_text)

                    # print("====yield====\n", {"text": truncated_text, "labels": [bot_utterance]}, new_episode)
                    yield {"text": truncated_text, "labels": [bot_utterance]}, True

                    # context += " </s> " + bot_utterance
                    context += " " + bot_utterance
                    # print("[context + bot utterance]", context)

                    bot_utterance = ""
                    context_speaker_2 = ""

    def get_dict_agent(self):
        BYTELEVEL_BPE_VOCAB = "data/models/blender/blender_3B/model.dict-vocab.json"
        BYTELEVEL_BPE_MERGE = "data/models/blender/blender_3B/model.dict-merges.txt"

        parser = ParlaiParser()
        parser.set_params(
            dict_tokenizer='bytelevelbpe',
            bpe_vocab=BYTELEVEL_BPE_VOCAB,
            bpe_merge=BYTELEVEL_BPE_MERGE,
        )
        opt = parser.parse_args([])
        agent = DictionaryAgent(opt)
        return agent

    def tokenize(self, text):
        agent = self.get_dict_agent()
        tokens = agent.bytelevelbpe_tokenize(text)
        return tokens


    def truncate_text_and_prepend_domain(self, text, domain):
        strategy_tokens = self.tokenize("[" + domain + "]")
        strategy_token_lens = len(strategy_tokens)

        agent = self.get_dict_agent()

        text_tokens = self.tokenize(text)
        truncat = 126 - strategy_token_lens
        text_tokens = text_tokens[-truncat:]

        if text_tokens:
            while ord(text_tokens[0][0]) != 288 and text_tokens: # make sure first token starts with G
                text_tokens = text_tokens[1:]
        text = agent.vec2txt([agent.tok2ind[w] for w in text_tokens])
        return "[" + domain + "]" + " " + text
