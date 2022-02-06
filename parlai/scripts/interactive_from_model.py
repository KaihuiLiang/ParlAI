#!/usr/bin/env python3

import torch
from parlai.core.agents import create_agent_from_model_file


class RawBlender():
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_num = 1

    def load_model(self):
        if self.model == None:
            opt_overrides = {}
            # self.gpu_num = gpu_num
            opt_overrides['gpu'] = self.gpu_num
            opt_overrides['datatype'] = 'test'
            # opt_overrides['datapath'] = '/home/cafe7/ParlAI/experiments/physical_activity/model/blender_3B_031501'

            opt_overrides['inference'] = 'nucleus'
            opt_overrides['skip_generation'] = False

            self.model = create_agent_from_model_file(self.model_checkpoint, opt_overrides=opt_overrides)
            print("model", self.model)
            print("load Raw Blender model from:{}".format(self.model_checkpoint))
            print("allocate Raw Blender model to gpu_{}".format(self.gpu_num))

    def _build_up_model_input(self, history, user_text):
        prev_input = ""
        for turn_text in history:
            prev_input += turn_text
        if prev_input:
            prev_input = prev_input + " " + user_text
        else:
            prev_input = user_text
        text = prev_input
        text = text.lower()
        return text

    def process(self, user_text):
        if not user_text:
            user_text = " [SEP] "
        torch.cuda.set_device(self.gpu_num)
        self.model.reset()
        # inputs = self._build_up_model_input(history, user_text)
        # print("input to the raw blender:{}".format(inputs))
        self.model.observe({'text': user_text, 'episode_done': False})
        output = self.model.act()
        if output is not None:
            return output['text']
        else:
            return "Raw Blender SYSTEM ERROR!"


if __name__ == '__main__':
    agent = RawBlender('/home/cafe7/ParlAI/experiments/physical_activity/model/blender_3B_031501')
    agent.load_model()
    result = agent.process("Hi")
    print(result)