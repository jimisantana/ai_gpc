
import re
from os import system, path
from threading import Thread
import pandas as pd
import numpy as np
from traceback import format_exc
from datetime import datetime

import pickle
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

PATH_TMP = './files_temp'
COL_TEXT_SEP = '__sys_gpc_text_sep__'
PAD_MAX_TOKENS = 25
BATCH_SIZE_AKA_MAX_ROWS_PER_GUESS_TO_FIT_GPU_MEM = int(4e3)
# BATCH_SIZE_AKA_MAX_ROWS_PER_GUESS_TO_FIT_GPU_MEM = int(1e3)  # small batches
PRINT_EVERY_N = int(1e4)
# PRINT_EVERY_N = int(1)  # print every time

path2here = '.'
dict_label_iid_pkl = f'{path2here}/model_save/dict_label_iid.pkl'
dict_label_t_pkl = f'{path2here}/model_save/dict_label_t.pkl'

dict_label_iid: dict = None
dict_label_t: dict = None
tokenizer: BertTokenizer = None
model: BertForSequenceClassification = None

# endregion globals

"""
originating from this tute - the test/eval inference part
http://mccormickml.com/2019/07/22/BERT-fine-tuning/
"""

# region cuda
# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

    # TODO - find out how to use all GPUs

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
# endregion cuda

# region load model
def load_model_n_stuff():

    print('ai_gpc loading model and tokenizer...')

    global dict_label_iid, dict_label_t
    global tokenizer, model

    if path.exists(dict_label_iid_pkl):
        with open(dict_label_iid_pkl, 'rb') as f:
            dict_label_iid = pickle.load(f)

    if path.exists(dict_label_t_pkl):
        with open(dict_label_t_pkl, 'rb') as f:
            dict_label_t = pickle.load(f)

    tokenizer = BertTokenizer.from_pretrained(f'{path2here}/model_save')
    model = BertForSequenceClassification.from_pretrained(f'{path2here}/model_save')


    print('setting model to', device)
    model.to(device)  # GPU or CPU
    model.cuda()
    model.eval()

    # if is_cuda:
    #     model.cuda()


load_model_n_stuff()
# endregion load model

r_dang_chars = re.compile(r'[{}"]+')  # dang stay for dangerous?


# region  M A I N  MAIN function to be called from flask
def gpc(name: str = 'file_name_unique_without_extension', top_cat: int = -1, top_cat_t: str = None):

    def fn(name2: str = 'file_name_unique_without_extension'):

        try:
            system(f'cd {PATH_TMP}; tar -zxvf {name2}.feather.tar.gz')

            ppath = f'{PATH_TMP}/{name2}.feather'

            df = pd.read_feather(ppath)
            print(f'original len {len(df)} titles')
            if len(df) > int(2e5):
                df = df.sample(n=int(2e5))
            print(f'doing inference on {len(df)} titles')

            with Gpc(df, top_cat, top_cat_t) as obj:
                df = obj.prepare_and_guess()
                obj.dump('end gpc instance... - we should be DONE ... maybe')

            print('end gpc static... - we should be DONE')

        except:

            err = format_exc()
            print(err)

    # async
    t = Thread(target=fn, args=(name,))
    t.start()

    # TODO TEMP DEBUG
    t.join()

# endregion  M A I N  MAIN function to be called from flask


# =============
# MAIN method in Gpc class is: prepare_and_guess()
# =============
class Gpc:

    def __init__(self, df: pd.DataFrame = None, top_cat: int = -1, top_cat_t: str = None):
        super().__init__()

        self.df = df
        self.top_cat = top_cat
        self.top_cat_t = top_cat_t

        self.column = COL_TEXT_SEP

        self.input_ids_test = []
        self.labels_test = []
        self.attention_masks_test = []
        self.texts_test = []
        self.test_dataloader: DataLoader = None

        self.d: datetime = datetime.now()

    def __del__(self):
        try:
            del self.df
            del self.input_ids_test
            del self.labels_test
            del self.attention_masks_test
            del self.texts_test
            del self.test_dataloader
        except:
            format_exc()

    def __enter__(self):
        return self

    def __exit__(self, ttype, value, traceback):
        self.__del__()

    # =============
    # MAIN
    # =============
    def prepare_and_guess(self) -> pd.DataFrame:

        self.texts_test = self.df[self.column].tolist()
        self.labels_test = [0] * len(self.texts_test)  # dummy
        self.input_ids_test, self.attention_masks_test, self.labels_test = self.encode_stuff()

        test_dataset = TensorDataset(self.input_ids_test, self.attention_masks_test, self.labels_test)
        self.test_dataloader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            # batch_size=len(test_dataset)  # AKA - single batch - nope! no mem for that
            batch_size=BATCH_SIZE_AKA_MAX_ROWS_PER_GUESS_TO_FIT_GPU_MEM,
            # tests
            num_workers=8,
            # maybe this is the culprit as suggested by user12750353 in stackoverflow
            # pin_memory=True
            pin_memory=False
        )

        # =======
        # call MAIN - that's what we are here for - the main GPU thing
        # =======
        # self.dump('start predictions...')
        predictions = self.guess()
        # self.dump('end predictions...')

        print('pytorch tensor shape is', predictions.shape)
        label_indices = torch.argmax(predictions, dim=1)

        self.dump('start loop df append...')

        df = []
        for i, o in enumerate(self.texts_test):

            # t, iid, s = self.find_with_top_level(predictions[i])

            label_index = label_indices[i]
            t = dict_label_t.get(label_index)
            # s = predictions[label_index]  # A! A! A! getting a number from the GPU to real CPU word is a shit load of time! Nope!

            # df.append(
            #     {
            #         'text': o,
            #         't': dict_label_t.get(label_index),
            #         'iid': dict_label_iid.get(label_index),
            #         's': predictions[label_index]
            #     }
            # )

        self.dump('end loop df append...')

        self.dump('start df...')
        df = pd.DataFrame(df)
        self.dump('end df...')
        return df

    # GPU
    def guess(self):

        # =======
        # MAIN - that's what we are here for - the main GPU thing
        # =======

        print()
        print("that's what we are here for - the main GPU inference thing...")
        print()

        # predictions, true_labels = [], []
        predictions = None

        # torch.cuda.empty_cache()

        for i, batch in enumerate(self.test_dataloader):

            print()

            self.dump('start empty cache...', i, 1)
            # torch.cuda.empty_cache()
            self.dump('end empty cache...', i, 1)

            self.dump('start to device...', i, 1)

            # region test shuffle
            # if not i:
            #     batch = tuple(t.to(device) for t in batch)  # to GPU when gpu (or CPU otherwise)
            # else:
            #     for t in batch:
            #         t[...] = t[torch.randperm(t.shape[0], device=t.device)]
            # endregion test shuffle

            # region to device, where first batch is fast, next ones are slow
            batch = tuple(t.to(device) for t in batch)  # to GPU when gpu (or CPU otherwise)
            # region to device, where first batch is fast, next ones are slow

            # batch = list(t.to(device) for t in batch)  # no real improvement
            # batch = batch.to(device)  # nope - this is just a list
            self.dump('end to device...', i, 1)

            b_input_ids, b_input_mask, b_labels = batch

            self.dump('start outputs...', i, 1)
            # torch.cuda.empty_cache()
            with torch.no_grad():
                # torch.cuda.empty_cache()
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            self.dump('end outputs...', i, 1)

            self.dump('logits...', i, 1)
            logits = outputs[0]
            self.dump('start detach...', i, 1)
            logits = logits.detach()
            self.dump('end detach...', i, 1)

            del outputs

            predictions = logits if predictions is None else torch.cat((predictions, logits), 0)

            del logits
            del b_input_ids
            del b_input_mask
            del b_labels
            for o in batch:
                del o
            del batch

        return predictions

    def find_with_top_level(self, predictions: torch.tensor) -> (str, int, float):

        if self.top_cat < 0:
            # # label_index = np.argmax(predictions)
            # label_index = torch.argmax(predictions)
            # return dict_label_t.get(label_index), dict_label_iid.get(label_index), predictions[label_index]
            return dict_label_t.get(0), dict_label_iid.get(0), 0

        t = None
        iid = None
        score = None
        # # for label_index in np.argsort(predictions)[::-1]:
        # for label_index in torch.argsort(predictions)[::-1]:
        #
        #     t = dict_label_t.get(label_index)
        #
        #     if self.top_cat_t in t:
        #         iid = dict_label_iid.get(label_index)
        #         score = predictions[label_index]
        #         break
        #     else:
        #         t = None

        if not t:
            t = self.top_cat_t
            iid = self.top_cat
            score = 0.

        return t, iid, score

    # just on CPU
    def encode_stuff(self) -> (list, list, list):

        # just on cpu - TODO - make on multiple cpu's

        print('disregard this - this runs on CPU and should be distributed along multi CPUs')
        print()

        for i, sent in enumerate(self.texts_test):

            if not i % PRINT_EVERY_N:
                print(f'encode_stuff {i}')

            encoded_dict = tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=PAD_MAX_TOKENS,  # Pad & truncate all sentences. was 64
                truncation=True,
                padding='max_length',
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            self.input_ids_test.append(encoded_dict['input_ids'])
            self.attention_masks_test.append(encoded_dict['attention_mask'])

        return torch.cat(self.input_ids_test, dim=0), \
            torch.cat(self.attention_masks_test, dim=0), \
               torch.tensor(self.labels_test)

    # --- measure and print times
    def dump(self, pref='blah', i = -1, print_every_n=None):

        if not print_every_n:
            print_every_n = PRINT_EVERY_N

        if i>-1 and not i % print_every_n:
            print(pref, (datetime.now() - self.d).total_seconds())
            self.d = datetime.now()
