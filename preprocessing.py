import os

from tqdm import tqdm
import pandas as pd


def prepare_imdb_data_csv(folder_name = 'aclImdb'):

    label_dict = {'neg': 0,
                'pos': 1}

    data = []

    for test_train_folder in ('test', 'train'):
        for neg_pos_folder in ('pos', 'neg'):
            path = os.path.join(folder_name, test_train_folder, neg_pos_folder)
            for txt_file in tqdm(os.listdir(path), desc=f"Reading files in {test_train_folder}/{neg_pos_folder}", leave=False):

                with open(os.path.join(path,txt_file), 'r', encoding='utf-8') as infile:
                    if neg_pos_folder in label_dict:
                        data.append([infile.read(), label_dict[neg_pos_folder]])

    df = pd.DataFrame(data, columns=['Review', "Sentiment"])
    df.to_csv('sentiment_imdb_data.csv', index= False, encoding='utf-8')

    return df