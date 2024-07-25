import os
import re
from tqdm import tqdm
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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


def regex_preprocessing(text):
    # Usuwanie znaczników HTML
    text = re.sub('<[^>]*>', '', text)
    
    # Wyszukiwanie emotikonów
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|D|P)', text)
    
    # Czyszczenie tekstu i konwersja na małe litery
    text = re.sub('[\W]+', ' ', text.lower())
    
    # Łączenie tekstu z emotikonami, usuwanie znaków '-'
    text = text + ' ' + ' '.join(emoticons).replace('-', '')
    
    # Usuwanie podwójnych spacji, jeśli występują
    text = re.sub('\s+', ' ', text).strip()
    
    return text


def tokenize(text):
    return text.split()

def tokenize_porter(text):
    porter = PorterStemmer()
    stop_words = stopwords.words('english')
    return [porter.stem(word) for word in tokenize(text) if word not in stop_words]