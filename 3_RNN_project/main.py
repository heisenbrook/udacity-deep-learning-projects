import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from transformers import pipeline
from utils.preprocess import preprocess_data, translate

df = preprocess_data()

fr_en_model_name = "Helsinki-NLP/opus-mt-fr-en"
es_en_model_name = "Helsinki-NLP/opus-mt-es-en"
fr_en_model = MarianMTModel.from_pretrained(fr_en_model_name)
es_en_model = MarianMTModel.from_pretrained(es_en_model_name)
fr_en_tokenizer = MarianTokenizer.from_pretrained(fr_en_model_name)
es_en_tokenizer = MarianTokenizer.from_pretrained(es_en_model_name)

fr_reviews_en, fr_synopsis_en, es_reviews_en, es_synopsis_en = [], [], [], []

# filter reviews in French and translate to English
fr_reviews = df['Review'][df['Original Language'] == 'French'].to_list()
for i in fr_reviews:
    fr_reviews_en.append(translate(str(i), fr_en_model, fr_en_tokenizer))
fr_reviews_en = pd.Series(fr_reviews_en)

# filter synopsis in French and translate to English
fr_synopsis = df['Synopsis'][df['Original Language'] == 'French'].to_list()
for i in fr_synopsis:
    fr_synopsis_en.append(translate(str(i), fr_en_model, fr_en_tokenizer))
fr_synopsis_en = pd.Series(fr_synopsis_en)

# filter reviews in Spanish and translate to English
es_reviews = df['Review'][df['Original Language'] == 'Spanish'].to_list()
for i in es_reviews:
    es_reviews_en.append(translate(str(i), es_en_model, es_en_tokenizer))
es_reviews_en = pd.Series(es_reviews_en)

# filter synopsis in Spanish and translate to English
es_synopsis = df['Synopsis'][df['Original Language'] == 'Spanish'].to_list()
for i in es_synopsis:
    es_synopsis_en.append(translate(str(i), es_en_model, es_en_tokenizer))
es_synopsis_en = pd.Series(es_synopsis_en)

# update dataframe with translated text
# add the translated reviews and synopsis - you can overwrite the existing data
df.replace(df['Review'][df['Original Language'] == 'French'].to_list(), fr_reviews_en, inplace=True)
df.replace(df['Synopsis'][df['Original Language'] == 'French'].to_list(), fr_synopsis_en, inplace=True)

df.replace(df['Review'][df['Original Language'] == 'Spanish'].to_list(), es_reviews_en, inplace=True)
df.replace(df['Synopsis'][df['Original Language'] == 'Spanish'].to_list(), es_synopsis_en, inplace=True)