import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from transformers import pipeline

def preprocess_data() -> pd.DataFrame:
    """
    Reads movie data from .csv files, map column names, add the "Original Language" column,
    and finally concatenate in one resultant dataframe called "df".
    """
    cols = ['Title','Year','Synopsis','Review']
    df_eng = pd.read_csv("data/movie_reviews_eng.csv")
    df_eng.columns = cols
    df_eng['Original Language'] = 'English'
    df_fr = pd.read_csv("data/movie_reviews_fr.csv")
    df_fr.columns = cols
    df_fr['Original Language'] = 'French'
    df_sp = pd.read_csv("data/movie_reviews_sp.csv")
    df_sp.columns = cols
    df_sp['Original Language'] = 'Spanish'
    
    df= pd.concat([df_eng, df_fr, df_sp])
    df.reset_index(drop=True, inplace=True)
    return df

def translate(text: str, model, tokenizer) -> str:
    """
    function to translate a text using a model and tokenizer
    """
    # encode the text using the tokenizer
    inputs = tokenizer(text, return_tensors='pt')

    # generate the translation using the model
    outputs = model.generate(**inputs)

    # decode the generated output and return the translated text
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded