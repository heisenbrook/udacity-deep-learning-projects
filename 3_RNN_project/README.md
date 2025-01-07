### Third project - Text Translation and Sentiment Analysis using Transformers

The objective of this project is to analyze the sentiment of movie reviews in three different languages - English, French, and Spanish. We have been given 30 movies, 10 in each language, along with their reviews and synopses in separate CSV files named ``movie_reviews_eng.csv``, ``movie_reviews_fr.csv``, and ``movie_reviews_sp.csv``.

1. The first step of this project is to convert the French and Spanish reviews and synopses into English. This will allow us to analyze the sentiment of all reviews in the same language. We will be using pre-trained transformers from HuggingFace to achieve this task.

2. Once the translations are complete, we will create a single dataframe that contains all the movies along with their reviews, synopses, and year of release in all three languages. This dataframe will be used to perform sentiment analysis on the reviews of each movie.

3. Finally, we will use pretrained transformers from HuggingFace to analyze the sentiment of each review. The sentiment analysis results will be added to the dataframe. The final dataframe will have 30 rows

The output of the project will be a CSV file with a header row that includes column names such as **Title**, **Year**, **Synopsis**, **Review**, **Review Sentiment**, and **Original Language**. The Original Language column will indicate the language of the review and synopsis (*en/fr/sp*) before translation. The dataframe will consist of 30 rows, with each row corresponding to a movie.