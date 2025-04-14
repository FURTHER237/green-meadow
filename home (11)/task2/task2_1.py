import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import string
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords

def task2_1():
    df = pd.read_csv("../accident.csv")
    stop_words = set(stopwords.words('english'))
    
    # Clean and process text
    cleaned_descriptions = (
        df['DCA_DESC']
        .dropna()
        .str.lower()
        .str.translate(str.maketrans('', '', string.punctuation))
        .apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
    )
    
    # Get word frequencies
    all_words = ' '.join(cleaned_descriptions).split()
    word_freq = Counter(all_words)
    top_20_words = dict(word_freq.most_common(20))
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white')
    wordcloud.generate_from_frequencies(top_20_words)
    
    # Save the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('task2_1_word_cloud.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return
