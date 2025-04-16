import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# classifies each accident's time of occurrence 
def time_accident(time):

        
    hour = int(time.split(":")[0])
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 24:
        return "Evening"
    elif 0 <= hour < 6:
        return "Late Night"
    else:
        return "Unknown"
        
## Clean and process text        
def clean_text(text):
    return(
        text.fillna("")
        .str.lower()
        .str.translate(str.maketrans('','',string.punctuation))
        .apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words]))

    )

def task2_2():
    df = pd.read_csv("../accident.csv")
    df['TIME_OF_DAY'] = df['ACCIDENT_TIME'].astype(str).apply(time_accident)
    # create a bar chart
    time_counts = df['TIME_OF_DAY'].value_counts()
    plt.figure(figsize=(8, 5))
    time_counts.plot(kind="bar", color="skyblue")
    plt.title("Accidents by Time of Day")
    plt.ylabel("Number of Accidents")
    plt.xlabel("Time of Day")
    plt.tight_layout()
    plt.savefig("task2_2_timeofday.png")
    plt.close()

    #create the pie charts
    fig,axs = plt.subplots(2,2,figsize=(14,10))
    fig.suptitle("Top 10 Words in DCA_DESC perTime of Day")
    cleaned_desc = clean_text(df["DCA_DESC"])
    df["CLEAN_DESC"] = cleaned_desc

    for ax, time_period in zip(axs.flatten(), ['Morning', 'Afternoon', 'Evening', 'Late Night']):
        subset = df[df['TIME_OF_DAY'] == time_period]
        vectorizer = CountVectorizer(max_features=10)
        matrix = vectorizer.fit_transform(subset['CLEAN_DESC'])
        freqs = matrix.toarray().sum(axis=0)
        words = vectorizer.get_feature_names_out()
        freq_dict = dict(zip(words, freqs))

        ax.pie(freq_dict.values(), labels=freq_dict.keys(), autopct='%1.1f%%')
        ax.set_title(time_period)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig("task2_2_wordpies.png")
    plt.close()

    # create the stacked bar charts
    df["ACCIDENT_DATE"] = pd.to_datetime(df["ACCIDENT_DATE"], errors='coerce')
    df["DAY_NUM"] = df["ACCIDENT_DATE"].dt.dayofweek  # Monday = 0, Sunday = 6
    selected_days = [0, 4, 6]  # Monday, Friday, Sunday
    filtered_df = df[df["DAY_NUM"].isin(selected_days)]
    
    pivot_table = pd.pivot_table(
    filtered_df,
    index='DAY_NUM',
    columns='TIME_OF_DAY',
    values='DCA_DESC',
    aggfunc='count'
    ).fillna(0)


    expected_order = ['Morning', 'Afternoon', 'Evening', 'Late Night']
    for col in expected_order:
        if col not in pivot_table.columns:
            pivot_table[col] = 0
    pivot_table = pivot_table[expected_order]

    pivot_table.index = pivot_table.index.map({0: "Monday", 4: "Friday", 6: "Sunday"})

    pivot_table.plot(
        kind='bar',
        stacked=True,
        figsize=(10, 6),
        colormap='tab20c'
    )
    plt.title("Accidents by Day and Time of Day (Mon, Fri, Sun)")
    plt.ylabel("Number of Accidents")
    plt.xlabel("Day of the Week")
    plt.tight_layout()
    plt.savefig("task2_2_stackbar.png")
    plt.close()
    return

