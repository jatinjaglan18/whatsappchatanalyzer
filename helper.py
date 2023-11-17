#from urlextract import URLExtract
#from wordcloud import WordCloud
import pandas as pd
from collections import Counter
#import emoji
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import numpy

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

extractor = URLExtract()

def fetch_stats(user_name, df):
    '''if user_name == "Overall":
        #total messages
        num_message = df.shape[0]

        #total words
        words = []
        for message in df['message']:
            words.extend(message.split())

        return num_message, len(words)
    else:
        user_df = df[df['users'] == user_name]
        num_message = df[df['users'] == user_name].shape[0]

        words = []
        for message in user_df['message']:
            words.extend(message.split())

        return num_message, len(words)'''

    if user_name != 'Overall':
        df = df[df['users'] == user_name]

    num_message = df.shape[0]

    words = []
    for message in df['message']:
        words.extend(message.split())

    #media shared
    media_shared = df[df['message'] == '<Media omitted>\n'].shape[0]

    #links
    links = []
    for message in df['message']:
        links.extend(extractor.find_urls(message))

    return num_message, len(words), media_shared, len(links)

def most_busy_users(df):
    x = df['users'].value_counts().head()
    p = p =round((df['users'].value_counts()/df.shape[0])*100,2).reset_index()
    p.columns=['Names','Percent']
    return x, p

def create_wordcloud(user_name, df):
    f = open('Stopwords_hinglish.txt', 'r')
    stop_words = f.read()

    if user_name != 'Overall':
        df = df[df['users'] == user_name]

    temp = df[df['users'] != 'group_notifications']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(user_name, df):
    f = open('Stopwords_hinglish.txt', 'r')
    stop_words = f.read()

    if user_name != 'Overall':
        df = df[df['users'] == user_name]

    temp = df[df['users'] != 'group_notifications']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(user_name, df):
    if user_name != 'Overall':
        df = df[df['users'] == user_name]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.UNICODE_EMOJI['en']])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(user_name, df):
    if user_name != 'Overall':
        df = df[df['users'] == user_name]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(user_name, df):
    if user_name != 'Overall':
        df = df[df['users'] == user_name]

    df['Date'] = df['date'].dt.date
    daily_timeline = df.groupby(df['Date']).count()['message'].reset_index()

    return daily_timeline

def week_activity_map(user_name, df):
    if user_name != 'Overall':
        df = df[df['users'] == user_name]

    return df['day_name'].value_counts()

def month_activity_map(user_name, df):
    if user_name != 'Overall':
        df = df[df['users'] == user_name]

    busy_month = df['month'].value_counts()
    return busy_month

def activity_heatmap(user_name, df):
    if user_name != 'Overall':
        df = df[df['users'] == user_name]

    pivot_table = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return pivot_table


def senti(message):
    t_xt = tokenizer(message, return_tensors='pt')
    output = model(**t_xt)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    score_dict = {
        'neg': round(scores[0], 2),
        'neu': round(scores[1], 2),
        'pos': round(scores[2], 2)
    }

    return score_dict

def sentiment_analysis(user_name, df):
    if user_name != 'Overall':
        df = df[df['users'] == user_name]

    temp = df[df['users'] != 'group_notifications']
    temp = temp[temp['message'] != '<Media omitted>\n']

    l = []
    for msg in temp['message']:
        l.append(senti(msg))

    s = pd.DataFrame(l)
    temp['pos'] = s['pos']
    temp['neu'] = s['neu']
    temp['neg'] = s['neg']

    return temp

