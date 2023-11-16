import streamlit as st
import pandas
import helper
import preprocessor
import matplotlib.pyplot as plt
import seaborn as sns

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")        #text data to string


    time = ['Select Format', 12, 24]
    time_format = st.sidebar.selectbox("Select Time Format", time)

    if time_format == 12:
        df = preprocessor.preprocess12(data)
        user_list = df['users'].unique().tolist()
        user_list.remove('group_notifications')
        user_list.sort()
        user_list.insert(0, "Overall")

        selected_user = st.sidebar.selectbox("Show Analysis Wrt", user_list)  # for selecting users
    elif time_format == 24:
        df = preprocessor.preprocess24(data)
        user_list = df['users'].unique().tolist()
        user_list.remove('group_notifications')
        user_list.sort()
        user_list.insert(0, "Overall")

        selected_user = st.sidebar.selectbox("Show Analysis Wrt", user_list)  # for selecting users

    else:
        st.title('Chose Format')


    if st.sidebar.button("Show Analysis"):
        num_messages, words, media_shared, links = helper.fetch_stats(selected_user, df)

        st.title('Total Statistics')
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(media_shared)
        with col4:
            st.header("Links Shared")
            st.title(links)

        #mothly analysis
        st.title('Monthly Timeline')
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='brown')
        plt.xticks(rotation=90)
        st.pyplot(fig)

        #daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['Date'], daily_timeline['message'], color='black')
        plt.xticks(rotation=90)
        st.pyplot(fig)

        #most busy day, month
        st.title('Activity Map')

        col1, col2 = st.columns(2)
        with col1:
            st.header('Most Busy Day')
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            st.pyplot(fig)

        with col2:
            st.header('Most Busy Month')
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            st.pyplot(fig)

        #activity heatmap
        st.title('Weekly Activity Map')
        pivot_table = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        #plt.figure(figsize=(20, 6))
        ax = sns.heatmap(pivot_table)
        plt.xticks(rotation=90)
        st.pyplot(fig)

        #most active person
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, p = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation=90)
                st.pyplot(fig)

            with col2:
                st.dataframe(p)

        #Wordcloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        #most common words
        most_common_df = helper.most_common_words(selected_user, df)

        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1], color='green')

        plt.xticks(rotation=90)

        st.title("Most Common Words")
        st.pyplot(fig)

        #st.dataframe(most_common_df)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        if emoji_df.empty:
            st.header('No Emoji Available for Analysis')
        else:

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(emoji_df)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                st.pyplot(fig)


        #Sentiment Analysis
        temp = helper.sentiment_analysis(selected_user, df)
        st.title('Sentiment Analysis')

        col1, col2, col3 = st.columns(3)

        with col1:
            st.header('Monthly Positive')
            fig, ax = plt.subplots()
            ax.bar(temp['month'], temp['pos'], color='green')
            st.pyplot(fig)

        with col2:
            st.header('Monthly Neutral')
            fig, ax = plt.subplots()
            ax.bar(temp['month'], temp['neu'], color='grey')
            st.pyplot(fig)

        with col3:
            st.header('Monthly Negative')
            fig, ax = plt.subplots()
            ax.bar(temp['month'], temp['neg'], color='red')
            st.pyplot(fig)

        with col1:
            st.header('Weekly Positive')
            fig, ax = plt.subplots()
            ax.bar(temp['day_name'], temp['pos'], color='green')
            st.pyplot(fig)

        with col2:
            st.header('Weekly Positive')
            fig, ax = plt.subplots()
            ax.bar(temp['day_name'], temp['pos'], color='grey')
            st.pyplot(fig)

        with col3:
            st.header('Weekly Positive')
            fig, ax = plt.subplots()
            ax.bar(temp['day_name'], temp['pos'], color='red')
            st.pyplot(fig)

        with col1:
            st.header('Weekly Activity Positive')
            pos_table = temp.pivot_table(index='day_name', columns = 'period', values = 'pos', aggfunc = 'count').fillna(0)
            fig, ax = plt.subplots()
            ax = sns.heatmap(pos_table)
            st.pyplot(fig)

        with col2:
            st.header('Weekly Activity Neutral')
            neu_table = temp.pivot_table(index='day_name', columns='period', values='neu', aggfunc='count').fillna(0)
            fig, ax = plt.subplots()
            ax = sns.heatmap(neu_table)
            st.pyplot(fig)

        with col3:
            st.header('Weekly Activity Negative')
            neg_table = temp.pivot_table(index='day_name', columns='period', values='neg', aggfunc='count').fillna(0)
            fig, ax = plt.subplots()
            ax = sns.heatmap(neg_table)
            st.pyplot(fig)
