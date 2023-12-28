"""
Project Name: MACHINE LEARNING TECHNIQUES FOR ESTIMATING SUICIDAL RISK ON SOCIAL MEDIA
Author: Samuel Bernier
Thesis Paper (French): https://uqo.on.worldcat.org/oclc/1415207814
GitHub repository: https://github.com/BernierS/SetFit_Suicidal_Risk
Huggin Face repository: https://huggingface.co/BernierS/SetFit_Suicidal_Risk
File Description:
    Application.py is the main file using the created Dataset to display the information gathered. 
--------------------------------------------------------------------------------
This file is part of the MACHINE LEARNING TECHNIQUES FOR ESTIMATING SUICIDAL RISK SUICIDAL RISK ON SOCIAL NETWORKS project, 
developed as a part of Samuel Bernier's thesis. For more information, visit https://uqo.on.worldcat.org/oclc/1415207814.
--------------------------------------------------------------------------------
"""


import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the CSV file into a DataFrame
reddit_df = pd.read_csv('Data/reddit_sentences.csv')

# ------------------Stats for the whole dataset----------------------------
def complete_dataset():
    print("Dataset stats: \n")

    # Count unique authors in the dataset
    unique_authors_count = reddit_df['Author'].nunique()
    print(f"There are {unique_authors_count} unique authors in the dataset.")

    # Count unique IDs in the dataset
    unique_ids_count = reddit_df['ID'].nunique()
    print(f"There are {unique_ids_count} unique publications in the dataset.")

    # Count the occurrences for each label text
    label_counts_overall = reddit_df['Label Text'].value_counts()
    print(" \nLabel counts for the whole dataset:")
    print(label_counts_overall)
    
    # Count how many unique Subreddits there are
    unique_subreddits_count = reddit_df['Subreddit'].nunique()
    print(f"There are {unique_subreddits_count} unique subreddits in the dataset.")

    # Count the occurrences for each subreddit
    subreddit_counts = reddit_df['Subreddit'].value_counts()
    print(" \nSubreddit counts for the whole dataset:")
    print(subreddit_counts[:20])

    #------------------Pie charts----------------------------
    # Define explode values to separate the smaller slices a bit
    explode_values = [0.1 if count < 1000 else 0 for count in label_counts_overall]

    # Plotting with legend and exploded slices
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the pie chart with legend and exploded slices
    label_counts_overall.plot.pie(explode=explode_values, startangle=90, ax=ax, autopct='%1.1f%%', pctdistance=0.85, labels=None)

    # Draw a circle in the center to make it a donut chart (this gives more space for labels)
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    # Add a legend to the chart
    ax.legend(title='Labels', labels=label_counts_overall.index, loc='best', bbox_to_anchor=(1, 0.5))

    ax.set_title('Distribution of Different Labels')
    ax.set_ylabel('')  # Remove y-axis label for clarity
    plt.tight_layout()
    plt.show()
    plt.savefig('complete_dataset_pie_chart.png', bbox_inches='tight')


    # ------------------Word cloud----------------------------
    # Count the number of occurrences for the top 100 subreddits
    subreddit_counts_100 = reddit_df['Subreddit'].value_counts().head(100)

    # Generate word cloud data for the top 100 subreddits
    wordcloud_data_100 = {subreddit: count for subreddit, count in subreddit_counts_100.items()}

    # Create a word cloud for the top 100 subreddits
    wordcloud_100 = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(wordcloud_data_100)

    # Plot the word cloud
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud_100, interpolation='bilinear')
    plt.axis('off')
    plt.title('Top 100 Subreddits')
    plt.show()
    plt.savefig('Data/complete_dataset_word_cloud.png', bbox_inches='tight')


# ------------------Stats for a random author----------------------------
def random_author():
    print("\nRandom author stats: \n")
    # Select a random author
    # random_author = np.random.choice(reddit_df['Author'].unique())
    random_author = '69f2597b'

    # Group by 'ID', 'Title', 'Author', then aggregate the 'Label' column into a list
    grouped_data_with_author_url = reddit_df.groupby(['Author', 'Title', 'ID'])['Label'].agg(list).reset_index()

    # Filter the grouped dataset for the selected author using the combined random author
    author_data_with_url_combined = grouped_data_with_author_url[grouped_data_with_author_url['Author'] == random_author]

    print(author_data_with_url_combined)

    # ------------------Pie chart----------------------------

    # Translation dictionary
    translations = {
    'Ability to hope for change': 'Capacité à espérer un changement',
    'Previous attempt': 'Tentatives de suicide antérieures',
    'Ability to control oneself': 'Capacité à se contrôler',
    'Ability to take care of oneself': 'Capacité à prendre soin de soi',
    'Presence of a loved one': 'Présence des proches',
    'Consumption': 'Usage de substances',
    'Suicidal planning': 'Planification du suicide',
    'Other': 'Autre',
    }

    # Filter the dataset for entries by that author
    author_df = reddit_df[reddit_df['Author'] == random_author]

    # Count the number of occurrences for each label text
    label_counts = author_df['Label Text'].value_counts()

    # Translate the labels in french
    label_counts = label_counts.rename(index=translations)


    # Print the number of occurrences for each label text
    print("Label counts for selected author: \n")
    print(label_counts)

    # Plot a pie chart
    plt.figure(figsize=(10, 7))
    label_counts.plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f'Classes associées avec l\'auteur: {random_author}')
    plt.ylabel('')  # Remove y-axis label for clarity
    plt.show()
    plt.savefig(f'Data/random_author_{random_author}_pie_chart.png', bbox_inches='tight')


# Main function
if __name__ == '__main__':
    # complete_dataset()
    random_author()