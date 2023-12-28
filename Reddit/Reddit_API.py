"""
Project Name: MACHINE LEARNING TECHNIQUES FOR ESTIMATING SUICIDAL RISK ON SOCIAL MEDIA
Author: Samuel Bernier
Thesis Paper (French): https://uqo.on.worldcat.org/oclc/1415207814
GitHub repository: 
Huggin Face repository: https://huggingface.co/BernierS/SetFit_Suicidal_Risk
File Description:
    This file is used to fetch posts and comments from Reddit using the Reddit API.
--------------------------------------------------------------------------------
This file is part of the MACHINE LEARNING TECHNIQUES FOR ESTIMATING SUICIDAL RISK SUICIDAL RISK ON SOCIAL NETWORKS project, 
developed as a part of Samuel Bernier's thesis. For more information, visit https://uqo.on.worldcat.org/oclc/1415207814.
--------------------------------------------------------------------------------
"""

import datetime
import hashlib
import os
import pickle
import time
import praw
import prawcore
import csv
import re
import os

from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

# Constants
file_name = 'Data/reddit_data.csv'
authors_limit = 100000 # The Reddit API seems to limit at 1000 per request
post_limit = 10

# Reddit app credentials
client_id = os.environ.get("REDDIT_CLIENT_ID")
client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
user_agent = os.environ.get("REDDIT_USER_AGENT")
username = os.environ.get("REDDIT_USERNAME")
password = os.environ.get("REDDIT_PASSWORD")

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
    username=username,
    password=password,
)

# Remove HTML tags and URLs
def process_text(text):
    # Remove HTML tags
    text = re.sub('<[^<]+?>', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)

    # Remove non-ASCII characters
    text = re.sub('[^\x00-\x7F]+', '', text)
    
    return text

# Get the authors on SuicideWatch
def GetAuthors():
    print(f"Fetching authors on SuicideWatch")
    subreddit = reddit.subreddit("SuicideWatch")

    # Fetch the hot X posts from SuicideWatch
    new_posts = subreddit.new(limit=authors_limit)
    authors = []

    # Counter
    author_counter = 0

    for post in new_posts:
        authors.append(post.author)
        author_counter += 1

    # Returns a list of authors
    return authors, author_counter

# Function to get the posts and comments from the authors
def GetPosts(authors, existing_ids):

    # File name
    file_name = 'Data/reddit_data.csv'

    # Set a minimum character limit for posts and comments
    min_char_count = 100 # 00 is about 1-2 sentences

    # Counter
    post_counter = 0
    comment_counter = 0

    # Try to load the hashed_usernames dictionary from a file
    try:
        with open('Data/hashed_usernames.pickle', 'rb') as f:
            hashed_usernames = pickle.load(f)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty dictionary
        hashed_usernames = {}
    
    # Iterate through authors
    for user in authors:
        # Check if the username is in the dictionary
        if user is None:
            print("**Deleted user account. Skipping...")
            continue
        if user in hashed_usernames:
            hashed_username = hashed_usernames[user]
        else:
            # Hash the username and add it to the dictionary
            hashed_username = hashlib.sha256(user.name.encode()).hexdigest()[:8]
            hashed_usernames[user] = hashed_username

        print(f"Fetching posts and comments for user {user}")
        # Fetch new data and write to CSV
        with open(file_name, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Fetch recent posts
            try:
                for post in user.submissions.new(limit=post_limit):
                    if post.id not in existing_ids:
                        
                        # Process the text before verifying the length
                        processed_selftext = process_text(post.selftext)

                        # Check if the post is long enough
                        if len(processed_selftext) > min_char_count:
                            print("Posts:")
                            created_date = datetime.datetime.fromtimestamp(post.created_utc).isoformat()
                            print(post.title)
                            writer.writerow(["Post", post.id, hashed_usernames[post.author], processed_selftext, process_text(post.selftext), str(post.subreddit), post.score, post.url, created_date])
                            post_counter += 1
                        else:
                            print("**Post is too short, skipping...")
                    else:
                        print("**Post already exists, skipping...")

            # If there are no more posts, skip to the next user
            except AttributeError as e:
                print(f"**An error occurred with user {user}, there is most likely no more post. Skipping to next user.")
                continue

            # If there are too many requests, wait for 90 seconds
            except prawcore.exceptions.TooManyRequests:
                print("**Too many requests, waiting for 90 seconds...")
                print("Current counters: ")
                print(f"Post counter: {post_counter}")
                print(f"Comment counter: {comment_counter}")
                time.sleep(90)
                continue

            # If the error "Forbidden" happens, skip to the next user
            except prawcore.exceptions.PrawcoreException as e:
                print(f"**PRAW error: {e}, waiting for 90 seconds...")
                print("Current counters: ")
                print(f"Post counter: {post_counter}")
                print(f"Comment counter: {comment_counter}")
                time.sleep(90)
                continue
            # Fetch recent comments
            try:                
                for comment in user.comments.new(limit=post_limit):
                    if comment.id not in existing_ids:
                        
                        # Process the text before verifying the length
                        processed_comment = process_text(comment.body)

                        # Check if the comment is long enough
                        if len(processed_comment) > min_char_count:
                            print("Comments:")
                            created_date = datetime.datetime.fromtimestamp(comment.created_utc).isoformat()
                            print(comment.body)
                            writer.writerow(["Comment", comment.id, hashed_usernames[comment.author],"", processed_comment, str(comment.subreddit), comment.score, comment.permalink, created_date])
                            comment_counter += 1
                        else:
                            print("**Comment is too short, skipping...")
                    else:
                        print("**Comment already exists, skipping...")

            # If there are no more comments, skip to the next user
            except AttributeError as e:
                print(f"**An error occurred with user {user}, there is most likely no more comments. Skipping to next user.")
                continue

            # If there are too many requests, wait for 90 seconds
            except prawcore.exceptions.TooManyRequests:
                print("**Too many requests, waiting for 90 seconds...")
                print("Current counters: ")
                print(f"Post counter: {post_counter}")
                print(f"Comment counter: {comment_counter}")
                time.sleep(90)
                continue

            # If the error "Forbidden" happens, skip to the next user
            except prawcore.exceptions.PrawcoreException as e:
                print(f"**PRAW error: {e}, waiting for 90 seconds...")
                print("Current counters: ")
                print(f"Post counter: {post_counter}")
                print(f"Comment counter: {comment_counter}")
                time.sleep(90)
                continue

        if post_counter + comment_counter >= authors_limit:
            print("Reached the limit of post, stopping...")
            break

    # Save the hashed_usernames dictionary to a file
    with open('Data/hashed_usernames.pickle', 'wb') as f:
        pickle.dump(hashed_usernames, f)

    return post_counter, comment_counter

def main():
    start_time = time.time()

    existing_ids = set()
    # Check if the file exists and write headers if it's not
    if not os.path.isfile(file_name):
        with open(file_name, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Type", "ID", "Author", "Title", "Body or Selftext", "Subreddit", "Score", "URL", "Created Date"])
    else:
        # Load existing data
        with open(file_name, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)  # skip the headers
            existing_ids = {row[1] for row in reader}

    # Get the authors on SuicideWatch
    authors, author_counter = GetAuthors()
    print(f"Authors: {authors}")

    # Write posts to CSV
    post_counter, comment_counter = GetPosts(authors, existing_ids)

    # Print counters
    print(f"Number of authors fetched (initially): {author_counter}")
    print(f"Number of posts fetched: {post_counter}")
    print(f"Number of comments fetched: {comment_counter}")

    end_time = time.time()

    print(f"Execution time: {(end_time - start_time)/60} minutes")

if __name__ == "__main__":
    main()