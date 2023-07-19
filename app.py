import os
import json
import streamlit as st
import feedparser
import vertexai
from langchain.llms import VertexAI
from datetime import datetime
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

# Retrieve the JSON key file path from Streamlit Secrets
secret = st.secrets["gcp_genai_con"]


# Define the file path (this file will be created in the same directory as your Streamlit script)
file_path = "service_account.json"

# Write the dictionary to this file
with open(file_path, 'w') as f:
    json.dump(dict(secret), f)

# Set the environment variable to the path of the created file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = file_path



vertexai.init(project='prabha-sandbox')

creative_level_map = {
    'temperature':{
        'Low': 0.1,
        'Medium': 0.5,
        'High': 0.9
    },
    'top_p':{
        'Low': 0.1,
        'Medium': 0.5,
        'High': 0.9
    }
}

st.set_page_config(
    page_title="RSS Summarizer",
    page_icon="📰",
)

##### User Inputs
rss_url = st.sidebar.text_input("Enter RSS Feed URL", "http://feeds.bbci.co.uk/news/rss.xml")
summary_level = st.sidebar.radio("I am a", ["Child", "Adult"],index=1)
creative_level = st.sidebar.radio("Creative Level", ["Low", "Medium", "High"])

# Use date_input to create a calendar widget
date_range = st.sidebar.date_input(
    "Select a date range",
    value=(datetime(2023, 7, 1), datetime(2023, 7, 16))
)

# Extract start_date and end_date from date_range
start_date, end_date = date_range
#####

# Format dates as strings to display on the app
start_date_str = start_date.strftime("%d-%b '%y")
end_date_str = end_date.strftime("%d-%b '%y")

st.title(f"RSS Feed {start_date_str} to {end_date_str}")

def summarize_text(url, summary_level, creative_level):
    vertex_llm_text = VertexAI(model_name="text-bison@001",
                               top_p=creative_level_map['temperature'][creative_level],
                               temperature=creative_level_map['top_p'][creative_level])
    loader = WebBaseLoader(url)
    web_data = loader.load()

    if summary_level == "Child":
        prompt_template = """ Write a concise summary in 3 to 4 sentence:
            context: Please provide your output in a manner that a 3 year old kid would understand, use simple words and examples such that 3 year old can relate to
            Use the tone of a parent explaining to a child and if possible explain it with a story

            {text}

            Summary:"""
    else:
        prompt_template = """ Write a concise summary in 3 to 4 sentence:
            {text}

            Summary:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(vertex_llm_text, chain_type="stuff", prompt=prompt)
    summarized_text = chain.run(web_data)

    if not summarized_text.strip():
        return "Text too short for summarization"

    return summarized_text

def read_rss(url, start_date, end_date, summary_level, creative_level):
    feed = feedparser.parse(url)
    posts = feed.entries

    summaries = []
    counter = 1

    for post in posts:
        if counter > 5:
            break

        post_date = datetime(*post.published_parsed[:6])
        if start_date <= post_date <= end_date:
            summaries.append({
                'title': post.title,
                'published': post.published,
                'link': post.link,
                'summary': summarize_text(post.link, summary_level, creative_level)
            })

            counter += 1

    return summaries



if st.sidebar.button('Summarize RSS Feed'):
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.min.time())

    with st.spinner('Generating summaries...'):
        summaries = read_rss(rss_url, start_date, end_date, summary_level, creative_level)

    for i, summary in enumerate(summaries, start=1):
        with st.expander(f"{i}. {summary['title']}", expanded=True):
            st.write(f"Published: {summary['published']}")
            st.write(f"summary:")
            st.write(f"{summary['summary']}")
            st.markdown(f"[Read Full Article]({summary['link']})")
