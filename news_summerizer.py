import requests
# from dotenv import load_dotenv
import streamlit as st
from bs4 import BeautifulSoup
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()

# Access the Hugging Face API key
huggingface_api_key = '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'

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model='t5-small', device=0 if torch.cuda.is_available() else -1)

tokenizer = T5Tokenizer.from_pretrained('fine-tuned-t5')
model = T5ForConditionalGeneration.from_pretrained('fine-tuned-t5').to(device)
# model.to(device)

def answer_question(question, context):
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def fetch_news(api_key, category="sports", text_input=None):
    if category == 'everything':
        url = 'https://newsapi.org/v2/everything'
        params = {
            'apiKey': api_key,
            'pageSize': 5,  # Number of articles to fetch
            'q': text_input
        }
    else:
        url = 'https://newsapi.org/v2/top-headlines'
        params = {
            'apiKey': api_key,
            'category': category,
            'language': 'en',
            'country': 'us',
            'pageSize': 5  # Number of articles to fetch
        }
    response = requests.get(url, params=params)
    return response.json()

def summarize_articles(content):
    summary = summarizer(content, max_length=150, min_length=100, do_sample=False)
    return summary[0]['summary_text']

def get_full_article(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        full_text = ' '.join([para.get_text() for para in paragraphs])
        if len(full_text) > 25:
            return full_text
        else:
            return None
    except Exception as e:
        print(f"Failed to retrieve article from {url}: {e}")
        return None

def clean_article_text(text):
    if text is None:
        return None
    text = text.replace('\n', ' ').strip()
    return text

api_key = '48fdf9717571498bb5af3f6924d78dbd'

st.title("Fast News")
st.write('Get all the latest news in short from various categories')
category = st.selectbox(
    "Select a category from the list:",
    ["business", "entertainment", "general", "health", "science", "sports", "technology", "everything"]
)

query = None
if category == "everything":
    query = st.text_input("Enter a search query:")

if "news_data" not in st.session_state:
    st.session_state.news_data = None

if st.button("Submit"):
    st.session_state.news_data = fetch_news(api_key, category=category, text_input=query)

if st.session_state.news_data:
    articles = st.session_state.news_data['articles']
    for i, article in enumerate(articles):
        content = article['content'] if article['content'] else article['description']
        url = article['url']
        full_text = clean_article_text(get_full_article(url))
        if full_text is not None:
            summarized_articles = summarize_articles(full_text)
            st.write(f"title: {article['title']}")
            summarized_articles = summarized_articles.replace("$", "\\$")
            st.write(f"Summary: {summarized_articles}")
            st.write(f"source: {article['source']['name']}")
            st.write(f"url: {article['url']}")
            if st.button(f"Ask about '{article['title']}'", key=f"ask_button_{i}"):
                st.session_state[f"show_input_{i}"] = True
            
            if st.session_state.get(f"show_input_{i}", False):
                question = st.text_input("Ask question here", key=f"question_input_{i}")
                if st.button("Submit Question", key=f"submit_question_{i}"):
                    answer = answer_question(question, full_text)
                    mid = len(answer)//2
                    st.write(answer[:mid])
            st.markdown("---")
