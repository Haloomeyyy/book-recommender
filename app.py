from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and prepare data
df = pd.read_csv('books.csv', on_bad_lines='skip')
df = df[['title', 'authors', 'average_rating', 'genre_and_votes']].dropna() if 'genre_and_votes' in df.columns else df[['title', 'authors', 'average_rating']].dropna()
df['content'] = df['title'] + ' ' + df['authors']
df = df.drop_duplicates(subset='title').reset_index(drop=True)

# Build similarity matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(title, n=5):
    title = title.lower()
    matches = df[df['title'].str.lower().str.contains(title)]
    if matches.empty:
        return None, None
    idx = matches.index[0]
    matched_title = df.loc[idx, 'title']
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:n]
    book_indices = [s[0] for s in sim_scores]
    results = df.iloc[book_indices][['title', 'authors', 'average_rating']].copy()
    return matched_title, results.to_dict('records')

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    matched_title = None
    query = ''
    error = None
    if request.method == 'POST':
        query = request.form.get('book_title', '')
        matched_title, recommendations = recommend(query)
        if recommendations is None:
            error = f'No book found matching "{query}". Try another title.'
    return render_template('index.html', 
                         recommendations=recommendations,
                         matched_title=matched_title,
                         query=query,
                         error=error)

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)