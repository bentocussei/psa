"""
Pulsar Social Angolano
Autor: Bento Cussei
Função: Data Analytics Specialist | Data Scientist
Data: 2025-08-15
Descrição: Aplicação para análise descritiva, diagnóstica e preditiva
           de interações nas redes sociais angolanas.
"""


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
from collections import Counter, defaultdict

# -----------------------------------------------------------------------------
# Global stopwords list to be used throughout the project
# -----------------------------------------------------------------------------
STOPWORDS_PT = set([
    'a','o','e','de','da','do','que','em','para','é','um','na','no','com','os','as','por','se',
    'mais','uma','não','nao','sou','pra','como','dos','das','nos','também','tambem','já','ja','ser','tem','vai',
    'está','estao','mesmo','pais','quem','quando','nada','essa','porque','pessoas','ainda','vamos','julho','pelo','entre','agosto','mas','saiba','dias','durante','após','antes','sobre','segundo','sem','foram','dizer','pode','minha','muitos','são','dia','neste','nesta','às','quer',
    'vida','verdade','onde','aqui','então','entao','tudo','algo','isto','esse','eles','elas','ela','ele','nós','nosso','você','vocês','assim',
    'seu','seus','sua','suas','dele','dela','delas','deles','isso','aquilo','muito','pouco','cada','todos','todas','nossos','nossas',
    'foi','será','sera','ao','aos','ate','até','ou','nas','pelos','pela','pelas','melhor','hoje','agora','nunca','ninguém','ninguem',
    'casa','talvez','sempre','nem','lá','la','cá','ca','estão','fazer','faz','falar','fala','este','esta','estes','estas','depois'
])

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load and cache the cleaned dataset with topic assignments."""
    df = pd.read_json('clean_instagram_data_topics.json')
    # convert epoch to datetime for filtering and grouping
    df['datetime_dt'] = pd.to_datetime(df['datetime_parsed'], unit='ms', utc=True).dt.tz_convert('Africa/Luanda')
    df['date'] = df['datetime_dt'].dt.date
    return df

@st.cache_data
def load_raw_data():
    """
    Load and cache the raw JSON dataset to explore comments if needed.
    Uses UTF‑8 encoding to avoid UnicodeDecodeError on Windows.
    """
    # Use UTF‑8 encoding and replace errors to handle any invalid byte sequences
    with open('instagram_data_2025-08.json', 'r', encoding='utf-8', errors='replace') as f:
        data = json.load(f)
    return data

# -----------------------------------------------------------------------------
# Additional helper functions for diagnostic visualisations
# -----------------------------------------------------------------------------
@st.cache_data
def get_topic_keywords(df: pd.DataFrame, n_top_words: int = 8):
    """
    Fit an LDA model to the post contents and extract the top words per topic.
    Returns a dictionary mapping topic index to list of top words.
    """
    # Preprocess text
    texts = df['content'].fillna('').astype(str)
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  # remove URLs
        text = re.sub(r'@\w+', ' ', text)  # remove mentions
        text = re.sub(r'#', ' ', text)
        text = re.sub(r'\d+', ' ', text)  # remove numbers
        text = re.sub(r'[^\w\s]', ' ', text)  # punctuation
        return text
    processed = texts.apply(preprocess)
    # Vectorize.  Use the global STOPWORDS_PT so that stopwords are consistent across functions.
    # Convert to list because CountVectorizer expects a list or None for stop_words
    vectorizer = CountVectorizer(stop_words=list(STOPWORDS_PT), max_df=0.9, min_df=5)
    X = vectorizer.fit_transform(processed)
    # Fit LDA with 6 topics (as usado anteriormente)
    lda = LatentDirichletAllocation(n_components=6, random_state=42)
    lda.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = {}
    for idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[::-1][:n_top_words]
        topic_keywords[idx] = [feature_names[i] for i in top_indices]
    return topic_keywords

def get_emotion_distribution(df: pd.DataFrame):
    """
    Compute the distribution of basic emotions relevant to protest analysis
    (raiva, medo, tristeza) using simple word lists. Returns a DataFrame with
    columns 'Emoção' and 'Contagem'. Posts sem palavras destas categorias são
    considerados 'Neutro'.
    """
    # Define lexicons
    emotions = {
        'Raiva': {'raiva','odio','ódio','furia','fúria','ira','indignacao','injustica','agressao','violencia'},
        'Medo': {'medo','pânico','panico','inseguranca','insegurança','receio','terror','ameaça','ameaca'},
        'Tristeza': {'triste','tristeza','luto','desanimo','desânimo','saudade','perda'},
    }
    counts = defaultdict(int)
    for text in df['content'].fillna('').astype(str):
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        found = []
        for emo, lex in emotions.items():
            if any(w in lex for w in words):
                found.append(emo)
        if found:
            # count the first matched emotion (could be multiple)
            counts[found[0]] += 1
        else:
            counts['Neutro'] += 1
    # Convert to DataFrame
    items = list(counts.items())
    return pd.DataFrame({'Emoção': [k for k,v in items], 'Contagem': [v for k,v in items]})

# New helper to compute emotion distribution over posts and all comments/replies
def get_emotion_distribution_all(df: pd.DataFrame, raw_data):
    """
    Compute the distribution of emotions (raiva, medo, tristeza) across all text in
    posts, comments e replies. Uses the same lexicons as get_emotion_distribution.
    """
    emotions = {
        'Raiva': {'raiva','odio','ódio','furia','fúria','ira','indignacao','injustica','agressao','violencia'},
        'Medo': {'medo','pânico','panico','inseguranca','insegurança','receio','terror','ameaça','ameaca'},
        'Tristeza': {'triste','tristeza','luto','desanimo','desânimo','saudade','perda'},
    }
    counts = defaultdict(int)
    # Start with post contents
    texts = list(df['content'].fillna('').astype(str))
    # Add comments and replies
    for post in raw_data:
        for comment in post.get('comments', []):
            texts.append(comment.get('text',''))
            for reply in comment.get('replies', []):
                texts.append(reply.get('text',''))
    # Compute counts
    for text in texts:
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        found = []
        for emo, lex in emotions.items():
            if any(w in lex for w in words):
                found.append(emo)
        if found:
            counts[found[0]] += 1
        else:
            counts['Neutro'] += 1
    items = list(counts.items())
    return pd.DataFrame({'Emoção': [k for k,v in items], 'Contagem': [v for k,v in items]})

def get_top_words(df: pd.DataFrame, n: int = 30):
    """Return a DataFrame with the n most frequent words across all posts."""
    text_all = ' '.join(df['content'].fillna('').astype(str))
    words = re.findall(r'\b\w+\b', text_all.lower())
    # Use the global STOPWORDS_PT so that stopwords are consistent across the app
    filtered = [w for w in words if w not in STOPWORDS_PT and len(w) > 3]
    freq = Counter(filtered)
    most = freq.most_common(n)
    return pd.DataFrame({'Palavra':[w for w,c in most],'Frequência':[c for w,c in most]})

def get_top_words_all(df: pd.DataFrame, raw_data, n: int = 30):
    """
    Return a DataFrame with the n most frequent words across posts, comments and replies.
    """
    text_all = []
    text_all.extend(df['content'].fillna('').astype(str).tolist())
    for post in raw_data:
        for comment in post.get('comments', []):
            text_all.append(comment.get('text',''))
            for reply in comment.get('replies', []):
                text_all.append(reply.get('text',''))
    joined = ' '.join(text_all)
    words = re.findall(r'\b\w+\b', joined.lower())
    # Use the global STOPWORDS_PT to remove common and uninformative terms
    filtered = [w for w in words if w not in STOPWORDS_PT and len(w) > 3]
    freq = Counter(filtered)
    most = freq.most_common(n)
    return pd.DataFrame({'Palavra':[w for w,c in most],'Frequência':[c for w,c in most]})

def get_mentions_summary(raw_data, top_n: int = 10):
    """Compute top mentioners and mentioned counts from the raw dataset."""
    mention_edges = []
    for post in raw_data:
        page = post.get('pageName','').lower()
        # post text
        text = post.get('content','') or ''
        for m in re.findall(r'@([A-Za-z0-9_\.]+)', text):
            mention_edges.append((page, m.lower()))
        # comments
        for comment in post.get('comments', []):
            c_author = comment.get('author','').lower()
            c_text = comment.get('text','') or ''
            for m in re.findall(r'@([A-Za-z0-9_\.]+)', c_text):
                mention_edges.append((c_author, m.lower()))
            for reply in comment.get('replies', []):
                r_author = reply.get('author','').lower()
                r_text = reply.get('text','') or ''
                for m in re.findall(r'@([A-Za-z0-9_\.]+)', r_text):
                    mention_edges.append((r_author, m.lower()))
    out_deg = defaultdict(int)
    in_deg = defaultdict(int)
    for src, dst in mention_edges:
        if src and dst:
            out_deg[src] += 1
            in_deg[dst] += 1
    top_out = sorted(out_deg.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_in = sorted(in_deg.items(), key=lambda x: x[1], reverse=True)[:top_n]
    mentioners_df = pd.DataFrame(top_out, columns=['Utilizador','Menções feitas'])
    mentioned_df = pd.DataFrame(top_in, columns=['Utilizador','Menções recebidas'])
    return mentioners_df, mentioned_df

# Compute network metrics for pages and commenters
def get_interaction_network(raw_data, top_n: int = 10):
    """
    Compute the number of unique comentadores por página e o número de páginas
    comentadas por cada utilizador. Returns two DataFrames: pages_df and users_df.
    """
    from collections import defaultdict
    page_commenters = defaultdict(set)
    commenter_pages = defaultdict(set)
    for post in raw_data:
        page = post.get('pageName','').lower()
        for comment in post.get('comments', []):
            author = comment.get('author','').lower()
            if author:
                page_commenters[page].add(author)
                commenter_pages[author].add(page)
            for reply in comment.get('replies', []):
                rep_author = reply.get('author','').lower()
                if rep_author:
                    page_commenters[page].add(rep_author)
                    commenter_pages[rep_author].add(page)
    page_degrees = {p: len(users) for p, users in page_commenters.items()}
    commenter_degrees = {u: len(pages) for u, pages in commenter_pages.items()}
    pages_df = pd.DataFrame(sorted(page_degrees.items(), key=lambda x: x[1], reverse=True)[:top_n],
                           columns=['Página','Comentadores únicos'])
    users_df = pd.DataFrame(sorted(commenter_degrees.items(), key=lambda x: x[1], reverse=True)[:top_n],
                           columns=['Utilizador','Páginas comentadas'])
    return pages_df, users_df

# Count occurrences of selected entities in posts, comments and replies
def get_entity_counts(df: pd.DataFrame, raw_data, entities: list):
    """Return a DataFrame with counts of entities across posts, comments and replies."""
    # Normalise entity names and handle sinónimos (ex.: policia vs polícia)
    canonical_map = {}
    for ent in entities:
        ent_lower = ent.lower()
        # Map known variants to a canonical form
        if ent_lower in ['policia', 'polícia']:
            canonical = 'polícia'
        else:
            canonical = ent_lower
        canonical_map[ent_lower] = canonical
    counts = {canonical: 0 for canonical in set(canonical_map.values())}
    # Combine all text sources (posts + comments + replies)
    texts = list(df['content'].fillna('').astype(str))
    for post in raw_data:
        for comment in post.get('comments', []):
            texts.append(comment.get('text',''))
            for reply in comment.get('replies', []):
                texts.append(reply.get('text',''))
    # Count occurrences
    for text in texts:
        text_low = text.lower()
        for original, canonical in canonical_map.items():
            if original in text_low:
                counts[canonical] += 1
    ent_df = pd.DataFrame({'Entidade': list(counts.keys()), 'Ocorrências': list(counts.values())})
    return ent_df.sort_values('Ocorrências', ascending=False)

# Compute top posts by total engagement (posts + comments + replies + likes on comments/replies)
@st.cache_data
def get_top_posts_engagement(df: pd.DataFrame, raw_data, top_n: int = 5):
    """
    For each post, compute total engagement as:
    likes on the post + number of comments (including replies) + likes on comments and replies.
    Returns a DataFrame with the top_n posts with highest engagement.
    """
    # Build a map from URL to raw post data for quick lookup
    raw_map = {post.get('url'): post for post in raw_data}
    records = []
    for _, row in df.iterrows():
        url = row['url']
        post_raw = raw_map.get(url, {})
        comment_count = 0
        comment_likes = 0
        for comment in post_raw.get('comments', []):
            comment_count += 1
            comment_likes += comment.get('likes', 0) or 0
            for reply in comment.get('replies', []):
                comment_count += 1
                comment_likes += reply.get('likes', 0) or 0
        engagement = (row['likes'] or 0) + comment_count + comment_likes
        records.append({
            'URL': url,
            'Página': row['pageName'],
            'Data': row['datetime_dt'].date(),
            'Trecho do conteúdo': (row['content'][:60] + '...') if isinstance(row['content'], str) else '',
            'Engajamento total': engagement
        })
    top_df = pd.DataFrame(records).sort_values('Engajamento total', ascending=False).head(top_n)
    return top_df

# Compute top comments by engagement
@st.cache_data
def get_top_comments_engagement(raw_data, top_n: int = 5):
    """
    Identify the first‑level comments with highest engagement. Engagement is defined as:
    likes on the comment + number of replies + number of user mentions in the comment (excluding the author).
    Returns a DataFrame with the top_n comments sorted by engagement.
    """
    import re
    records = []
    for post in raw_data:
        page = post.get('pageName')
        for comment in post.get('comments', []) or []:
            author = comment.get('author', '') or ''
            text = comment.get('text', '') or ''
            likes = comment.get('likes', 0) or 0
            replies = comment.get('replies', []) or []
            num_replies = len(replies)
            # find mentions in comment text
            mentions = re.findall(r'@([A-Za-z0-9_\.]+)', text)
            mentions_filtered = [m.lower() for m in mentions if m.lower() != author.lower()]
            num_mentions = len(mentions_filtered)
            engagement = likes + num_replies + num_mentions
            # convert datetime (if present) to date
            try:
                dt = pd.to_datetime(comment.get('datetime', 0), unit='ms', utc=True).tz_convert('Africa/Luanda')
                date = dt.date()
            except Exception:
                date = None
            records.append({
                'Página': page,
                'Autor': author,
                'Data': date,
                'Trecho do comentário': (text[:60] + '...') if len(text) > 60 else text,
                'Engajamento': engagement,
                'URL do post': post.get('url')
            })
    df_comments = pd.DataFrame(records)
    if df_comments.empty:
        return df_comments
    df_comments = df_comments.sort_values('Engajamento', ascending=False).head(top_n)
    return df_comments

# Generate a dynamic word cloud figure using Plotly
@st.cache_data
def generate_wordcloud_figure(df: pd.DataFrame, raw_data, max_words: int = 50):
    """
    Generate a word cloud as a Plotly scatter plot where words are
    positioned randomly in a unit square. Font sizes and colours
    reflect word frequency. Includes posts, comments and replies.
    """
    # Get top words across all texts
    words_df = get_top_words_all(df, raw_data, n=max_words)
    if words_df.empty:
        return None
    words = words_df['Palavra'].tolist()
    freqs = words_df['Frequência'].tolist()
    max_freq = max(freqs)
    # Map frequencies to font sizes (between 14 and 50)
    sizes = [14 + (f / max_freq) * 36 for f in freqs]
    import random
    xs = [random.random() for _ in range(len(words))]
    ys = [random.random() for _ in range(len(words))]
    # Map frequencies to colours using Viridis colour scale
    from plotly.colors import sample_colorscale
    colors = sample_colorscale('Viridis', [f / max_freq for f in freqs])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='text', text=words,
        textfont={'size': sizes, 'color': colors},
        hoverinfo='text'
    ))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        title="Nuvem de Palavras",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

# Compute engagement per page including posts, comments, replies and likes
@st.cache_data
def get_engagement_per_page(df: pd.DataFrame, raw_data, top_n: int = 10):
    """
    Compute total engagement per página. Engagement é definido como:
    posts_count + likes nos posts + comments_count (incluindo replies) + likes nos comentários e replies.
    Retorna DataFrame ordenado pelo total_engagement.
    """
    # Initialize dictionary for each page
    stats = {}
    # First, sum posts and likes from df
    for _, row in df.iterrows():
        page = row['pageName']
        stats.setdefault(page, {'posts_count': 0, 'post_likes': 0, 'comment_count': 0, 'comment_likes': 0})
        stats[page]['posts_count'] += 1
        stats[page]['post_likes'] += row['likes'] or 0
    # Now incorporate comment and reply counts/likes from raw_data
    for post in raw_data:
        page = post.get('pageName')
        if not page:
            continue
        stats.setdefault(page, {'posts_count': 0, 'post_likes': 0, 'comment_count': 0, 'comment_likes': 0})
        for comment in post.get('comments', []):
            stats[page]['comment_count'] += 1
            stats[page]['comment_likes'] += comment.get('likes', 0) or 0
            for reply in comment.get('replies', []):
                stats[page]['comment_count'] += 1
                stats[page]['comment_likes'] += reply.get('likes', 0) or 0
    # Build DataFrame
    data=[]
    for page, s in stats.items():
        total_engagement = s['posts_count'] + s['post_likes'] + s['comment_count'] + s['comment_likes']
        data.append({
            'Página': page,
            'posts_count': s['posts_count'],
            'post_likes': s['post_likes'],
            'comment_count': s['comment_count'],
            'comment_likes': s['comment_likes'],
            'total_engagement': total_engagement
        })
    df_stats = pd.DataFrame(data)
    df_stats = df_stats.sort_values('total_engagement', ascending=False).head(top_n)
    return df_stats

# Compute total comments (incl replies) and likes per page
@st.cache_data
def get_comments_likes_per_page(df: pd.DataFrame, raw_data, top_n: int = 10):
    """
    Compute total comments (including replies) and total likes (posts only) per page.
    Returns a DataFrame with columns: Página, total_comments, total_likes_posts.
    """
    # Initialize counts
    comments_count = defaultdict(int)
    likes_posts = defaultdict(int)
    # Posts and likes from df
    for _, row in df.iterrows():
        page = row['pageName']
        likes_posts[page] += row['likes'] or 0
    # Comments and replies from raw_data
    for post in raw_data:
        page = post.get('pageName')
        for comment in post.get('comments', []):
            comments_count[page] += 1
            for reply in comment.get('replies', []):
                comments_count[page] += 1
    data = []
    for page in set(list(comments_count.keys()) + list(likes_posts.keys())):
        data.append({
            'Página': page,
            'total_comments': comments_count.get(page, 0),
            'total_likes_posts': likes_posts.get(page, 0)
        })
    df_metrics = pd.DataFrame(data)
    # Sort by comments for top_n or show both separate charts later
    return df_metrics

# Compute summary statistics
@st.cache_data
def compute_summary(df: pd.DataFrame):
    summary = {}
    summary['n_posts'] = len(df)
    summary['n_pages'] = df['pageName'].nunique()
    summary['period_start'] = df['datetime_dt'].min().date()
    summary['period_end'] = df['datetime_dt'].max().date()
    # Averages and counts based purely on post-level data
    summary['avg_likes_posts'] = df['likes'].mean() if len(df) > 0 else 0
    summary['avg_comments_posts'] = df['num_comments'].mean() if len(df) > 0 else 0
    # Maintain backward-compatible keys for any remaining references
    summary['avg_likes'] = summary['avg_likes_posts']
    summary['avg_comments'] = summary['avg_comments_posts']
    # Total number of comments on posts (excluding replies)
    summary['n_comments_total_posts'] = df['num_comments'].sum()

    # Compute total comments and total likes across posts, comments and replies
    # Leverage the raw data for comment and like counts
    raw_data = load_raw_data()
    total_comments_all = 0
    total_comment_likes = 0
    for post in raw_data:
        for comment in post.get('comments', []):
            total_comments_all += 1
            total_comment_likes += comment.get('likes', 0) or 0
            for reply in comment.get('replies', []):
                total_comments_all += 1
                total_comment_likes += reply.get('likes', 0) or 0
    # Likes from posts
    total_likes_posts = df['likes'].sum()
    summary['total_comments_all'] = total_comments_all
    summary['total_likes_posts'] = total_likes_posts
    summary['total_likes_all'] = total_likes_posts + total_comment_likes
    # Averages including all likes and comments across posts, comments and replies
    n_posts = summary['n_posts']
    summary['avg_likes_all'] = (summary['total_likes_all'] / n_posts) if n_posts > 0 else 0
    summary['avg_comments_all'] = (summary['total_comments_all'] / n_posts) if n_posts > 0 else 0
    return summary

# Top tables
@st.cache_data
def get_top_pages_posts(df):
    counts = df['pageName'].value_counts().reset_index()
    counts.columns = ['page', 'n_posts']
    return counts.sort_values('n_posts', ascending=True)

@st.cache_data
def get_top_pages_likes(df):
    avg_likes = df.groupby('pageName')['likes'].mean().reset_index()
    avg_likes.columns = ['page', 'avg_likes']
    return avg_likes.sort_values('avg_likes', ascending=True)

@st.cache_data
def get_posts_per_day(df):
    return df.groupby('date').size().reset_index(name='n_posts')

# Sidebar menu mapping.  Each entry maps a human‑friendly label to an internal
# page identifier.  A radio button in the sidebar will be used instead of a
# dropdown so that all menu items aparecem listados directamente.
PAGES = {
    "Home": "home",
    "Problema": "problema",
    "Composição dos dados": "composicao",
    "EDA": "eda",
    "Análises Diagnósticas": "diagnostica",
    "Modelos Preditivos": "preditiva",
    "Resultados": "resultados",
    "Metodologia (CRISP-DM)": "metodologia",
    "Recomendações": "recomendacoes",
}

# -----------------------------------------------------------------------------
# Main Streamlit app
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Pulsar Social Angolano", layout="wide")
    # Sidebar menu as radio buttons instead of a dropdown
    page_label = st.sidebar.radio("Menu", list(PAGES.keys()))
    # Map the human‑friendly label to an internal page identifier
    page = PAGES[page_label]

    df = load_data()
    summary = compute_summary(df)

    if page == "home":
        st.title("Pulsar Social Angolano")
        st.markdown(
            """
            **Contexto:**
            O projecto Pulsar Social Angolano analisa interações nas redes sociais durante Julho e Agosto de 2025, período marcado por manifestações e reivindicações sociais.

            **Objectivo Geral:**
            Usar dados para compreender as causas e a intensidade das manifestações, identificar actores mais influentes, mapear temas mais discutidos e detectar tendências de novas mobilizações, aplicando análises descritivas, diagnósticas e preditivas.

            **Escopo actual:**
            Esta aplicação apresenta a Fase 1 - Análise Descritiva e Diagnóstica - com resultados de *modelação de tópicos*, *distribuição de sentimentos*, *nuvens de palavras* e *rede de interacções*. A fase preditiva será incluída quando os modelos estiverem maduros.
            
            **Desenvolvido por:**
            **Bento Cussei** - *Data Analytics Specialist* | *Data Scientist*.
            """
        )

        st.subheader("Principais métricas do dataset")
        # Organizar métricas em duas linhas de quatro colunas cada.  As
        # estatísticas distinguem entre likes nos posts e likes totais (posts +
        # comentários + replies), bem como comentários directos e totais.
        col_tot1, col_tot2, col_tot3, col_tot4 = st.columns(4)
        col_tot1.metric("Total de posts", summary['n_posts'])
        col_tot2.metric("Total de comentários (posts + respostas)", summary.get('total_comments_all', 0))
        col_tot3.metric("Total de likes nos posts", summary.get('total_likes_posts', 0))
        col_tot4.metric("Total de likes (todos)", summary.get('total_likes_all', 0))
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Páginas únicas", summary['n_pages'])
        col_m2.metric("Média de likes nos posts", f"{summary.get('avg_likes_posts', 0):.2f}")
        col_m3.metric("Média total de likes por post", f"{summary.get('avg_likes_all', 0):.2f}")
        col_m4.metric("Média total de comentários por post", f"{summary.get('avg_comments_all', 0):.2f}")
        # Nota sobre o período
        st.caption(f"Período analisado: {summary['period_start']} a {summary['period_end']} | Redes sociais cobertas: Instagram")
        # Explicação das métricas apresentadas
        st.markdown(
            f"Foram analisados **{summary['n_posts']}** posts provenientes de **{summary['n_pages']}** páginas entre "
            f"\"{summary['period_start']}\" e \"{summary['period_end']}\". No total, estes posts geraram **{summary['total_comments_all']}** comentários "
            f"(incluindo respostas) e **{summary['total_likes_all']}** likes ao considerar todas as interacções. "
            f"Considerando apenas os likes dos posts, cada publicação recebeu em média **{summary.get('avg_likes_posts', 0):.0f}** likes, "
            f"enquanto a média total de likes por post (incluindo likes de comentários e respostas) é **{summary.get('avg_likes_all', 0):.0f}**. "
            f"Os posts registaram em média **{summary.get('avg_comments_posts', 0):.1f}** comentários directos, "
            f"ou **{summary.get('avg_comments_all', 0):.1f}** quando se somam as respostas."
        )
        
        st.subheader("Insights chave")
        st.markdown(
            """
            - **Engajamento:** A maior parte dos posts recebe poucos likes e comentários, enquanto um pequeno número de publicações torna‑se viral, especialmente em páginas como **“XÉ Agora Aguenta”**.
            - **Temas predominantes:** Tópicos relacionados a manifestações, polícia e taxistas geram maior engajamento, apontando para preocupações com transporte e segurança.
            - **Picos temporais:** Os dias de 23 a 31 de Julho de 2025 concentram a actividade, coincidindo com protestos sobre preços de combustíveis e transporte.
            - **Rede de influência:** Algumas páginas e utilizadores funcionam como hubs, mencionando ou sendo mencionados por muitos outros, o que indica capacidade de amplificar mensagens.
            - **Sentimentos:** A maioria das mensagens é neutra, mas sentimentos negativos como raiva e tristeza estão presentes e alinhados com as reclamações registadas.
            """
        )

        # Mostrar a lista de tópicos com palavras‑chave
        st.subheader("Tópicos identificados")
        topic_keywords_home = get_topic_keywords(df)
        for idx in sorted(topic_keywords_home.keys()):
            words = ', '.join(topic_keywords_home[idx])
            st.markdown(f"**Tópico {idx}:** {words}")

        # Mostrar um dashboard geral com os principais gráficos da análise descritiva e diagnóstica
        st.subheader("Visão geral – dashboard")
        # Top páginas por número de posts e por média de likes
        top_counts_all = get_top_pages_posts(df).head(10)
        fig_counts_all = px.bar(
            top_counts_all,
            x='n_posts', y='page', orientation='h',
            title='Top páginas por número de posts', color='n_posts',
            labels={'n_posts': 'Nº de posts', 'page': 'Página'},
            color_continuous_scale='Blues'
        )
        top_likes_all = get_top_pages_likes(df).head(10)
        fig_likes_all = px.bar(
            top_likes_all,
            x='avg_likes', y='page', orientation='h',
            title='Top páginas por média de likes nos posts', color='avg_likes',
            labels={'avg_likes': 'Média de likes nos posts', 'page': 'Página'},
            color_continuous_scale='Reds'
        )
        # Posts por dia
        posts_day_all = get_posts_per_day(df)
        fig_posts_day_all = px.line(
            posts_day_all,
            x='date', y='n_posts',
            title='Número de posts por dia',
            labels={'date': 'Data', 'n_posts': 'Nº de posts'}
        )
        # Distribuição de tópicos e engajamento por tópico
        topic_counts_all = df['topic'].value_counts().sort_index()
        labels_topics_all = [f"Tópico {i}" for i in topic_counts_all.index]
        fig_topic_all = px.bar(
            x=labels_topics_all, y=topic_counts_all.values,
            labels={'x': 'Tópico', 'y': 'Nº de posts'},
            title='Distribuição de posts por tópico',
            color=topic_counts_all.values, color_continuous_scale='Blues'
        )
        # Assegurar que coluna total_engagement existe
        if 'total_engagement' not in df.columns:
            df['total_engagement'] = df['likes'] + df['num_comments']
        eng_all = df.groupby('topic')['total_engagement'].mean().reset_index()
        eng_all['topic_label'] = eng_all['topic'].apply(lambda x: f"Tópico {x}")
        fig_eng_all = px.bar(
            eng_all, x='topic_label', y='total_engagement',
            labels={'topic_label': 'Tópico', 'total_engagement': 'Engajamento médio'},
            title='Engajamento médio por tópico', color='total_engagement',
            color_continuous_scale='Viridis'
        )
        # Palavras mais frequentes
        raw_data_home = load_raw_data()
        words_df_home = get_top_words_all(df, raw_data_home, n=15)
        fig_words_home = px.bar(
            words_df_home.sort_values('Frequência', ascending=True),
            x='Frequência', y='Palavra', orientation='h',
            title='Palavras mais frequentes', color='Frequência',
            color_continuous_scale='Purples'
        )
        # Emoções
        emo_df_home = get_emotion_distribution_all(df, raw_data_home)
        fig_emo_home = px.bar(
            emo_df_home, x='Emoção', y='Contagem', color='Emoção',
            title='Distribuição de emoções',
            labels={'Contagem': 'Número de textos'}
        )
        # Entidades
        entidades_home = ['presidente','governo','polícia','policia','mpla','taxistas','joão lourenço','ministro','ministério','unita']
        ent_df_home = get_entity_counts(df, raw_data_home, entidades_home)
        fig_ent_home = px.bar(
            ent_df_home.sort_values('Ocorrências', ascending=True),
            x='Ocorrências', y='Entidade', orientation='h',
            title='Entidades mais mencionadas', color='Ocorrências',
            color_continuous_scale='Magma'
        )
        # Layout de gráficos em colunas
        # Primeira linha: top páginas posts e likes
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_counts_all, use_container_width=True)
        with c2:
            st.plotly_chart(fig_likes_all, use_container_width=True)
        # Segunda linha: posts por dia e distribuição de tópicos
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(fig_posts_day_all, use_container_width=True)
        with c4:
            st.plotly_chart(fig_topic_all, use_container_width=True)
        # Terceira linha: engajamento médio e emoções
        c5, c6 = st.columns(2)
        with c5:
            st.plotly_chart(fig_eng_all, use_container_width=True)
        with c6:
            st.plotly_chart(fig_emo_home, use_container_width=True)
        # Quarta linha: palavras frequentes e entidades
        c7, c8 = st.columns(2)
        with c7:
            st.plotly_chart(fig_words_home, use_container_width=True)
        with c8:
            st.plotly_chart(fig_ent_home, use_container_width=True)

        # Quinta linha: exibir a nuvem de palavras dinâmica
        st.subheader("Nuvem de palavras")
        cloud_home = generate_wordcloud_figure(df, raw_data_home, max_words=40)
        if cloud_home:
            st.plotly_chart(cloud_home, use_container_width=True)
        else:
            st.info("Não foi possível gerar a nuvem de palavras.")

        # Adicionar gráficos de interacções por página (comentários, likes e engajamento)
        st.subheader("Interacções por página")
        # Calcular métricas por página usando dados completos
        comments_likes_home = get_comments_likes_per_page(df, raw_data_home)
        # Top páginas por comentários
        comments_top_home = comments_likes_home.sort_values('total_comments', ascending=True).head(10)
        fig_comments_home = px.bar(
            comments_top_home,
            x='total_comments', y='Página', orientation='h',
            title='Total de comentários por página', color='total_comments',
            labels={'total_comments': 'Número de comentários', 'Página': 'Página'},
            color_continuous_scale='IceFire'
        )
        # Top páginas por likes nos posts
        likes_top_home = comments_likes_home.sort_values('total_likes_posts', ascending=True).head(10)
        fig_likes_home = px.bar(
            likes_top_home,
            x='total_likes_posts', y='Página', orientation='h',
            title='Total de likes nos posts por página', color='total_likes_posts',
            labels={'total_likes_posts': 'Likes nos posts', 'Página': 'Página'},
            color_continuous_scale='Peach'
        )
        # Engajamento total por página
        engagement_home = get_engagement_per_page(df, raw_data_home, top_n=10)
        fig_engagement_home = px.bar(
            engagement_home.sort_values('total_engagement', ascending=True),
            x='total_engagement', y='Página', orientation='h',
            title='Engajamento total por página', color='total_engagement',
            labels={'total_engagement': 'Engajamento total', 'Página': 'Página'},
            color_continuous_scale='Cividis'
        )
        # Apresentar os gráficos em colunas
        c9, c10 = st.columns(2)
        with c9:
            st.plotly_chart(fig_comments_home, use_container_width=True)
            st.caption("Total de comentários (incluindo respostas) gerados por cada página. Páginas no topo suscitam maior volume de debate.")
        with c10:
            st.plotly_chart(fig_likes_home, use_container_width=True)
            st.caption("Likes acumulados apenas nos posts de cada página. Indica popularidade directa dos conteúdos publicados.")
        # Engajamento ocupa linha inteira
        st.plotly_chart(fig_engagement_home, use_container_width=True)
        st.caption("Engajamento total combina publicações, likes nos posts, comentários (e respostas) e likes nos comentários. Páginas com mais interacções aparecem no topo.")

    elif page == "problema":
        st.title("Problema a ser resolvido")
        st.markdown(
            """
            Angola viveu, em Julho e Agosto de 2025, um período de manifestações motivadas pela subida do preço do petróleo, aumento da tarifa de táxi e outras questões socioeconómicas.

            **Objectivo do Pulsar Social Angolano**  
            Usar dados das redes sociais para obter informações, encontrar padrões e gerar insights que permitam compreender:
            - as **causas** e a **intensidade** da revolta;
            - **quem** são os actores mais influentes;
            - **quais** são os temas e reclamações mais discutidos;
            - as **tendências** de novas manifestações.

            Para isso, aplicamos três níveis de análise - **descritiva**, **diagnóstica** e **preditiva** - numa abordagem contínua de actualização.

            **Para quem é?**  
            Para **todas as pessoas** e organizações que desejam entender e medir a *pulsação* da sociedade angolana através de dados gerados nas redes sociais.
            """
        )

    elif page == "composicao":
        st.title("Composição do conjunto de dados")
        st.markdown("Resumo das principais características do dataset:")
        # Construir uma tabela de composição com métricas detalhadas.  Distinguimos
        # entre likes e comentários apenas nos posts e totais (posts + comentários + replies).
        composicao_rows = []
        composicao_rows.append({'Métrica': 'Redes sociais cobertas', 'Valor': 'Instagram'})
        composicao_rows.append({'Métrica': 'Total de posts', 'Valor': summary['n_posts']})
        composicao_rows.append({'Métrica': 'Total de páginas únicas', 'Valor': summary['n_pages']})
        composicao_rows.append({'Métrica': 'Período abrangido', 'Valor': f"{summary['period_start']} a {summary['period_end']}"})
        composicao_rows.append({'Métrica': 'Período analisado', 'Valor': f"{summary['period_start']} a {summary['period_end']}"})
        composicao_rows.append({'Métrica': 'Total de comentários (posts)', 'Valor': summary.get('n_comments_total_posts', 0)})
        composicao_rows.append({'Métrica': 'Total de comentários (posts + respostas)', 'Valor': summary.get('total_comments_all', 0)})
        composicao_rows.append({'Métrica': 'Total de likes nos posts', 'Valor': summary.get('total_likes_posts', 0)})
        composicao_rows.append({'Métrica': 'Total de likes (posts + comentários + respostas)', 'Valor': summary.get('total_likes_all', 0)})
        composicao_rows.append({'Métrica': 'Média de likes nos posts', 'Valor': f"{summary.get('avg_likes_posts', 0):.2f}"})
        composicao_rows.append({'Métrica': 'Média total de likes por post', 'Valor': f"{summary.get('avg_likes_all', 0):.2f}"})
        composicao_rows.append({'Métrica': 'Média de comentários por post (posts)', 'Valor': f"{summary.get('avg_comments_posts', 0):.2f}"})
        composicao_rows.append({'Métrica': 'Média total de comentários por post', 'Valor': f"{summary.get('avg_comments_all', 0):.2f}"})
        comp_df = pd.DataFrame(composicao_rows)
        comp_df['Valor'] = comp_df['Valor'].apply(lambda v: f"{v}")
        st.table(comp_df)
        # Mostrar estrutura
        with st.expander("Ver algumas linhas do dataset"):
            st.write(df[['pageName','datetime_dt','likes','num_comments','content']].head(10))

    elif page == "eda":
        st.title("Exploração de Dados (EDA)")
        st.markdown("Nesta secção pode explorar os dados através de diferentes filtros e gráficos.")
        # Filtros
        pages_selected = st.multiselect(
            "Filtrar por páginas", options=sorted(df['pageName'].unique()),
            default=[]
        )
        date_range = st.date_input(
            "Seleccionar intervalo de datas", value=(df['date'].min(), df['date'].max()),
            min_value=df['date'].min(), max_value=df['date'].max()
        )
        filtered_df = df.copy()
        if pages_selected:
            filtered_df = filtered_df[filtered_df['pageName'].isin(pages_selected)]
        start, end = date_range
        filtered_df = filtered_df[(filtered_df['date'] >= start) & (filtered_df['date'] <= end)]

        # Top páginas por número de posts
        top_counts = get_top_pages_posts(filtered_df).head(10)
        fig_counts = px.bar(
            top_counts,
            x='n_posts', y='page', orientation='h',
            title='Top páginas por número de posts', color='n_posts',
            labels={'n_posts': 'Nº de posts', 'page': 'Página'},
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_counts, use_container_width=True)

        # Top páginas por média de likes
        top_likes = get_top_pages_likes(filtered_df).head(10)
        fig_likes = px.bar(
            top_likes,
            x='avg_likes', y='page', orientation='h',
            title='Top páginas por média de likes nos posts', color='avg_likes',
            labels={'avg_likes': 'Média de likes nos posts', 'page': 'Página'},
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_likes, use_container_width=True)

        # Posts por dia
        posts_day = get_posts_per_day(filtered_df)
        fig_posts_day = px.line(
            posts_day,
            x='date', y='n_posts',
            title='Número de posts por dia',
            labels={'date': 'Data', 'n_posts': 'Nº de posts'}
        )
        st.plotly_chart(fig_posts_day, use_container_width=True)

        # Distribuição dos likes
        # Histograma dos likes com escala linear (remove log para evitar problemas de visualização)
        fig_likes_dist = px.histogram(
            filtered_df, x='likes', nbins=50,
            title='Distribuição de likes nos posts',
            labels={'likes': 'Likes nos posts', 'count': 'Número de posts'},
            color_discrete_sequence=['#636EFA']
        )
        st.plotly_chart(fig_likes_dist, use_container_width=True)

        # Total de comentários (inclui respostas) e total de likes por página
        st.subheader("Comentários e likes por página")
        raw_data_all = load_raw_data()
        comments_likes_df = get_comments_likes_per_page(df, raw_data_all)
        # Se houver filtro de páginas, restringir
        if pages_selected:
            comments_likes_df = comments_likes_df[comments_likes_df['Página'].isin([p.lower() for p in pages_selected])]
        # Top páginas por comentários
        comments_top = comments_likes_df.sort_values('total_comments', ascending=True).head(10)
        fig_comments = px.bar(
            comments_top,
            x='total_comments', y='Página', orientation='h',
            title='Total de comentários por página', color='total_comments',
            labels={'total_comments': 'Nº de comentários', 'Página': 'Página'},
            color_continuous_scale='IceFire'
        )
        st.plotly_chart(fig_comments, use_container_width=True)
        # Top páginas por likes
        likes_top = comments_likes_df.sort_values('total_likes_posts', ascending=True).head(10)
        fig_likes_page = px.bar(
            likes_top,
            x='total_likes_posts', y='Página', orientation='h',
            title='Total de likes (posts) por página', color='total_likes_posts',
            labels={'total_likes_posts': 'Total de likes', 'Página': 'Página'},
            color_continuous_scale='Peach'
        )
        st.plotly_chart(fig_likes_page, use_container_width=True)

    elif page == "diagnostica":
        st.title("Análises Diagnósticas")
        st.markdown("""Esta secção apresenta resultados da modelação de tópicos, sentimento, nuvem de palavras e análise de menções.""")

        # Informar sobre os tópicos e suas palavras-chave
        st.subheader("Descrição dos tópicos")
        topic_keywords = get_topic_keywords(df)
        # Apresentar cada tópico numa nova linha
        for idx in sorted(topic_keywords.keys()):
            words = ', '.join(topic_keywords[idx])
            st.markdown(f"**Tópico {idx}:** {words}")

        # Distribuição de tópicos (rótulos amigáveis)
        topic_counts = df['topic'].value_counts().sort_index()
        labels_topics = [f"Tópico {i}" for i in topic_counts.index]
        fig_topic = px.bar(
            x=labels_topics, y=topic_counts.values,
            labels={'x': 'Tópico', 'y': 'Nº de posts'},
            title='Distribuição de posts por tópico',
            color=topic_counts.values, color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_topic, use_container_width=True)

        # Engajamento médio por tópico (likes + comentários) com rótulos amigáveis
        df['total_engagement'] = df['likes'] + df['num_comments']
        eng = df.groupby('topic')['total_engagement'].mean().reset_index()
        eng['topic_label'] = eng['topic'].apply(lambda x: f"Tópico {x}")
        fig_eng = px.bar(
            eng, x='topic_label', y='total_engagement',
            labels={'topic_label': 'Tópico', 'total_engagement': 'Engajamento médio'},
            title='Engajamento médio por tópico', color='total_engagement',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_eng, use_container_width=True)

        # Palavras mais frequentes – incluindo comentários e replies
        st.subheader("Palavras mais frequentes (posts + comentários + replies)")
        raw_data = load_raw_data()
        words_df = get_top_words_all(df, raw_data, n=15)
        fig_words = px.bar(
            words_df.sort_values('Frequência', ascending=True),
            x='Frequência', y='Palavra', orientation='h',
            title='Top palavras mais frequentes', color='Frequência',
            color_continuous_scale='Purples'
        )
        st.plotly_chart(fig_words, use_container_width=True)

        # Distribuição de sentimentos (emoções) – categorias raiva, medo, tristeza, neutro
        st.subheader("Distribuição de emoções")
        raw_data = load_raw_data()
        emo_df = get_emotion_distribution_all(df, raw_data)
        fig_emo = px.bar(
            emo_df, x='Emoção', y='Contagem', color='Emoção',
            title='Distribuição de emoções (posts, comentários e respostas)',
            labels={'Contagem': 'Número de textos'}
        )
        st.plotly_chart(fig_emo, use_container_width=True)

        # Menções: gerar dados e gráficos dinâmicos
        st.subheader("Top utilizadores que mencionam outros")
        mentioners_df, mentioned_df = get_mentions_summary(raw_data)
        fig_out = px.bar(
            mentioners_df.sort_values('Menções feitas'),
            x='Menções feitas', y='Utilizador', orientation='h',
            title='Top utilizadores que mencionam outros', color='Menções feitas',
            color_continuous_scale='Teal'
        )
        st.plotly_chart(fig_out, use_container_width=True)

        st.subheader("Top utilizadores mencionados")
        fig_in = px.bar(
            mentioned_df.sort_values('Menções recebidas'),
            x='Menções recebidas', y='Utilizador', orientation='h',
            title='Top utilizadores mencionados', color='Menções recebidas',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig_in, use_container_width=True)

        # Rede de interacções: páginas vs. comentadores
        st.subheader("Rede de interacções – páginas e comentadores")
        pages_df, users_df = get_interaction_network(raw_data, top_n=20)
        fig_pages = px.bar(
            pages_df.sort_values('Comentadores únicos'),
            x='Comentadores únicos', y='Página', orientation='h',
            title='Páginas com mais comentadores únicos', color='Comentadores únicos',
            color_continuous_scale='Blugrn'
        )
        st.plotly_chart(fig_pages, use_container_width=True)
        fig_users = px.bar(
            users_df.sort_values('Páginas comentadas'),
            x='Páginas comentadas', y='Utilizador', orientation='h',
            title='Utilizadores que comentam em mais páginas', color='Páginas comentadas',
            color_continuous_scale='Sunset'
        )
        st.plotly_chart(fig_users, use_container_width=True)

        # Entidades mais mencionadas
        st.subheader("Entidades mais mencionadas")
        entidades = ['presidente','governo','polícia','policia','mpla','taxistas','joão lourenço','ministro','ministério','unita']
        ent_df = get_entity_counts(df, raw_data, entidades)
        fig_ent = px.bar(
            ent_df.sort_values('Ocorrências'),
            x='Ocorrências', y='Entidade', orientation='h',
            title='Frequência de entidades nos textos', color='Ocorrências',
            color_continuous_scale='Inferno'
        )
        st.plotly_chart(fig_ent, use_container_width=True)

        # Engajamento por página
        st.subheader("Engajamento por página")
        engagement_df = get_engagement_per_page(df, raw_data, top_n=10)
        fig_eng_page = px.bar(
            engagement_df.sort_values('total_engagement'),
            x='total_engagement', y='Página', orientation='h',
            title='Engajamento total por página', color='total_engagement',
            labels={'total_engagement': 'Engajamento total', 'Página': 'Página'},
            color_continuous_scale='Cividis'
        )
        st.plotly_chart(fig_eng_page, use_container_width=True)

        # Nuvem de palavras (dinâmica com Plotly)
        st.subheader("Nuvem de palavras")
        cloud_fig = generate_wordcloud_figure(df, raw_data, max_words=40)
        if cloud_fig:
            st.plotly_chart(cloud_fig, use_container_width=True)
        else:
            st.info("Não foi possível gerar a nuvem de palavras.")

        # Top posts por engajamento
        st.subheader("Top posts com maior engajamento")
        top_posts_df = get_top_posts_engagement(df, raw_data, top_n=5)
        st.table(top_posts_df)

        # Top comentários por engajamento
        st.subheader("Top comentários com maior engajamento")
        top_comments_df = get_top_comments_engagement(raw_data, top_n=5)
        st.table(top_comments_df)

    elif page == "preditiva":
        st.title("Modelos Preditivos")
        st.warning("Os modelos preditivos ainda estão em desenvolvimento e serão adicionados nesta secção assim que estiverem prontos.")

    elif page == "resultados":
        st.title("Resultados e Insights")
        st.markdown(
            """
            **Principais descobertas até agora:**

            - **Engajamento:** A média de likes por post é de ~1 674 (sem likes dos comentaários) e de ~2036.81 (incluindo likes dos comentários). A maior parte dos posts recebe poucos likes e comentários, enquanto um pequeno número de publicações torna‑se viral, especialmente em páginas como **“XÉ Agora Aguenta”**.
            - **Páginas influentes:** “XÉ Agora Aguenta” lidera em média de likes, enquanto “AngoRussia” e “ANGO PORTAL” contribuem com mais publicações.
            - **Picos temporais:** O período de 23 a 31 de Julho de 2025 apresenta o maior número de posts, coincidindo com protestos sobre preços de combustíveis e transporte.
            - **Temas predominantes:** Tópicos associados a manifestações, polícia e taxistas têm maior engajamento.
            - **Rede de interacções:** Algumas páginas concentram a maioria dos comentadores e certos utilizadores actuam como pontes, mencionando várias contas.
            - **Sentimentos:** A maioria das mensagens é neutra, mas existe parcela significativa de publicações negativas, reflectindo descontentamento.
            """
        )

    elif page == "metodologia":
        st.title("Metodologia – CRISP-DM adaptado")
        st.markdown(
            """
            **1. Compreensão do negócio** – Definição do problema, identificação de stakeholders e questões de interesse.

            **2. Compreensão dos dados** – Inspecção do dataset, identificação de duplicados, valores em falta e estrutura dos campos.

            **3. Preparação dos dados** – Limpeza e enriquecimento: normalização de datas, extração de hashtags, menções, emojis, e criação de variáveis como contagem de palavras e engagement.

            **4. Análise Descritiva e Diagnóstica** – Estatísticas básicas, séries temporais, modelação de tópicos, análise de sentimento, nuvens de palavras e redes de interacção.

            **5. Modelagem Preditiva (fase futura)** – Definição de problemas de previsão, preparação avançada de features e treino de modelos (serão integrados posteriormente).

            **6. Avaliação e Interpretação** – Comparação de modelos, interpretação de resultados e geração de insights.

            **7. Implementação** – Construção desta aplicação para disponibilizar os resultados de forma interativa e planeamento de actualizações contínuas.
            """
        )

    elif page == "recomendacoes":
        st.title("Recomendações")
        st.markdown(
            """
            *(Esta secção será preenchida posteriormente com recomendações baseadas nos resultados das análises.)*
            """
        )

    elif PAGES[page] == "recomendacoes":
        st.title("Recomendações")
        st.markdown(
            """
            *(Esta secção será preenchida posteriormente com recomendações baseadas nos resultados das análises.)*
            """
        )

if __name__ == '__main__':
    main()
