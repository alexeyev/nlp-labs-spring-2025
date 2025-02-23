import time

from flask import Flask, request, render_template_string

from whoosh_searcher import WhooshEngine

app = Flask(__name__)
engine = WhooshEngine()

if not engine.ix:
    engine.index()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Yet Another Search Engine</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px; 
        }
        .search-box { text-align: center; margin: 50px 0; }
        input[type="text"] { 
            width: 60%; padding: 12px 20px; margin: 8px 0;
            border: 2px solid #ddd; border-radius: 24px; font-size: 16px;
        }
        input[type="submit"] { 
            background: #f8f9fa; border: 1px solid #dadce0; padding: 10px 20px;
            border-radius: 24px; cursor: pointer; margin-left: 10px;
        }
        .results { margin-top: 30px; }
        .result { margin-bottom: 30px; }
        .title { font-size: 20px; color: #1a0dab; text-decoration: none; }
        .url { color: #006621; font-size: 14px; margin: 5px 0; }
        .snippet { color: #545454; line-height: 1.5; }
        .stats { color: #70757a; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="search-box">
        <h1 style="color:#ADD8E6; font-family: 'Comic Sans MS', 'Comic Sans';">
            Yet Another Web Search Engine
        </h1>
        <form method="POST">
            <input type="text" name="query" value="{{ query }}" autofocus>
            <input type="submit" value="Search">
        </form>
    </div>

    {% if query %}
    <div class="stats">
        About {{ results|length }} results ({{ "%.2f"|format(time) }} seconds)
    </div>
    {% endif %}

    <div class="results">
        {% for result in results %}
        <div class="result">
            <div class="title">{{ result.title }}</div>
            <div class="url">doc{{ result.id }}</div>
            <div class="snippet">{{ result.text[:250] }}{% if result.text|length > 250 %}...{% endif %}</div>
        </div>
        {% else %}
        {% if query %}
        <div class="no-results">
            <h3>No results found for "{{ query }}"</h3>
            <p>Suggestions:</p>
            <ul>
                <li>Make sure all words are spelled correctly</li>
                <li>Try different keywords</li>
                <li>Try more general keywords</li>
            </ul>
        </div>
        {% endif %}
        {% endfor %}
    </div>
</body>
</html>
'''


@app.route('/', methods=['GET', 'POST'])
def search():
    query, results, search_time = '', [], 0.0

    if request.method == 'POST':
        query = request.form.get('query', '').strip()

        if query:
            start_time = time.time()
            try:
                results = engine.search(query)
            except Exception as e:
                app.logger.error(f"Search error: {e}")
                results = []

            search_time = time.time() - start_time

    return render_template_string(
        HTML_TEMPLATE,
        query=query,
        results=results,
        time=search_time
    )


if __name__ == '__main__':
    app.run(debug=True)
