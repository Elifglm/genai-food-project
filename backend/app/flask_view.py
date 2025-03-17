from flask import Flask, render_template_string
from chromadb import PersistentClient

# Flask uygulaması
app = Flask(__name__)

# ChromaDB Ayarları
persist_path = "./chromadb"
client = PersistentClient(persist_path)
collection = client.get_or_create_collection(name="user_preferences")

@app.route("/")
def view_data():
    # ChromaDB'deki verileri alın
    results = collection.get()
    rows = [
        {
            "Document": doc,
            "User ID": meta.get("user_id", ""),
            "Category": meta.get("category", ""),
            "Text": meta.get("text", ""),
            "Like Status": meta.get("like_status", ""),
            "Cuisine": meta.get("cuisine", ""),
            "Servings": meta.get("servings", "")
        }
        for doc, meta in zip(results['documents'], results['metadatas'])
    ]

    # HTML Şablonu
    html_template = """
    <html>
    <head>
        <title>ChromaDB Verileri</title>
        <style>
            table {
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid black;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>
        <h1>ChromaDB Verileri</h1>
        <table>
            <tr>
                <th>Document</th>
                <th>User ID</th>
                <th>Category</th>
                <th>Text</th>
                <th>Like Status</th>
                <th>Cuisine</th>
                <th>Servings</th>
            </tr>
            {% for row in rows %}
            <tr>
                <td>{{ row['Document'] }}</td>
                <td>{{ row['User ID'] }}</td>
                <td>{{ row['Category'] }}</td>
                <td>{{ row['Text'] }}</td>
                <td>{{ row['Like Status'] }}</td>
                <td>{{ row['Cuisine'] }}</td>
                <td>{{ row['Servings'] }}</td>
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    """
    return render_template_string(html_template, rows=rows)

if __name__ == "__main__":
    app.run(debug=True)
