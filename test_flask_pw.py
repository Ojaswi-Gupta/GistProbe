from flask import Flask
from crawler import scrape_url

app = Flask(__name__)

@app.route("/")
def test():
    try:
        df = scrape_url('https://medium.com')
        return f"Success: {len(df)} items"
    except Exception as e:
        import traceback
        return traceback.format_exc()

if __name__ == "__main__":
    app.run(port=5001)
