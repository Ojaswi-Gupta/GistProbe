# # import requests
# # import pandas as pd
# # from bs4 import BeautifulSoup
# # import sys

# # url = sys.argv[1]

# # headers = {
# #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
# #     "Accept-Language": "en-US,en;q=0.9",
# #     "Accept-Encoding": "gzip, deflate, br",
# #     "Connection": "keep-alive"
# # }

# # try:
# #     response = requests.get(url, headers=headers, timeout=10)

# #     # ✅ Correct place
# #     print("Fetching:", url)
# #     print("Status Code:", response.status_code)

# #     soup = BeautifulSoup(response.text, "html.parser")

# #     data = []

# #     # Extract headings
# #     for tag in soup.find_all(["h1", "h2", "h3"]):
# #         text = tag.get_text().strip()
# #         if len(text) > 15:
# #             data.append(text)

# #     # Extract paragraphs if headings empty
# #     if len(data) == 0:
# #         for tag in soup.find_all("p"):
# #             text = tag.get_text().strip()
# #             if len(text) > 40:
# #                 data.append(text)

# #     df = pd.DataFrame(data, columns=["text"])

# #     ignore_words = ["login", "subscribe", "advertisement", "cookie"]

# #     for tag in soup.find_all(["h1","h2","h3"]):
# #         text = tag.get_text().strip()

# #         if len(text) > 15 and not any(word in text.lower() for word in ignore_words):
# #             data.append(text)

# #     print(df.head())

# #     df.to_csv("data.csv", index=False)

# #     print("Data saved to data.csv")



# # except Exception as e:
# #     print("Error:", e)


# import requests
# import pandas as pd
# from bs4 import BeautifulSoup
# import sys

# url = sys.argv[1]

# headers = {
#     "User-Agent": "Mozilla/5.0",
# }

# try:
#     response = requests.get(url, headers=headers, timeout=10)

#     print("Fetching:", url)
#     print("Status Code:", response.status_code)

#     soup = BeautifulSoup(response.text, "html.parser")

#     data = []
#     ignore_words = ["login", "subscribe", "advertisement", "cookie"]

#     # Extract headings
#     for tag in soup.find_all(["h1", "h2", "h3"]):
#         text = tag.get_text().strip()

#         if len(text) > 15 and not any(word in text.lower() for word in ignore_words):
#             data.append(text)

#     # Extract paragraphs if headings empty
#     if len(data) == 0:
#         for tag in soup.find_all("p"):
#             text = tag.get_text().strip()
#             if len(text) > 40:
#                 data.append(text)

#     df = pd.DataFrame(data, columns=["text"])

#     print(df.head())

#     df.to_csv("data.csv", index=False)

#     print("Data saved to data.csv")

# except Exception as e:
#     print("Error:", e)




import requests
import pandas as pd
from bs4 import BeautifulSoup

def scrape_url(url):
    """Phase 1: Crawling. Fetches raw headlines from the web."""
    # headers = {"User-Agent": "Mozilla/5.0"}
    # In crawler.py
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}


    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        # Terminal Logs
        print(f"Fetching: {url}")
        print(f"Status Code: {response.status_code}")
        
        soup = BeautifulSoup(response.text, "html.parser")
        data = []
        ignore_words = ["login", "subscribe", "advertisement", "cookie"]

        # Target headings for higher semantic value
        for tag in soup.find_all(["h1", "h2", "h3"]):
            text = tag.get_text().strip()
            if len(text) > 15 and not any(word in text.lower() for word in ignore_words):
                data.append(text)

        # Fallback to paragraphs if headings are sparse
        if len(data) == 0:
            for tag in soup.find_all("p"):
                text = tag.get_text().strip()
                if len(text) > 40:
                    data.append(text)

        df = pd.DataFrame(data, columns=["text"])
        
        # Terminal Logs
        print(df.head())
        print("Data saved to data.csv")
        
        return df

    except Exception as e:
        print(f"Error during crawl: {e}")
        return pd.DataFrame(columns=["text"])