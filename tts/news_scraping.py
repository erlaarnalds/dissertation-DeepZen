from bs4 import BeautifulSoup
import requests
import csv


def get_quote(text):
    start = [pos for pos, char in enumerate(text) if char == "„"]
    end = [pos for pos, char in enumerate(text) if char == "“"]

    quotes = []

    start_ind = 0
    end_ind = 0

    while end_ind < len(end) and start_ind < len(start) and end[end_ind] < len(text):
        quote_start = start[start_ind]
        quote_end = end[end_ind]

        if end_ind+1 < len(end) and end_ind > 0 and start_ind+1 < len(start):
            while end[end_ind+1] < start[start_ind+1]:
                end_ind += 1
                quote_end = end[end_ind]

            while quote_start < end[end_ind-1]:
                start_ind += 1
                quote_start = start[start_ind]

        quote = text[quote_start+1:quote_end]

        if len(quote.split(" ")) > 5:
            quotes.append(quote)
        end_ind += 1
        start_ind += 1
        

    return quotes


id_start = "20242602"

with open("visir_quotes.txt", "w") as file:
    writer = csv.DictWriter(file, fieldnames=["news_id", "quote"])
    writer.writeheader()

    for id in range(1000):
        print(id)
        id_end = "0" * (3 - len(str(id))) + str(id)
        news_id = id_start + id_end
        url = f"https://www.visir.is/g/{news_id}d/"
        page_to_scrape = requests.get(url)

        if page_to_scrape.status_code == 200:
            soup = BeautifulSoup(page_to_scrape.text, "html.parser")

            article = soup.find_all('div', attrs={"data-element-label": "Meginmál"})


            sentences = article[0].findAll('p')
            
            for line in sentences:
                quotes = get_quote(line.text)

                for quote in quotes:
                    writer.writerow({"news_id": news_id, "quote": quote})


