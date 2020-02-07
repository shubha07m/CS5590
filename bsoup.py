from bs4 import BeautifulSoup as bs
import requests

# with open('simple.html') as html_file:
#    soup = bs(html_file, 'lxml')

myhtml = requests.get("https://en.wikipedia.org/wiki/Deep_learning")
soup = bs(myhtml.content, "html.parser")
print(soup.title)
x = (soup.find_all('a'))
for i in x:
    print(i.get('href'))
