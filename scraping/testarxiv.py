from bs4 import BeautifulSoup
from bs4.element import Tag
import requests

with open('arXiv.org e-Print archive.html',encoding='utf8') as f:
        html=f.read()
soup = BeautifulSoup(html,"xml")
fields = soup.find_all('ul')[1:9]
field=['astro-ph','math','cs','q-bio','q-fin','stat','eess','econ']
yr=['2018','2019','2020','2021','2022','2023']
url="https://arxiv.org/list?archive={}&year={}&submit=Go"
NoPerfield={}
sumperyear={}
for y in yr:
    sum=0
    for f in field:
        link =  url.format(*(f,y))
        response = requests.get(link)
        soup = BeautifulSoup(response.text,'xml')
        a=soup.body.div.header.main.div.div.div.select('.paging')[0]
        total = ''.join([child for child in a.children if isinstance(child, str)]).strip('Total of entries : \n')
        NoPerfield[' '.join([f,y])]=int(total)
        sum+=int(total)
        print('.')
    sumperyear[y]=sum
print(NoPerfield)
print(sumperyear)
    
ratioPerYearPerField={}
for y in sumperyear:
    for f in NoPerfield:
        ratioPerYearPerField[f]=NoPerfield[f]/sumperyear[y]
print(ratioPerYearPerField)

