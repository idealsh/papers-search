from bs4 import BeautifulSoup
from bs4.element import Tag
import requests
import math
import numpy as np
from time import sleep

with open('arXiv.org e-Print archive.html',encoding='utf8') as f:
        html=f.read()
soup = BeautifulSoup(html,"xml")
fields = soup.find_all('ul')[1:9]
field=['astro-ph','math','cs','q-bio','q-fin','stat','eess','econ']
yr=['2020','2021','2022','2023']
url="https://arxiv.org/list/{}/{}?skip={}&show=25"

from subject_data import ratio_per_yr_field,sum_per_yr,no_per_field
papers={}
for y in yr:
    per_field={}
    for f in field:
        if(f=='astro-ph' and y=='2018'):
             continue
        number_to_scrape = math.floor(ratio_per_yr_field[' '.join([f,y])]*1000)
        index = np.random.randint(0,no_per_field[' '.join([f,y])],number_to_scrape)
        temp=[]
        for i in index:
            tries = 0
            finished = False

            while not finished and tries < 3:
                link = url.format(*(f,y,i))
                print(i,y,f,link)
                response = requests.get(link)
                soup = BeautifulSoup(response.text,'xml')
                if response.status_code == 429:
                    print("Sleeping")
                    sleep(30)
                    tries += 1
                    continue
                item = soup.body.div.main.dt.find_all('a')[1].get('href')
                print(item)
                temp.append(item)
                finished = True
        with open(f'{f}_{y}',"w") as f:
             for i in temp:
                  f.write(i+"\n")