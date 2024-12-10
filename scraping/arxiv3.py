from bs4 import BeautifulSoup
from bs4.element import Tag
import requests
import math
import numpy as np
import pandas as pd
from time import sleep
import os
from urllib.parse import urljoin
field=['astro-ph','math','cs','q-bio','q-fin','stat','eess','econ']
yr=['2018','2019','2020','2021','2022','2023']

url="https://arxiv.org"
Columns = [
     'Year',
     'Field',
     'Title',
     'Authors',
     'Abstract',
     'DOI']
df = pd.DataFrame(columns=Columns)
for y in yr:
     for fi in field:
        fn = '_'.join([fi,y])
        count=0
        with open(f"links\\{fn}") as f:
            for l in f:
                tries = 0
                finished = False
                while not finished and tries < 3:
                    try:
                        link = urljoin(url,l)
                        response = requests.get(link)
                        soup = BeautifulSoup(response.text,'xml')
                        if response.status_code == 429:
                            print("Sleeping")
                            sleep(30)
                            tries += 1
                            continue
                        print(link,soup.body)
                        content = soup.body.select('.leftcolumn #content-inner #abs')[0]
                        title = content.h1.get_text().strip('Title:')
                        a = content.select('.authors')[0].find_all('a')
                        authors = [author.string for author in a]
                        abstract = content.select('.abstract')[0].get_text().strip('\nAbstract: \n')
                        doi = content.select('.metatable')[0].table.select('.arxivdoi')[0].find('a').get('href')
                        new_row = {'Year':y,'Field':fi,'Title':title,'Authors':authors,'Abstract':abstract,'DOI':doi}
                        print(new_row)
                        count+=1
                        print(count)
                        df.loc[len(df)] = new_row
                        finished=True
                    except:
                        sleep(10)
                    else:
                        finished = True
                        

df.to_csv('output.csv',index=True)

# count=0
# with open(r"links\q-fin_2021","r") as f:
#          for l in f:
#              if count==1:
#                     continue
#              link = urljoin(url,l)
#              response = requests.get(link)
#              soup = BeautifulSoup(response.text,'xml')
#              content = soup.body.div.main.select('.leftcolumn #content-inner #abs')[0]
#              title = content.h1.get_text().strip('Title:')
#              a = content.select('.authors')[0].find_all('a')
#              authors = [author.string for author in a]
#              abstract = content.select('.abstract')[0].get_text().strip('\nAbstract: \n')
#              doi = content.select('.metatable')[0].table.select('.arxivdoi')[0].find('a').get('href')
#              count+=1

