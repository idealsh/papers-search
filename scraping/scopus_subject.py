from bs4 import BeautifulSoup
from bs4.element import Tag

with open('List of Subject Areas Covered By Scopus Database - iLovePhD.html',encoding='utf8') as f:
        html=f.read()

soup = BeautifulSoup(html,"lxml")
parent = soup.select('.wp-block-table')[0].find('table').find('tbody').find_all('tr')
mydict ={}
count=0
for child in parent:
    count+=1
    if count==1:
          pass
    else:
        td = child.find_all('td')
        key = td[0].string.replace('00','')
        subject_area=td[1].text
        subject_area=" ".join(subject_area.split())
        mydict[key]=subject_area
print(mydict)

parent2 = soup.select('.wp-block-table')[1].find('table').find('tbody').find_all('tr')
mydict2 ={}
count2=0
for child in parent2:
    count2+=1
    if count2==1:
          pass
    else:
        td = child.find_all('td')
        key = td[0].string
        key = " ".join(key.split())
        if(key!=''):
            subject_area=td[1].text
            subject_area=" ".join(subject_area.split())
            mydict2[key]=subject_area
print(mydict2)


