import  requests
from bs4 import BeautifulSoup
import pandas as pd

url =['https://www.reuters.com/markets/companies/AAPL.OQ','https://www.reuters.com/markets/companies/AMZN.OQ/']
member =[]

for a in url:
    executive =[]
    response = requests.get(a).text
    soup = BeautifulSoup(response, 'html5lib')

    search = soup.find('div',{"class":"about-company-card__company-leadership__1mNWX"})
    for dt,dd in zip(search.find_all('dt'),search.find_all('dd')):
        executive.append((dt.text.strip(),dd.text.strip()))

    member.append(executive)

c= 1
for name in member:
    df = pd.DataFrame(name,columns=['Name','Job_Title'])
    print(df)
    df.to_csv(str(c)+'.csv',index=False)
    c += 1

print("Data scraped ")
