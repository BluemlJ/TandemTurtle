"""
Script to download and extract the data from https://www.bughouse-db.org/dl/
"""
import requests
from lxml import html
import wget
import bz2
import os

url = "https://www.bughouse-db.org/dl/"
dir = "data/"

page = requests.get(url)
webpage = html.fromstring(page.content)


for link in webpage.xpath('//a/@href'):
    if ".bz2" in link:
        print("downloading  %s"%url + link)
        file_name = dir + link
        wget.download(url+link,file_name)
        print("")


        new_file_name = file_name[:-4]
        print("extracting to %s "%new_file_name)
        with open(new_file_name, 'wb') as new_file, bz2.BZ2File(file_name, 'rb') as file:
            for data in iter(lambda: file.read(100 * 1024), b''):
                new_file.write(data)

        print("removing to %s: " % file_name)
        os.remove(file_name)

