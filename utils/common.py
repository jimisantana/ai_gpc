import requests
from os import system, path

from utils.ftp import download as ftp_download


def postt(url='http://localhost:4891/read', json_obj: dict = None):

    r = requests.post(url=url, params=json_obj)
    return r.json()


def fread(fpath:str='dump/dump.xml') -> str:

    txt = None
    with open(fpath ,'r', encoding='utf-8') as f:
        txt = f.read()

    return txt
