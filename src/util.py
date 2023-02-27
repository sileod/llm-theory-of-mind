import dropbox

import censusname
from easydict import EasyDict as edict

import requests
import urllib.parse
import re

# --- Dropbox ---

def to_dropbox(dataframe, path, token):
    dbx = dropbox.Dropbox(token)

    df_string = dataframe.to_json(orient='records', lines=True)
    db_bytes = bytes(df_string, 'utf8')
    
    dbx.files_upload(
        f=db_bytes,
        path=path,
        mode=dropbox.files.WriteMode.overwrite
    )

def get_url_of_file(path, token):
    dbx = dropbox.Dropbox(token)
    url = dbx.sharing_create_shared_link(f'{path}').url
    return url

# --- Name generation ---

def random_first_name():
    return censusname.generate(nameformat='{given}')

def n_random_first_names(n):
    names = []
    for _ in range(n):
        name = random_first_name()
        while name in names:
            name = random_first_name()
        names.append(name)
    return names

# --- Problem solving ---

TAUTOLOGY = '\\top'
ENTAILMENT = '\\vDash'
NOT_ENTAILMENT = '\\not\\vDash'

def parse_response(response):
    result = re.search(r'(?s)<p>.*</p>', response).group(0)

    if TAUTOLOGY in result:
        return -1
    elif NOT_ENTAILMENT in result:
        return 0
    elif ENTAILMENT in result:
        return 1
    return -2


def class_to_dict(x):
    name=x.__name__
    x=edict({a:getattr(x,a) for a in dir(x) if not a.startswith('__')})
    x['name']=name
    return x


def solve(problem):
    headers = {
        'authority': 'w4eg.de',
        'accept': '*/*',
        'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,en-US;q=0.7,de;q=0.6',
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'origin': 'https://w4eg.de',
        'referer': 'https://w4eg.de/malvin/illc/smcdelweb/index.html',
        'sec-ch-ua': '"Not_A Brand";v="99", "Google Chrome";v="109", "Chromium";v="109"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest',
    }

    data = 'smcinput=' + urllib.parse.quote(problem)

    response = requests.post('https://w4eg.de/malvin/illc/smcdelweb/check', headers=headers, data=data)

    label = parse_response(response.text)

    return label