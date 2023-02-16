import requests
import urllib.parse
import re

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

def solve(problem):
    print(problem)
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