import requests
import json


def prompt_propcess(text): 
    num = ['1st', '2nd', '3rd']
    prompt_text = ''
    c = 0
    s = ''
    for ch in text:
        if ch == '|':
            # prompt_text += 'The ' + (num[c] if c <= 2 else str(c) + 'th') + ' diagnosis is {' + s + '}. '
            if c == 0:
                prompt_text += 'Most importantly, the 1st diagnosis is {' + s + '}.'
            else:
                prompt_text += 'As a supplementary condition, the ' + (num[c] if c <= 2 else str(c + 1) + 'th') + ' diagnosis is {' + s + '}.'
            c += 1
            s = ''
        else:
            s += ch
    if s != '':
        if c == 0:
            prompt_text += 'Most importantly, the 1st diagnosis is {' + s + '}.'
        else:
            prompt_text += 'As a supplementary condition, the ' + (num[c] if c <= 2 else str(c + 1) + 'th') + ' diagnosis is {' + s + '}.'
        c += 1
        s = ''

    # print(prompt_text)
    return prompt_text 

def get_text_embedding(text): 
    # return [len_text_embedding] 

    url = "https://kidneytalk.bjmu.edu.cn/api/v1/embeddings"
    # Headers
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"
    }

    # Data payload
    text = prompt_propcess(text)
    data = {
        "input": text,
        "model": "E-72B",
        "encoding_format": "float",
        # "dimensions": 1536
    }

    # Sending the POST request
    response = requests.post(url, headers=headers, json=data)

    # Check and print the response
    if response.status_code == 200:
        response_list = response.json()['data']
        embedding = response_list[0]['embedding']
        return embedding
    else:
        print(f"Error: {response.status_code} - {response.text}")
        raise ConnectionError
