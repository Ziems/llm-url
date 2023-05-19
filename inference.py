import openai
import os
import time
import json
import _thread
import requests

import threading

from tqdm import tqdm
from retry import retry

from contextlib import contextmanager
from collections import defaultdict

import re
link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)

openai.api_key = ""

class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=''):

    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


def extract_urls(string):
    def clean(url):
        if url[-1] == '.':
            return url[:-1]
        return url
    r = re.findall(link_regex, string)
    return [clean(l[0]) for l in r]

def extract_topic(url):
    url = url.split('#')[0]
    return url.split('/')[-1]

@retry(Exception, tries=3, delay=10)
def fetch_pages(titles, debug=False):
    if titles == ['']:
        return {'pages': [], 'titles': []}
    if debug:
        print(titles)
    api_url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'titles': '|'.join(titles),
        'format': 'json',
        'prop': 'revisions',
        'rvslots': '*',
        'rvprop': 'content',
        'redirects': ''
    }
    response = requests.get(api_url, params=params, timeout=20)
    data = response.json()

    ret_titles = []
    ret_pages = []

    if 'warnings' in data or 'query' not in data:
        print("data warning")
        print(data)
        return {'pages': ret_pages, 'titles': ret_titles}

    for k in data['query']['pages'].keys():
        page_data = data['query']['pages'][k]
        if 'missing' not in page_data and 'revisions' in page_data:
            title = page_data['title']
            page = page_data['revisions'][0]['slots']['main']['*']
            page = page.replace("\n", " ").replace("\t", " ")
            page = ' '.join(page.split())[:10_000]
            ret_pages.append(page)
            ret_titles.append(title)
    return {'pages': ret_pages, 'titles': ret_titles}


def add_prompt(item, prompt):

    def rmreturn(s):
        s = s.replace('\n\n', ' ')
        s = s.replace('\n', ' ')
        return s.strip()

    query = item['question']
    prompt = prompt.replace('{query}', query)

    if '{top_passages_concat}' in prompt:
        if item.get('top_passages_concat'): # background info
            passages = rmreturn(item['top_passages_concat'][0])
            prompt = prompt.replace('{top_passages_concat}', passages)
    elif '{background}' in prompt:
        if item.get('output'): # background info
            backinfo = rmreturn(item['output'][0])
            prompt = prompt.replace('{background}', backinfo)

    return ' '.join(prompt.split(' ')) # max 1000 words


def run_embeddings(input_text, engine='text-similarity-davinci-001'):

    texts = [t.replace('\n', '') for t in input_text]
    outputs = openai.Embedding.create(input=texts, model=engine)['data']
    embeddings = [o['embedding'] for o in outputs]

    return embeddings

class OAIOverloadedException(Exception):
    "Raised when the OpenAI servers are overloaded."
    pass

@retry((OAIOverloadedException, Exception), tries=50, delay=10)
def openai_request(inputs_with_prompts, engine, max_tokens, num_sequence=1, temp=0):
    headers = {
        'Content-Type': "application/json",
        'Authorization': "Bearer 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"
    }
    params = {
        "model": engine,
        "prompt": inputs_with_prompts,
        "max_tokens": max_tokens,
        "temperature": temp,
        "n": num_sequence,
    }
    response = requests.post("https://api.openai.com/v1/completions", json=params, headers=headers, timeout=20*3)
    data = response.json()
    if 'choices' not in data:
        print("Error: ", data['error'])
        raise OAIOverloadedException
    if 'error' in data:
        print("Error: ", data['error'])
        raise OAIOverloadedException
    return [c["text"] for c in data["choices"]]

def run_main(inlines, outfile, engine, prompt, max_tokens, n=1, temp=0, parse_url=True, filter_docs=False):

    if os.path.exists(outfile):
        outs = open(outfile, 'a', encoding='utf8')
        num_lines = len(open(outfile, 'r').readlines())
        inlines = inlines[num_lines - 1: ]
    else: # not os.path.exists(outfile)
        outs = open(outfile, 'a', encoding='utf8')
        outs.write(json.dumps({"prompt": prompt}) + '\n')

    pbar = tqdm(total = len(inlines))
    index = 0
    pbar.update(index)
    while index < len(inlines):
        inputs, answers = [], []
        inputs_with_prompts = []
        for _ in range(5): # default 20
            if index >= len(inlines): break
            input_with_prompt = add_prompt(inlines[index], prompt)
            inputs.append(inlines[index]['question']) ## a string
            answers.append(inlines[index]['answer']) ## a list of strings
            inputs_with_prompts.append(input_with_prompt)
            index += 1
        if parse_url:
            samples = defaultdict(list)
            url_responses = defaultdict(list)
            extracted_topics = defaultdict(list)
            concat_pages = defaultdict(list)
            fetched_page_titles = defaultdict(list)
            fetched_page_texts = defaultdict(list)

            outputs = openai_request(inputs_with_prompts, 
                engine, max_tokens, n, temp)
            for j, output in enumerate(outputs):
                samples[j//n].append(output)

                extracted_url_set = extract_urls(output)
                extracted_topic_set = [extract_topic(url) for url in extracted_url_set]

                fetched_page_set = fetch_pages(extracted_topic_set)
                _fetched_page_texts = fetched_page_set['pages']
                _fetched_page_titles = fetched_page_set['titles']


                url_responses[j//n].append(extracted_url_set)
                extracted_topics[j//n].append(extracted_topic_set)
                concat_pages[j//n].append(' '.join(_fetched_page_texts))
                fetched_page_titles[j//n].append(_fetched_page_titles)
                fetched_page_texts[j//n].append(_fetched_page_texts)

            for i in range(len(inputs_with_prompts)):
                outs.write(json.dumps({
                    'question': inputs[i], 
                    'answer': answers[i],
                    'gpt3_response': output,
                    'url_response': url_responses[i],
                    'extracted_topic': extracted_topics[i][0],
                    'output': concat_pages[i],
                    'fetched_page_titles': fetched_page_titles[i],
                    'fetched_page_texts': fetched_page_texts[i]
                    }) 
                    +'\n')
        else:
            samples = defaultdict(list)
            outputs = openai_request(inputs_with_prompts, 
                engine, max_tokens, n, temp)
            for j, output in enumerate(outputs):
                samples[j//n].append(output)

            for i in range(len(inputs_with_prompts)):
                outs.write(json.dumps({
                    'question': inputs[i], 
                    'answer': answers[i], 
                    'output': samples[i]}) 
                    +'\n')

        pbar.update(len(inputs_with_prompts))

    pbar.close()
    outs.close()
