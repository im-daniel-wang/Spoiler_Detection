import requests


# Chenge the json text to test different api calls
response=requests.post("http://localhost:5000/classify", json={'review_sentences':['He died in the last episode.', 'I really enjoy this show.']})
print(response.json())