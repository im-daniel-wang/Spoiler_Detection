__Machine Learning Approaches To Detect Spoiler Reviews__

To access the API:
```
python api.py
```
In another terminal:
```
python api_call.py
```

- Example
  - Input: {'review_sentences':['He died in the last episode.', 'I really enjoy this show.']}
  - Output: {'reviews': ['He died in the last episode.', 'I really enjoy this show.'], 'Class': ['Spoiler', 'Non-Spoiler']}



Note: vectorizer.pkl (126 MB) and svm.sav (22.9 MB) are not pushed here. 