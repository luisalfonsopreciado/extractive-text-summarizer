An Extractive text summarizer based on word frequencies and spacy. Used as a baseline model for the final project.

# Requirements
* Spacy
* Scikit-learn
* Pandas

# Starting

Make sure you are located in the project directory.

First, create a virtual environment, use virtualenv:
```virtualenv env```

Activate the environment:
```source env/bin/activate```

Install dependencies:
```pip install -r requirements.txt```

A necessary step is downloading the general-purpose spacy pre-trained models. Type the following command: ```./download.sh```

Next, run the following script to make a summary of the first 10 news articles located in ./files/news_inshort.xlsx:   
```python summarization.py```
