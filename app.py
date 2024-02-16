import streamlit as st
import os
import gdown
from transformers import TextClassificationPipeline, DistilBertTokenizerFast,TFDistilBertForSequenceClassification

labels = {"LABEL_1" :"POSITIVE", "LABEL_0" :"NEGATIVE"}

###Take Fine tuned model from gdrive directory###
BASE_DIR = os.getcwd()
import gdown
url = "https://drive.google.com/drive/folders/1-6BgNpyOCcG_4scieAuhIJPU1Rcr5_zI"
url2 = "https://drive.google.com/drive/folders/1NnPO7hiqzkOpGXwzLPcWHRV1ZkYF7KsH"
if os.path.isdir('tokenizer') == False and os.path.isdir('sentiBert_model_imdb') == False:
    gdown.download_folder(url, quiet=True, use_cookies=False)
    gdown.download_folder(url2, quiet=True, use_cookies=False)

###Set Tokenizer and Model###
tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_DIR + "/tokenizer")
model = TFDistilBertForSequenceClassification.from_pretrained(BASE_DIR + "/sentiBert_model_imdb")

###Create UI with Streamlit###
st.sidebar.title("Review Sentiment Analysis")
title = st.text_input('Try Out review here:')

###Text Clasiification###
try:
    if title:
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
        result = pipe(title)
        print(result)
        result_label = max(result[0],key=lambda item:item['score'])['label']
        result_sent = labels[result_label]

        if result_label == 'LABEL_1':
            sung = ':sunglasses:'
        else:
            sung = ""

        st.write('The review is', result_sent + sung)

except Exception as error:
    st.write('Give a proper input')
    print(str(error))


