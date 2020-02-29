import re
import os
from imp import reload
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pyspark import SparkContext
import sys
reload(sys)
sys.setdefaultencoding('utf8')
ps = PorterStemmer()
# to preprocess the text call this method, remove the stop words and create stemmed word using PorterStemmer
def Preprocessing(text):
    article = re.sub('[^a-zA-Z]', ' ', str(text))
    article = article.lower()
    article = article.split()
    article = [ps.stem(word)
 for word in article if not word in set(stopwords.words('english')) and len(word)
 != 1]
    article = ' '.join(article)
    return article
def CleanBook(input_path, file_name):
    inputStream = open(input_path + file_name, 'r')
    documents = inputStream.readlines()
    # dynamicall defined the cleaned file name
    cleaned_path = input_path +file_name.split(".")[0] +"_cleaned."+file_name.split(".")[1]
    corpus = ""
    for document in documents:
        paragraph = document
        #if ":" not in document:
        cleaned_word = Preprocessing(document)
        # paragraph = cleaned_word
        if len(cleaned_word) > 0:
            corpus = corpus+"\n"+cleaned_word
            #print(corpus)
    inputStream.close()
    directory = os.path.dirname(cleaned_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    trainingStream = open(cleaned_path, 'w')
    trainingStream.write(corpus)
    trainingStream.close()
    return cleaned_path


input_path = "/home/ubuntu/"
file = "Book.txt"
cleaned_file = CleanBook(input_path,file)
# Spark context with configuration
sc = SparkContext("local", "......TP2 Word count......")
# read the text file and split each line
words = sc.textFile(cleaned_file).flatMap(lambda line: line.split(" "))
# count the occurrence of each word
wordCounts = words.map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b) \
             .sortBy(lambda a:a[1])
words = wordCounts.collect()
top = 20
result_path = input_path +"wordCounts_top_"+str(top)+".txt"
result = ""
for index in range(len(words)-1, len(words)-(top+1), -1):
    #print (words[index])
    result = result+str(words[index])+"\n"

dir = os.path.dirname(result_path)
if not os.path.exists(dir):
    os.makedirs(dir)
# save the word count results now......
output = open(result_path,'w')
output.write(result)
output.close()