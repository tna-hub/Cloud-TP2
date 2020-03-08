
from nltk.corpus import stopwords
from pyspark import SparkContext
import boto3
import re
import string
from nltk.stem import *
import nltk
#from nltk.stem.porter import *

nltk.download('punkt')

nltk.download('stopwords')

def clean_str(x):
    x = x.lower()
    x = x.replace('--', ' ')
    x = x.replace('“', "")
    clean_str = re.sub(r'^\W+|\W+$', '', x)
    return re.split(r'\W+', clean_str)


def stemWord(word):
    porter = PorterStemmer()
    word = porter.stem(word)
    return word


if __name__ == '__main__':

    s3 = boto3.client('s3')
    s3.download_file('advanced-cloud-tp2', 'file.txt', 'result.txt')  # downoloading file from S3

    sc = SparkContext("local", "wordcount")

    stopWords = set(stopwords.words('english'))

    rdd = sc.textFile('result.txt')  # creating rdd

    #x = rdd.count()
    #print("number of rdds:" + str(x))

    rdd = rdd.filter(lambda x: len(x) > 0)  # delete empty lines

    #x = rdd.count()
    #print("number of parts:" + str(x))

    rdd = rdd.flatMap(lambda x: clean_str(x))  # 'clean' the text line by line and split it in words

    rdd = rdd.filter(lambda x: x not in stopWords)  # remove stopwords

    rdd = rdd.map(lambda x: stemWord(x))  # stemm the words

    rdd = rdd.map(lambda x: (x, 1)) #  perform mapReduce

    rdd = rdd.reduceByKey(lambda x, y: x + y)

    rdd = rdd.map(lambda x: (x[1], x[0]))

    rdd = rdd.sortByKey(False)

    print(rdd.take(20))
