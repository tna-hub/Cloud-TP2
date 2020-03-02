
import nltk
from nltk.corpus import stopwords
from pyspark import SparkContext
import boto3
import re
import string

nltk.download('stopwords')

def clean_str(x):
    x = x.lower()
    x = x.replace('--',' ')
    x = x.replace('“', "")
    clean_str = re.sub(r'^\W+|\W+$', '', x)
    return re.split(r'\W+',clean_str)

if __name__ == '__main__':

        s3 = boto3.client('s3')
        s3.download_file('advanced-cloud-tp2', 'file.txt', 'result.txt')

        sc = SparkContext("local", "wordcount")

        stopWords = set(stopwords.words('english'))

        rdd = sc.textFile('result.txt')
        rdd = rdd.filter(lambda x: len(x) > 0) 

        rdd = rdd.flatMap(lambda x: clean_str(x))
        rdd = rdd.filter(lambda x: x not in stopWords)

        rdd = rdd.map(lambda x: (x, 1))
        rdd = rdd.reduceByKey(lambda x, y: x + y)
        rdd = rdd.map(lambda x: (x[1], x[0])) # switch values and keys to sort by the largest number of occurances
        rdd = rdd.sortByKey(False)
        print(rdd.take(20))



