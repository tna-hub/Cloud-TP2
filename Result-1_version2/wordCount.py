import nltk
from nltk.corpus import stopwords
from pyspark import SparkContext
import boto3
import re
import string

nltk.download('stopwords')

punc = ' !"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~ '

def clean_str(x):
    x = x.lower()
    x = x.replace('--',' ')
    x = x.replace('“', "")
    clean_str = re.sub(r'^\W+|\W+$', '', x)
   # x = x.replace('\'', "")
   # table = str.maketrans('', '', string.punctuation)
   # clean_str = x.translate(table)
   # clean_str = x.strip(punc)
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

#        print(rdd.take(3000))
        rdd = rdd.map(lambda x: (x, 1))
        rdd = rdd.reduceByKey(lambda x, y: x + y)
#        print(rdd.take(5))
        rdd = rdd.map(lambda x: (x[1], x[0]))
 #       print(rdd.take(5))
        rdd = rdd.sortByKey(False)
        print(rdd.take(20))

