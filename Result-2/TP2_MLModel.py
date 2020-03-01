from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

file_list = ['PRSA_Data_Gucheng_20130301-20170228.csv',
             'PRSA_Data_Aotizhongxin_20130301-20170228.csv',
             'PRSA_Data_Changping_20130301-20170228.csv',
             'PRSA_Data_Dingling_20130301-20170228.csv',
             'PRSA_Data_Dongsi_20130301-20170228.csv',
             "PRSA_Data_Guanyuan_20130301-20170228.csv",
             'PRSA_Data_Huairou_20130301-20170228.csv',
             'PRSA_Data_Nongzhanguan_20130301-20170228.csv',
             'PRSA_Data_Shunyi_20130301-20170228.csv',
             "PRSA_Data_Tiantan_20130301-20170228.csv",
             "PRSA_Data_Wanliu_20130301-20170228.csv"]

def categorizeTemp(temp):
    val = ""
    if temp < 0:
        val = "verycold"
    elif 0 <= temp < 10:
        val = "cold"
    elif 10 <= temp < 20:
        val = "moderate"
    elif 20 <= temp < 30:
        val = "hot"
    elif temp >= 30:
        val = "veryhot"
    return val

def Model1(file,Y, X):
    #split the data into test and train
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.32, random_state=42)
    #Parameter tunning....
    kernels = ['linear','rbf','poly']
    gammas = [1e-3, 1e-4]
    Cs = [1, 10, 100, 1000]
    highest_score = 0
    for kernel in kernels:
        for c in Cs:
            for gamma in gammas:
                clf = SVC(gamma=gamma, kernel=kernel, C=c).fit(x_train, y_train)
                score = clf.score(x_test, y_test, sample_weight=None)
                if score > highest_score:
                    highest_score = score
                    print(score)
    print("FileName:{} - Train_size:{}, test_size:{}, highest_score:{}".format(file,len(x_train), len(x_test), highest_score))

for file in file_list:
    df = pd.read_csv('/Users/mosesopenja/Documents/Winter2020/CloudComputing/TP2/tp2-dataset/' + file)
    # Convert to list
    TEMP = df.TEMP.values.tolist()
    df.drop(['TEMP'], axis=1)
    Y = []
    for temp in TEMP:
        Y.append(categorizeTemp(temp))
    X = df.iloc[:0,16]
    #print(Y)
    #print(X)
    Model1(file,Y,X)


