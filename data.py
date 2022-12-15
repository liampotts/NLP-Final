import pandas as pd

#put csv path here
csvFile="/Users/liampotts/Desktop/NLP/finalProject/train-balanced-sarcasm.csv"
df = pd.read_csv(csvFile, usecols = ['label','comment','parent_comment'])


#puts all sarcastic comments at top followed by non sarcastic at bottom
df=df.sort_values(by=['label'])


#df0=only the not sarcastic comments
df0 = df.loc[df['label'] == 0]

#df1= only the sarcastic comments
df1 = df.loc[df['label'] == 1]

print(len(df0))
print(len(df1))


def createDataset(df0,df1):
    test0=df0.sample(25000)
    test1 = df1.sample(25000)
    df3 = pd.concat([test0, test1])
    df3.to_csv('dataset1.csv',index=False)
    return df3


df3= createDataset(df0,df1)


def createTrainTestSet(df3):
    sarcastic_train=df3[:20000]
    sarcastic_test=df3[20000:25000]
    notsarcastic_train=df3[25000:45000]
    notsarcastic_test=df3[45000:]

    trainFinal=pd.concat([sarcastic_train,notsarcastic_train])
    testFinal=pd.concat([sarcastic_test, notsarcastic_test])

    trainFinal.to_csv('trainFinal1.csv', index=False)
    testFinal.to_csv('testFinal1.csv', index=False)

createTrainTestSet(df3)





