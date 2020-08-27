import pandas as pd
df = pd.read_csv('dataset_diabetes.csv')
df.head()

df.shape

df.columns.isnull().sum().any()
# No missing values

X = df.drop("Outcome",axis= 1)
Y = df["Outcome"]

# scaling down the features
#from sklearn.preprocessing import normalize
#df_scaled = normalize(X)


#df_scaled = pd.DataFrame(df_scaled,columns = list(df.columns[0:8]))
#df_scaled.head()




#target = np.asarray(df.iloc[:,-1])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3,random_state = 321)





from sklearn.linear_model import LogisticRegression
lr =LogisticRegression()

model = lr.fit(x_train,y_train)

pred = model.predict(x_test)


from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
classification_report(y_test,pred)


accuracy_score(y_test,pred)
confusion_matrix(y_test,pred)



import pickle


pickle.dump(model,open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))