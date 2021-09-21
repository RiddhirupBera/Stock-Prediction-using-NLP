import pandas as pd
import numpy as np

dataset = pd.read_csv("/home/riddhirup/Downloads/Google_dataset.csv",delimiter = ",")
import re # used to replace special characters
import nltk #natural language tool kit - all the necessary libaries for 
nltk.download("stopwords") # nltk corpus - data is stored
from nltk.corpus import stopwords # detect stopwords
from nltk.stem.porter import PorterStemmer # stem your word 
ps = PorterStemmer()

data = []


for i in range(0,2006):
    review = dataset["Title"][i]
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split() #[wow, loved , this, place]
    
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    data.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
x = cv.fit_transform(data).toarray()
y = dataset.iloc[:,3:].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model =  Sequential()
model.add(Dense(units = 1602,kernel_initializer = "random_uniform",activation= "relu"))
model.add(Dense(units = 3000,kernel_initializer = "random_uniform",activation= "relu"))
model.add(Dense(units = 3000,kernel_initializer = "random_uniform",activation= "relu"))
model.add(Dense(units = 1,kernel_initializer = "random_uniform",activation= "sigmoid"))
model.compile(optimizer = "rmsprop",loss = "binary_crossentropy",metrics = ["accuracy"])

model.fit(x_train,y_train,epochs = 3)