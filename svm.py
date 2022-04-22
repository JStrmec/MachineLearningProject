
import data_processing as d
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable


X,_ = d.get_encoded_data()
y = d.get_y()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)
#Pipeline(steps=[('standardscaler', StandardScaler()),('svc', SVC(gamma='auto'))])
predict = clf.predict(X_test)
print(predict)
print(accuracy_score(y_test, predict))
predict2 = clf.score(X_test, y_test)
print(predict2)
EPOCHS = 1000 
plt.plot(EPOCHS , predict2, color='red', linewidth=3)