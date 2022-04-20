import data_processing as d
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split   

X,_ = d.get_encoded_data()
y = d.get_SVM_y()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

clf.fit(X_train,  y_train)
predictions=clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)