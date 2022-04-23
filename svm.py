
import data_processing as d
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

average_accurracy=[]

for x in range(0,10):
    X,_ = d.get_encoded_data('MachineLearningProject/datasets/VirusSample.csv')
    y = d.get_SVM_y('MachineLearningProject/datasets/VirusSample.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=4)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='rbf'))
    clf.fit(X_train, y_train)
    average_accurracy.append(clf.score(X_test, y_test))

plt.plot([x for x in range(0,10)], average_accurracy, color='red', linewidth=3)