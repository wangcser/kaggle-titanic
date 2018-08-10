# data set
import pandas as pd
from data_set.data_analysis import data_set

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model.perceptron import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


class Model:
    def __init__(self):
        train_df, test_df = data_set()
        # X_train is 891*8, y_train is 891*1, X_test is 418*8
        self.X_train = train_df.drop('Survived', axis=1)
        self.y_train = train_df['Survived']
        self.X_test = test_df.drop('PassengerId', axis=1).copy()
        self.y_pred = None  # y_pred is 418*1

    def logistic_regression(self):
        # Logistic Regression
        logreg = LogisticRegression()
        logreg.fit(self.X_train, self.y_train)

        acc = round(logreg.score(self.X_train, self.y_train)*100, 2)
        print("acc with LR:", acc)

        self.y_pred = logreg.predict(self.X_test)

        """
        # we can use LR to validation our assumptions and decisions for feature creating and completing goals
        # 求各个特征与结果之间的相关性
        coeff_df = pd.DataFrame(train_df.columns.delete(0))
        coeff_df.columns = ['Feature']
        coeff_df['Correlation'] = pd.Series(logreg.coef_[0])
        result = coeff_df.sort_values(by='Correlation', ascending=False)
        print(result)
        """

    def ridge(self):
        reg = Ridge(alpha=0.5)
        reg.fit(self.X_train, self.y_train)

        acc = round(reg.score(self.X_train, self.y_train)*100, 2)
        print("acc with Ridge:", acc)
        y_pred = reg.predict(self.test)


    def svm(self):
        # SVM
        svc = SVC()
        svc.fit(self.X_train, self.y_train)

        acc = round(svc.score(self.X_train, self.y_train)*100, 2)
        print("acc with SVM:", acc)

        y_pred = svc.predict(self.X_test)

    def knn(self):
        # KNN
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(self.X_train, self.y_train)

        acc = round(knn.score(self.X_train, self.y_train)*100, 2)
        print("acc with KNN:", acc)

        y_pred = knn.predict(self.X_test)

    def naive_bayes(self):
        # Gaussian Naive Bayes Classifiers
        gaussian = GaussianNB()
        gaussian.fit(self.X_train, self.y_train)

        acc = round(gaussian.score(self.X_train, self.y_train)*100, 2)
        print("acc with naive bayes:", acc)

        y_pred = gaussian.predict(self.X_test)

    def preceptron(self):
        # Perceptron
        perceptron = Perceptron()
        perceptron.fit(self.X_train, self.y_train)

        acc = round(perceptron.score(self.X_train, self.y_train)*100, 2)
        print("acc with Perceptron:", acc)

        Y_pred = perceptron.predict(self.X_test)

    def linear_svc(self):
        # Linear SVC
        linear_svc = LinearSVC()
        linear_svc.fit(self.X_train, self.y_train)

        acc = round(linear_svc.score(self.X_train, self.y_train)*100, 2)
        print("acc with Linear SVC:", acc)

        Y_pred = linear_svc.predict(self.X_test)

    def sgd(self):
        # SGD
        sgd = SGDClassifier()
        sgd.fit(self.X_train, self.y_train)

        acc = round(sgd.score(self.X_train, self.y_train)*100, 2)
        print("acc with SGD:", acc)

        Y_pred = sgd.predict(self.X_test)

    def decision_tree(self):
        # Decision Tree
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(self.X_train, self.y_train)

        acc = round(decision_tree.score(self.X_train, self.y_train)*100, 2)
        print("acc with Decision Tree:", acc)

        Y_pred = decision_tree.predict(self.X_test)

    def random_forest(self):
        # Random Forest
        random_forest = RandomForestClassifier(n_estimators=100)
        random_forest.fit(self.X_train, self.y_train)

        acc = round(random_forest.score(self.X_train, self.y_train) * 100, 2)
        print("acc with Random Forest:", acc)

    def boosting(self):
        pass

    def bagging(self):
        pass

    def submission(self):
        submission = pd.DataFrame({
                "PassengerId": self.test_df["PassengerId"],
                "Survived": self.Y_pred
            })

        # print(submission)
        data_path = '/media/super/Dev Data/ml_data_set/Kaggle_Titanic'
        submission.to_csv(data_path + '/output/submission.csv', index=False)


def main():
    pass


if __name__ == '__main__':
    main()
