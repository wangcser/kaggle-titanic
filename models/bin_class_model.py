# data set
import pandas as pd
from data_set.data_analysis import data_set

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model.perceptron import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


class Model:
    """
    this class include fine tuning models in sk-learn with Titanic data set
    also use the ensemble methods to obtain a better result.
    """
    def __init__(self):
        self.train_df, self.test_df = data_set()
        # X_train is 891*8, y_train is 891*1, X_test is 418*8
        self.X_train = self.train_df.drop('Survived', axis=1)
        self.y_train = self.train_df['Survived']
        self.X_test = self.test_df.drop('PassengerId', axis=1).copy()
        self.y_pred = None  # y_pred is 418*1

    def logistic_regression(self):
        # Logistic Regression
        logreg = LogisticRegression(penalty='l2', dual=True, max_iter=50)
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
        self.y_pred = reg.predict(self.X_test)

    def lasso(self):
        lasso = Lasso(alpha=0.1)
        lasso.fit(self.X_train, self.y_train)

        acc = round(lasso.score(self.X_train, self.y_train)*100, 2)
        print("acc with Lasso:", acc)

        self.y_pred = lasso.predict(self.X_test)

    def svm(self):
        # SVM
        svc = SVC(C=100.0, kernel='rbf')
        svc.fit(self.X_train, self.y_train)

        acc = round(svc.score(self.X_train, self.y_train)*100, 2)
        print("acc with SVM:", acc)

        self.y_pred = svc.predict(self.X_test)

    def knn(self):
        # KNN
        knn = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='auto')
        knn.fit(self.X_train, self.y_train)

        acc = round(knn.score(self.X_train, self.y_train)*100, 2)
        print("acc with KNN:", acc)

        self.y_pred = knn.predict(self.X_test)

    def naive_bayes(self):
        # Gaussian Naive Bayes Classifiers
        gaussian = GaussianNB()
        gaussian.fit(self.X_train, self.y_train)

        acc = round(gaussian.score(self.X_train, self.y_train)*100, 2)
        print("acc with naive bayes:", acc)

        self.y_pred = gaussian.predict(self.X_test)

    def preceptron(self):
        # Perceptron
        perceptron = Perceptron(penalty='l2', max_iter=1000, shuffle=True)
        perceptron.fit(self.X_train, self.y_train)

        acc = round(perceptron.score(self.X_train, self.y_train)*100, 2)
        print("acc with Perceptron:", acc)

        self.y_pred = perceptron.predict(self.X_test)

    def linear_svc(self):
        # Linear SVC
        linear_svc = LinearSVC(penalty='l2', C=10.0)
        linear_svc.fit(self.X_train, self.y_train)

        acc = round(linear_svc.score(self.X_train, self.y_train)*100, 2)
        print("acc with Linear SVC:", acc)

        self.y_pred = linear_svc.predict(self.X_test)

    def sgd(self):
        # SGD
        sgd = SGDClassifier(max_iter=500)
        sgd.fit(self.X_train, self.y_train)

        acc = round(sgd.score(self.X_train, self.y_train)*100, 2)
        print("acc with SGD:", acc)

        self.y_pred = sgd.predict(self.X_test)

    def decision_tree(self):
        # Decision Tree
        decision_tree = DecisionTreeClassifier(criterion='entropy')
        decision_tree.fit(self.X_train, self.y_train)

        acc = round(decision_tree.score(self.X_train, self.y_train)*100, 2)
        print("acc with Decision Tree:", acc)

        self.y_pred = decision_tree.predict(self.X_test)

    def random_forest(self):
        # Random Forest
        random_forest = RandomForestClassifier(n_estimators=100)
        random_forest.fit(self.X_train, self.y_train)

        acc = round(random_forest.score(self.X_train, self.y_train) * 100, 2)
        print("acc with Random Forest:", acc)
        self.y_pred = random_forest.predict(self.X_test)

    def adaboost(self):
        ada = AdaBoostClassifier()
        ada.fit(self.X_train, self.y_train)

        acc = round(ada.score(self.X_train, self.y_train)*100, 2)
        print("acc with adaboost:", acc)
        self.y_pred = ada.predict(self.X_test)

    def bagging(self):
        bag = BaggingClassifier(n_estimators=100)
        bag.fit(self.X_train, self.y_train)

        acc = round(bag.score(self.X_train, self.y_train)*100, 2)
        print("acc with bagging:", acc)
        self.y_pred = bag.predict(self.X_test)

    def gradient_boost(self):
        gb = GradientBoostingClassifier(learning_rate=1.0)
        gb.fit(self.X_train, self.y_train)

        acc = round(gb.score(self.X_train, self.y_train)*100, 2)
        print("acc with gradient boost:", acc)
        self.y_pred = gb.predict(self.X_test)

    def xgb(self):
        pass

    def mlp(self):
        mlp = MLPClassifier(learning_rate='adaptive', hidden_layer_sizes=(100, 10))
        mlp.fit(self.X_train, self.y_train)

        acc = round(mlp.score(self.X_train, self.y_train)*100, 2)
        print("acc with MLP:", acc)
        y_pred = mlp.predict(self.X_test)

    def submission(self):
        submission = pd.DataFrame({
                "PassengerId": self.test_df["PassengerId"],
                "Survived": self.y_pred
            })

        # print(submission)
        data_path = '/media/super/Dev Data/ml_data_set/Kaggle_Titanic'
        submission.to_csv(data_path + '/output/knn_submission.csv', index=False)


def main():
    pass


if __name__ == '__main__':
    main()
