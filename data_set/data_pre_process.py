# data analysis and wrangling
import os
import pandas as pd
import numpy as np
import data_set.data_config as cfg


def data_set():
    # acquire data
    data_path = cfg.DATA_PATH
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
    combine = [train_df, test_df]

    # drop the Cabin and Ticket features.
    # in pandas, axis=0 refers to rows, axis=1 refers to cols.
    # print("before", train_df.shape, test_df.shape)
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    # print("after", train_df.shape, test_df.shape)
    combine = [train_df, test_df]

    # creating new features

    # extract title from name
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # Title_result = pd.crosstab(train_df['Title'], train_df['Sex'])
    # print(Title_result)
    # replace many title with Rare target.
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # Title_result = train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
    # print(Title_result)
    # convert the categorical title to ordinal.
    title_mapping = {
        "Mr": 1,
        "Miss": 2,
        "Mrs": 3,
        "Master": 4,
        "Rare": 5
    }
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    # print(train_df.head())

    # now, we can safely drop the name feature.
    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)  # there is no passenger Id feature.
    combine = [train_df, test_df]

    # convert features which contain strings to numerical vals.
    # convert sex to gender in 0,1
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

    # print(train_df.head())

    # estimating and completing features with missing or null vals.
    # there are three ways to complete a numerical continuous features.
    # 1. generate random numbers between mean and std-deviation
    # 2. use other correlated features
    # 3. combine methods 1 and 2.

    # there use methods 2 to complete age.
    # start by preparing an empty array to contain guessed age vals based on the Pclass * Gender.
    guess_ages = np.zeros((2, 3))
    # print(guess_ages)
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & \
                                   (dataset['Pclass'] == j + 1)]['Age'].dropna()

                age_guess = guess_df.median()

                # convert random age float to nearest .5 age
                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                            'Age'] = guess_ages[i, j]

        dataset['Age'] = dataset['Age'].astype(int)

    # print(train_df.head())

    # create age-band and determine the correlations with survived.
    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
    # Ageband_result = train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False)\
    #     .mean()\
    #     .sort_values(by='AgeBand', ascending=True)
    # print(Ageband_result)
    # replace age with ordinals based on the bands.
    for dataset in combine:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age']
    # print(train_df.head())
    # here we can remove ageband feature
    train_df = train_df.drop(['AgeBand'], axis=1)
    combine = [train_df, test_df]
    # print(train_df.head())

    # create new feature for family size which combines Parch and SibSp
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # Familysize_result = train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False)\
    #     .mean()\
    #     .sort_values(by='Survived', ascending=False)
    # print(Familysize_result)

    # create another feature called isAlone
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    # Isalone_result = train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
    # print(Isalone_result)

    # now drop Parch, SibSp, Family size in favor of IsAlone.
    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_df, test_df]

    # print(train_df.head())

    # we can also create an artificial feature combining Pclass and Age.
    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass
    # print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))

    # completing the categorical feature Embarked
    freq_port = train_df.Embarked.dropna().mode()[0]
    # print(freq_port)

    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    # Port_result = train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(
    # by='Survived', ascending=False)
    # print(Port_result)

    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # print(train_df.head())

    # finally convert the Fare feature.
    # We can now complete the Fare feature for single missing value in test dataset
    # using mode to get the value that occurs most frequently for this feature.
    # We do this in a single line of code.
    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
    # print(test_df.head())

    # next, use fare band to remap the fare.

    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    train_df = train_df.drop(['FareBand'], axis=1)
    combine = [train_df, test_df]

    # print(train_df.head(10))
    # print(test_df.head(10))

    # now, all data prepared.

    return train_df, test_df
