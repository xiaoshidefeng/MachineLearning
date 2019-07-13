import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model


def firstGraph(pd, data_train):
    fig = plt.figure()
    # 设置不透明度0.2
    fig.set(alpha=0.2)

    # 设置2 * 3的子图区域 第一个图位于（0，0）即第一行第一个列
    plt.subplot2grid((2, 3), (0, 0))
    # 用data_train里的Survived数据作柱形图
    data_train.Survived.value_counts().plot(kind='bar')
    plt.title(u"获救情况（1为获救）")
    plt.ylabel(u"人数")

    # 第二个图位于第一行第二列
    plt.subplot2grid((2, 3), (0, 1))
    data_train.Pclass.value_counts().plot(kind="bar")
    plt.ylabel(u"人数")
    plt.title(u"乘客等级分布")

    plt.subplot2grid((2, 3), (0, 2))
    plt.scatter(data_train.Survived, data_train.Age, alpha=0.1)
    plt.ylabel(u"年龄")
    plt.grid(b=True, which='major', axis='y')
    plt.title(u"按年龄看获救分布（1为获救）")

    # 第四个图位于第二行第一列和第二列 colspan=2 是指它的列跨度有两列
    plt.subplot2grid((2, 3), (1, 0), colspan=2)
    # kde是密度图
    data_train.Age[data_train.Pclass == 1].plot(kind='kde')
    data_train.Age[data_train.Pclass == 2].plot(kind='kde')
    data_train.Age[data_train.Pclass == 3].plot(kind='kde')
    plt.xlabel(u"年龄")
    plt.ylabel(u"密度")
    plt.title(u"各等级的乘客年龄分布")
    plt.legend((u"头等舱", u"二等舱", u"三等舱"), loc="best")

    plt.subplot2grid((2, 3), (1, 2))
    data_train.Embarked.value_counts().plot(kind='bar')
    plt.title(u"各登船口岸上岸人数")
    plt.ylabel(u"人数")

    plt.show()


def secondGraph(pd, data_train):
    # 看看各乘客等级的获救情况
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({u"获救": Survived_1, u"未获救": Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title(u"各乘客等级的获救情况")
    plt.ylabel(u"人数")
    plt.xlabel(u"乘客等级")
    plt.show()


def thirdGraph(pd, data_train):
    fig = plt.figure()
    fig.set(alpha=0.2)

    Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
    Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
    df = pd.DataFrame({u"男性": Survived_m, u"女性": Survived_f})
    df.plot(kind='bar', stacked=True)
    plt.title(u"按性别看获救情况")
    plt.ylabel(u"人数")
    plt.xlabel(u"获救")

    plt.show()


def fourthGraph(pd, data_train):
    fig = plt.figure()
    fig.set(alpha=0.65)
    plt.title(u"根据舱等级和性别的获救情况")

    ax1 = fig.add_subplot(141)
    data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3] \
        .value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
    ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
    ax1.legend([u"女性/高级仓"], loc="best")

    ax2 = fig.add_subplot(142, sharey=ax1)
    data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3] \
        .value_counts().plot(kind='bar', label="famale, low class", color='pink')
    ax2.set_xticklabels([u"获救", u"未获救"], rotation=0)
    ax2.legend([u"女性/低级仓"], loc="best")

    ax3 = fig.add_subplot(143, sharey=ax1)
    data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3] \
        .value_counts().plot(kind='bar', label="male highclass", color="lightblue")
    ax3.set_xticklabels([u"获救", u"未获救"], rotation=0)
    ax3.legend([u"男性/高级仓"], loc="best")

    ax4 = fig.add_subplot(144, sharey=ax1)
    data_train.Survived[data_train.Sex == "male"][data_train.Pclass == 3] \
        .value_counts().plot(kind="bar", label="male, low class", color='steelblue')
    ax4.legend([u"男性/低级仓"], loc="best")

    plt.show()


def fifthGraph(pd, data_train):
    fig = plt.figure()
    fig.set(alpha=0.2)

    Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title(u'各港口乘客的获救情况')
    plt.xlabel(u'登录港口')
    plt.ylabel(u'人数')

    plt.show()


def dataFirst(pd, data_train):
    g = data_train.groupby(['SibSp', 'Survived'])
    df = pd.DataFrame(g.count()['PassengerId'])
    # print(df)

    g = data_train.groupby(['Parch', 'Survived'])
    df = pd.DataFrame(g.count()['PassengerId'])
    # print(df)

    print(data_train.Cabin.value_counts())


def dataSecond(pd, data_train):
    fig = plt.figure()
    fig.set(alpha=0.2)

    Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
    Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
    df = pd.DataFrame({u'有票': Survived_cabin, u'无票': Survived_nocabin}).transpose()
    df.plot(kind='bar', stacked=True)
    plt.title(u"按照有无Cabin的获救情况")
    plt.ylabel(u'人数')
    plt.xlabel(u'Cabin有无')

    plt.show()


def set_missing_age(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    # y即目标年龄
    # [行,列]
    y = known_age[:, 0]
    # X即特征属性值
    # 舍弃第0列（即需要预测的Age那列）
    X = known_age[:, 1:]

    # RandomForestRegressor参数：
    # n_estimators弱学习器的最大迭代次数大小影响拟合度
    # n_jobs并行job个数 -1 是指与本机核心数相关
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    # 拟合
    rfr.fit(X, y)

    # [所有行, 舍弃第一列]
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 将预测的年龄结果 赋值给 数据中Age为空的数据的Age列
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'
    return df


def change_dummies(pd, data):
    dummies_cabin = pd.get_dummies(data['Cabin'], prefix='Cabin')
    dummies_embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')
    dummies_sex = pd.get_dummies(data['Sex'], prefix='Sex')
    dummies_pclass = pd.get_dummies(data['Pclass'], prefix='Pclass')

    # 拼接新的列数据
    df = pd.concat([data, dummies_cabin, dummies_embarked, dummies_sex, dummies_pclass], axis=1)

    # 删除多余的列
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    return df


def data_preporocessing(data):
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(data['Age'].values.reshape(-1, 1))
    data['Age_scaled'] = scaler.fit_transform(data['Age'].values.reshape(-1, 1), age_scale_param)
    fare_scale_param = scaler.fit(data['Fare'].values.reshape(-1, 1))
    data['Fare_scaled'] = scaler.fit_transform(data['Fare'].values.reshape(-1, 1), fare_scale_param)
    return data


def logistic_regression():
    # 通过正则拿出需要的那几列
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.values

    # 取出Survived那列 即第一列 作为y
    y = train_np[:, 0]

    # 取出第一列后面的所有列 作为x
    X = train_np[:, 1:]

    # 逻辑回归 正则化选择参数（惩罚项的种类）：
    # penalty 'l1'or 'l2', default: 'l2' 对应L1的正则化和L2的正则化 影响solver（损失函数优化算法）的选择
    # l1: solver 只能选liblinear
    # l2: solver 可选'newton-cg', 'lbfgs', 'liblinear', 'sag'
    # a) liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
    # b) lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
    # c) newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
    # d) sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，

    # C为正则化系数λ的倒数，通常默认为1
    # tol是迭代终止判据的误差范围
    clf = linear_model.LogisticRegression(solver='liblinear', C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)

    return clf


def format_test_data(data_test):
    # 将test文件的格式改成和train一样
    data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
    tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = tmp_df[data_test.Age.isnull()].values

    # 用之前创建号的随机森林的决策树来补充test数据的缺失Age
    X = null_age[:, 1:]
    predictedAge = rfr.predict(X)
    data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAge

    # 复用先前的函数
    data_test = set_Cabin_type(data_test)
    df_test = change_dummies(pd, data_test)
    df_test = data_preporocessing(df_test)
    # print(df_test.head())
    return df_test

if __name__ == '__main__':
    # 显示所有列
    # pd.set_option('display.max_columns', None)
    # # 显示所有行
    # pd.set_option('display.max_rows', None)
    # # 设置value的显示长度为100，默认为50
    # pd.set_option('max_colwidth', 100)
    # 全部显示在一行
    # pd.set_option('expand_frame_repr', False)

    data_train = pd.read_csv("data/train.csv")
    data_test = pd.read_csv("data/test.csv")
    # print(data_train.head(10))
    # print(data_train.info())
    # print(data_train.describe())

    # firstGraph(pd, data_train)
    # secondGraph(pd, data_train)
    # thirdGraph(pd, data_train)
    # fourthGraph(pd, data_train)
    # fifthGraph(pd, data_train)
    # dataFirst(pd, data_train)
    # dataSecond(pd, data_train)

    # 返回两个变量 用了随机森林
    data_train, rfr = set_missing_age(data_train)
    data_train = set_Cabin_type(data_train)
    # print(data_train.head(10))

    df = change_dummies(pd, data_train)
    # print(df.head(10))

    df = data_preporocessing(df)
    # print(df.head(10))
    clf = logistic_regression()
    # print(clf)
    df_test = format_test_data(data_test)
    # print(df_test.head(10))

    # 取出需要的列
    test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(test)
    # 得出结果 写入csv文件
    result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
    result.to_csv("data/logistic_regression_prediction.csv", index=False)
