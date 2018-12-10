
# coding: utf-8

# In[2]:


# 导入相关包
import pandas as pd
# 读取数据
titanic_df = pd.read_csv('./datasets/titanic/train.csv')
# drop 掉 PassengerId,Name,Ticket,Cabin
titanic_df = titanic_df.drop(['PassengerId','Name','Ticket','Cabin'], axis = 1)
# 构造一个新变量 AgeIsMissing
titanic_df['AgeIsMissing'] = 0
titanic_df.loc[titanic_df['Age'].isnull(), 'AgeIsMissing'] = 1
# 对 Age 缺失值进行均值填充
age_mean = round(titanic_df['Age'].mean())
titanic_df['Age'].fillna(age_mean, inplace=True)
# 对 Embarked 缺失值用'S'替换
titanic_df['Embarked'].fillna('S', inplace=True)
# 对 Age 进行分箱--自定义分箱
cut_points = [0,18,25,40,60,100]
titanic_df["AgeBin"] = pd.cut(titanic_df.Age, bins=cut_points)
# 对 Fare 船票价格进行分箱--等深分箱
titanic_df["FareBin"] = pd.qcut(titanic_df.Fare, 5)
# 构造 FamilySize 变量
titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1
# 构造一个新变量 IsAlone（是否独自一人)
titanic_df['IsAlone'] = 0
titanic_df.loc[titanic_df['FamilySize'] == 1, 'IsAlone'] = 1
# 构造一个新变量 IsMother（是否是母亲）
titanic_df['IsMother'] = 0
titanic_df.loc[(titanic_df['Sex']=='female') & (titanic_df['Parch']>0) & (titanic_df['Age']>20),
'IsMother'] = 1
# 把 Sex 性别和 AgeBin 特征进行组合
titanic_df['SexAgeCombo'] = titanic_df['Sex'] + "_" + titanic_df['AgeBin'].astype(str)
# 对 Pclass,Sex,Embarked,AgeBin,FareBin,FamilySize,Sex_Age_combo 进行独热编码
Pclass = pd.get_dummies(titanic_df.Pclass,prefix='Pclass')
Sex = pd.get_dummies(titanic_df.Sex,prefix='Sex')
Embarked = pd.get_dummies(titanic_df.Embarked,prefix='Embarked')
AgeBin = pd.get_dummies(titanic_df.AgeBin,prefix='AgeBin')
FareBin = pd.get_dummies(titanic_df.FareBin,prefix='FareBin')
FamilySize = pd.get_dummies(titanic_df.FamilySize,prefix='FamilySize')
SexAgeCombo = pd.get_dummies(titanic_df.SexAgeCombo,prefix='SexAgeCombo')
# 把需要的变量全部拼接在一起 
TrainData =pd.concat([titanic_df[['Survived','AgeIsMissing','IsAlone','IsMother']],
Pclass,Sex,Embarked,AgeBin,FareBin,FamilySize,SexAgeCombo],axis=1)
# 描述性统计
TrainData.describe().transpose()

