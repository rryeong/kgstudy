# kgstudy
import os
import pandas as pd
import numpy as np
print(os.getcwd())
os.chdir("C:/Users/user/Desktop/세진_령아")
print(os.getcwd())

df=pd.read_csv('train.csv')
#열확인
df.columns
df.dtypes
#칼럼별 결측값 개수 
df.isnull().sum(0)
#결측치 있는 칼럼이름 칼럼 11개
columns=["LotFrontage","Alley","FireplaceQu","GarageType","GarageYrBlt","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]
#결측치 있는 칼럼만 추출
df3=df[columns]
df3
#수치형변수;LotFrontage,GarageType,GarageYrBlt
#범주형변수:Alley,FireplaceQu,GarageFinish,GarageQual,GarageCond,PoolQC,Fence,MiscFeature

columns_s=["LotFrontage","GarageYrBlt"]
columns_b=["Alley","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]

df4=df[columns_s]
df4
df5=df[columns_b]

#수치형칼럼 결측치 결측치 바로 평균으로 대체
np.mean(df4['LotFrontage'])
np.mean(df4['GarageYrBlt'])
df4['LotFrontage']=df4['LotFrontage'].replace(to_replace=np.nan,value=70)
df4['GarageYrBlt']=df4['GarageYrBlt'].replace(to_replace=np.nan,value=1978)

#수치형칼럼결측치 시각화해보기
df4.info()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.heatmap(df4.isnull(),cbar=True)

#수치형변수칼럼df4에서 제거할 칼럼 없음

#이상치 확인
plt.rcParams["font.family"] = 'Malgun Gothic'
plt.rcParams["font.size"] = 16
plt.rcParams["figure.figsize"] = (20, 10)
df4.boxplot()
plt.show()
#이상치 필터링
#df4_outlier=df4.query('LotFrontage>250')
#df4_outlier
#이상치 제거
#df4_outlier_del=df4_outlier[df4_outlier['LotFrontage']==313.0].index
#df4_outlier=df4_outlier.drop(df4_outlier_del)

#df4=df4_outlier
#df4

#범주형칼럼 결측치 시각화해보기
df5.info()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.heatmap(df5.isnull(),cbar=True)
#결측치 너무 많은 칼럼 Alley,FireplaceQu ,PoolQC,Fence,MiscFeature은 삭제하자!

df5_2=df5.drop(['Alley','FireplaceQu' ,'PoolQC','Fence','MiscFeature'],axis=1)

#나머지 결측치는 최빈값으로 대체
df5_2['GarageFinish'].value_counts()
df5_2['GarageFinish'].fillna('Unf')

df5_2['GarageQual'].value_counts()
df5_2['GarageQual'].fillna('TA')

df5_2['GarageCond'].value_counts()
df5_2['GarageCond'].fillna('TA')

df5_3=df5_2[['GarageFinish','GarageQual','GarageCond']]
df5_3

#결측치 처리한 칼럼만 모아놓음
df6=pd.concat([df4,df5_3],axis=1)
df6

df7=df.drop(["LotFrontage","GarageYrBlt","Alley","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"],axis=1)
df8=pd.concat([df6,df7],axis=1)
df8

from sklearn.ensemble import RandomForestRegressor # 회귀트리(모델)
from sklearn.model_selection import train_test_split # train/test
from sklearn.datasets import fetch_california_housing, load_boston # dataset 
from sklearn.metrics import mean_squared_error # 평균제곱오차


#dataset loading
x=df8
y=df8['SalePrice']

x.shape #(1460, 75)
y.shape #(1459, 80)

#관측치 10개 확인 
#x[:10,:] # x변수 정규화 
#y[:10]

#정규화 
import numpy as np
y = np.log1p(y)
y[:10]
y[:100]
#model 평가 : rmse로 하자(!!)
#rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
print('rmse=', rmse)                                                                                                                               
y_true[:10]
y_pred[:10]

# 상관관계
import pandas as pd
df = pd.DataFrame({'y_true':y_true, 'y_pred':y_pred})
cor = df['y_true'].corr(df['y_pred'])








