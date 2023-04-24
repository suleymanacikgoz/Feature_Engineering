import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)



#Görev 1 : Keşifçi Veri Analizi


#Adım 1: Genel resmi inceleyiniz.

df=pd.read_csv("Odevler/6.Hafta/diabetes.csv")

df.describe().T

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

df[(df["Age"] < low) | (df["Age"] > up)]

df[(df["Age"] < low) | (df["Age"] > up)].index



df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)
df[(df["Age"] < low)].any(axis=None)


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


outlier_thresholds(df, "Age")

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age")


#Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))


#Adım 3:  Numerik ve kategorik değişkenlerin analizini yapınız.
#Adım 4:Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

cat_cols
num_cols

df.groupby("Age")["Outcome"].mean()
df.groupby("Outcome")["Age"].mean()
df.groupby("BMI").agg({"Outcome":["count","mean"]})
#Adım 5: Aykırı gözlem analizi yapınız.

[check_outlier(df,col) for col in df.columns]

#Adım 6: Eksik gözlem analizi yapınız.
df.isnull().any()
#Adım 7: Korelasyon analizi yapınız.

corr_matrix = np.corrcoef(df['Outcome'], df['BMI'])
corr_matrix[0,1]

#Görev 2 : Feature Engineering

#Adım 1:  Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb. değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0 olamayacaktır. Bu durumudikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik değerlere işlemleri uygulayabilirsiniz.

df.columns

df[df["Glucose"]==0]

df["Glucose"]=df["Glucose"].replace(0,np.nan)

df[df["Insulin"]==0]

df["Insulin"]=df["Insulin"].replace(0,np.nan)

df[df["BloodPressure"]==0]

df["BloodPressure"]=df["BloodPressure"].replace(0,np.nan)

df[df["SkinThickness"]==0]

df[df["BMI"]==0]


def checkZero(dataframe,columnName):
    dataframe[columnName]=dataframe[columnName].replace(0,np.nan)


checkZero(df,"SkinThickness")
checkZero(df,"BMI")


##################################################

low, up = outlier_thresholds(df, "Age")

df[((df["Age"] < low) | (df["Age"] > up))]["Age"]

df.loc[((df["Age"] < low) | (df["Age"] > up)), "Age"]

df.loc[(df["Age"] > up), "Age"] = up

df.loc[(df["Age"] < low), "Age"] = low


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.shape

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

##### aykırı gözlemleri düzelttin şimdi eksik değerleri doldur ve sonraki adıma geç

# eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()

# degiskenlerdeki eksik deger sayisi
df.isnull().sum()

# degiskenlerdeki tam deger sayisi
df.notnull().sum()

# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

missing_values_table(df, True)


#############################################
# Eksik Değer Problemini Çözme
#############################################



#Adım 2: Yeni değişkenler oluşturunuz.


# age level
df.loc[(df['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 18) & (df['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# pregnancy x age
df.loc[(df['Pregnancies'] > 0) & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Pregnancies'] == 0),'NEW_SEX_CAT' ] = 'male'
df.loc[(df['Pregnancies'] > 0) & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Pregnancies'] > 0) & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

# weight
df.loc[(df['BMI'] < 18.5),'NEW_WGHT_CAT'] = 'underweight'
df.loc[(df['BMI'] >= 18.5) & (df['BMI'] < 24.9),'NEW_WGHT_CAT'] = 'normal'
df.loc[(df['BMI'] >= 24.9) & (df['BMI'] < 29.9),'NEW_WGHT_CAT'] = 'overweight'
df.loc[(df['BMI'] >= 29.9) & (df['BMI'] < 34.9),'NEW_WGHT_CAT'] = 'obese'
df.loc[(df['BMI'] >= 34.9),'NEW_WGHT_CAT'] = 'extremely obese'

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df["NEW_SEX_CAT"].isnull

#############################################
# 2. Outliers (Aykırı Değerler)
#############################################

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))



#Adım 3:  Encoding işlemlerini gerçekleştiriniz.

missing_values_table(df)



#Adım 4: Numerik değişkenler için standartlaştırma yapınız.




#Adım 5: Model oluşturunuz.



