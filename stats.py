import pandas 
import numpy as np
from pandas import plotting
from scipy import stats
import scipy
from statsmodels.formula.api import ols
import seaborn
from matplotlib import pyplot as plt

data = pandas.read_csv('brain_size.csv', sep=';', na_values=".")

data

t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)

pandas.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})

data.shape

print(data['Gender'])  # Columns can be addressed by name

# Simpler selector
data[data['Gender'] == 'Female']['VIQ'].mean()

groupby_gender = data.groupby('Gender')

for gender, value in groupby_gender['VIQ']:
    print((gender, value.mean()))

groupby_gender.mean()

plotting.scatter_matrix(data[['Weight', 'Height', 'MRI_Count']])

plotting.scatter_matrix(data[['PIQ', 'VIQ', 'FSIQ']])

stats.ttest_1samp(data['VIQ'], 0)

female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender'] == 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq)

stats.ttest_ind(data['FSIQ'], data['PIQ']) 

stats.ttest_rel(data['FSIQ'], data['PIQ'])

stats.ttest_1samp(data['FSIQ'] - data['PIQ'], 0) 

stats.wilcoxon(data['FSIQ'], data['PIQ'])


x = np.linspace(-5, 5, 20)
np.random.seed(1)

# normal distributed noise
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)

# Create a data frame containing all the relevant variables
data = pandas.DataFrame({'x': x, 'y': y})

model = ols("y ~ x", data).fit()

print(model.summary())

model = ols("VIQ ~ Gender + 1", data).fit()
print(model.summary())

model = ols('VIQ ~ C(Gender)', data).fit()

data_fisq = pandas.DataFrame({'iq': data['FSIQ'], 'type': 'fsiq'})
data_piq = pandas.DataFrame({'iq': data['PIQ'], 'type': 'piq'})
data_long = pandas.concat((data_fisq, data_piq))
print(data_long)  

model = ols("iq ~ type", data_long).fit()
print(model.summary())

stats.ttest_ind(data['FSIQ'], data['PIQ']) 

model = ols('sepal_width ~ name + petal_length', data).fit()
print(model.summary())  

print(model.f_test([0, 1, -1, 0])) 
print(data)
seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],
                 kind='reg')

seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],
                 kind='reg', hue='SEX')

plt.rcdefaults()

seaborn.lmplot(y='WAGE', x='EDUCATION', data=data)


#could not resolve the bug sm
result = sm.ols(formula='wage ~ education + gender + education * gender', data=data).fit()    

print(result.summary())