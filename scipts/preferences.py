import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка набора данных
data = pd.read_csv('ds1.csv')

# Вывод основной информации о наборе данных
print("Информация о наборе данных:")
data.info()
print("\nПервые строки набора данных:")
print(data.head())

# Выбор релевантных столбцов для анализа предпочтений
relevant_columns = [
    'socialNbFollowers', 'socialNbFollows', 'socialProductsLiked',
    'productsListed', 'productsSold', 'productsPassRate',
    'productsWished', 'productsBought', 'gender',
    'hasAnyApp', 'hasAndroidApp', 'hasIosApp',
    'daysSinceLastLogin', 'seniorityAsMonths'
]

# Создание поднабора данных для анализа
subset = data[relevant_columns]

# Проверка на наличие пропущенных значений
print("\nПропущенные значения:")
print(subset.isnull().sum())

# Обработка пропущенных значений (если есть)
subset = subset.dropna()

# Преобразование категориальных столбцов в числовые значения для корреляционного анализа
subset['gender'] = subset['gender'].astype('category').cat.codes

# Матрица корреляций для выявления взаимосвязей
correlation_matrix = subset.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Матрица корреляций факторов")
plt.show()

# Визуализация ключевых факторов, влияющих на предпочтения товаров
# Пример: Понравившиеся продукты vs Подписчики в соцсетях
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=subset,
    x='socialNbFollowers',
    y='socialProductsLiked',
    hue='gender',
    alpha=0.7
)
plt.title("Понравившиеся продукты vs Подписчики в соцсетях")
plt.xlabel("Количество подписчиков в соцсетях")
plt.ylabel("Количество понравившихся продуктов")
plt.legend(title="Пол")
plt.show()

# Пример: Купленные продукты vs Желанные продукты
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=subset,
    x='productsWished',
    y='productsBought',
    hue='hasAnyApp',
    alpha=0.7
)
plt.title("Купленные продукты vs Желанные продукты")
plt.xlabel("Количество желанных продуктов")
plt.ylabel("Количество купленных продуктов")
plt.legend(title="Наличие приложения")
plt.show()

# Описательная статистика
print("\nОписательная статистика:")
print(subset.describe())

# Проверка гипотез: Пример с использованием T-теста или ANOVA
from scipy.stats import ttest_ind

group1 = subset[subset['hasAnyApp'] == 1]['productsBought']
group2 = subset[subset['hasAnyApp'] == 0]['productsBought']

stat, p_value = ttest_ind(group1, group2)
print("\nРезультаты T-теста:")
print(f"Статистика: {stat}, P-значение: {p_value}")
if p_value < 0.05:
    print("Существуют статистически значимые различия между группами.")
else:
    print("Статистически значимых различий между группами не обнаружено.")
