import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Загрузка данных
data = pd.read_csv('ds1.csv')

# Предобработка данных
# Выбор релевантных признаков для прогнозирования вероятности покупки
features = ['socialNbFollowers', 'socialNbFollows', 'socialProductsLiked',
            'productsListed', 'productsSold', 'productsPassRate',
            'productsWished', 'gender', 'hasAnyApp', 'daysSinceLastLogin',
            'seniorityAsMonths']

# Целевая переменная: совершал ли пользователь покупки
# Создание бинарной целевой переменной

data['bought_something'] = data['productsBought'] > 0

# Кодирование категориальных переменных
categorical_features = ['gender', 'hasAnyApp']
label_encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature].astype(str))
    label_encoders[feature] = le

# Заполнение пропущенных значений
data.fillna(0, inplace=True)

# Стандартизация числовых признаков
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])

# Разделение данных
X = scaled_features
y = data['bought_something']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Прогнозирование
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Оценка модели
print("Отчёт классификации:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# Объяснение результатов
# Важность признаков
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("\nВажность признаков:\n", feature_importance_df)

# Сохранение модели и кодировщиков
import joblib
joblib.dump(model, 'purchase_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
