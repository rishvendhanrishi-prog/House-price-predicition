from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import joblib


housing = fetch_california_housing()
X = housing.data
y = housing.target



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)


print("R² Score:", r2_score(y_test, prediction))
print("MAE:", mean_absolute_error(y_test, prediction))
joblib.dump(model, "model.pkl")
print("Model saved!")