import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge":            Ridge(alpha=1.0),
        "Lasso":            Lasso(alpha=1.0, max_iter=5000),
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained[name] = model
        joblib.dump(model, f"models/{name}.pkl")
        print(f"✅ {name} trained and saved!")
    joblib.dump(scaler, "models/scaler.pkl")
    return trained, scaler, X_train_scaled, X_test_scaled, y_train, y_test
