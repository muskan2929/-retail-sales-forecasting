import joblib

def predict(features, model_name="Ridge"):
    model  = joblib.load(f"models/{model_name}.pkl")
    scaler = joblib.load("models/scaler.pkl")
    scaled = scaler.transform([features])
    return model.predict(scaled)[0]
