import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(name, y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    print(f"\n========================================")
    print(f"  {name}")
    print(f"  MSE  : {mse:,.2f}")
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  R2   : {r2:.4f}")
    return {"Model": name, "MSE": mse, "RMSE": rmse, "R2": r2}

def plot_results(y_test, predictions_dict):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(predictions_dict), figsize=(16, 5))
    for ax, (name, preds) in zip(axes, predictions_dict.items()):
        ax.scatter(y_test, preds, alpha=0.3, s=10, color="steelblue")
        ax.plot([0, y_test.max()], [0, y_test.max()], "r--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Actual Sales")
        ax.set_ylabel("Predicted Sales")
    plt.tight_layout()
    plt.savefig("reports/model_comparison.png", dpi=150)
    plt.show()
