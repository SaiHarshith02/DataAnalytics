import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix,classification_report, ConfusionMatrixDisplay
    from sklearn.preprocessing import StandardScaler

    #1.Load and prepare data
    df =pd.read_csv(r"C:\Users\PGCP-AI\Downloads\diabetes.csv")

    X=df.drop("Outcome", axis=1)
    y= df["Outcome"]

    #2. Train-Test split
    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=101, stratify=y)


    scaler= StandardScaler()
    X_train["Insulin"]= scaler.fit_transform(X_train[["Insulin"]])
    X_test["Insulin"]= scaler.transform(X_test[["Insulin"]])


    model=LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)


    y_pred= model.predict(X_test)


    cm=confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No Diabetes", "Diabetes"]).plot(cmap="Blues")
    plt.title("Confusion Matrix(Test set)")
    plt.show()


    print("\n== Test set Performance ===")
    print("\nClassification|_report(y_test")




    X_train_glucose= X_train[["Glucose"]].copy()
    X_test_glucose= X_test[["Glucose"]].copy()
    model_glucose=LogisticRegression(max_iter=500)
    model_glucose.fit(X_train_glucose,y)


    #create smooth curve
    glucose_min= df["Glucose"].min()
    glucose_max= df["Glucose"].max()
    glucose_values= np.linspace(glucose_min, glucose_max, 200).reshape(-1,1)
    y_curve= model_glucose.predict_proba(glucose_values)[:, 1]


    plt.figure(figsize=(8,5))
    plt.plot(glucose_values, y_curve, color="green", linwidth=2, label="Logistic Curve")
    plt.scatter(X_train_glucose, y_train, color="blue", alpha=0.4, label="Training Data")
    plt.scatter(X_test_glucose, y_test, color="red",alpha=0.6, marker='x', s=80, label="Test Data")
    plt.xlabel("Glucose level")

    plt.ylabel("Probability of diabetes(outcome =1")
    plt.title(" Logistic regression curve- Glucose Feature Only")
    plt.legend()
    plt.grid(True,alpha=0.3)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
