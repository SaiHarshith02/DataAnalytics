import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import marimo as mo

    return mo, pd, train_test_split


@app.cell
def _(pd):
    df=pd.read_csv(r"C:\Users\PGCP-AI\Downloads\housing.csv")
    df.head(3)
    return (df,)


@app.cell
def _(df):
    cols=df.columns
    cols
    return


@app.cell
def _(df):
    X=df[['LotArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea',
           'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageArea',
           ]]
    y=df[['SalePrice']]
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Train test split
    """)
    return


@app.cell
def _(X, train_test_split, y):
    X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=101)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    model training
    """)
    return


@app.cell
def _(X_train, y_train):
    from sklearn.linear_model import LinearRegression
    model=LinearRegression()
    model.fit(X_train,y_train)
    return (model,)


@app.cell
def _(X_test, model):
    y_pred=model.predict(X_test)
    print('Pridected values on test set :', y_pred)
    return (y_pred,)


@app.cell
def _(X_train, model, y_train):
    import matplotlib.pyplot as plt
    plt.scatter(X_train['LotArea'],y_train,color='r')
    plt.plot(X_train,model.predict(X_train),color='b')
    plt.title('Living area Vs price (Training set')
    plt.xlabel('Living area')
    plt.xlabel('Price')
    plt.show()
    return


@app.cell
def _(y_pred, y_test):
    from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
    mae=mean_absolute_error(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    print('model Evalution Metrics: ')
    print('mean Absolute Error (MAE):',mae)
    print('mean Squared Error (MSE):',mse)
    print('r2 score :',r2)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
