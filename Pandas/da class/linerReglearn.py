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
    df=pd.read_csv(r"C:\Users\PGCP-AI\Downloads\salary.csv")
    X=df[['YearsExperience']]
    y=df['Salary']

    return X, y


@app.cell
def _(X, train_test_split, y):
    X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=101)

    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Training the model
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
    plt.scatter(X_train,y_train,color='r')
    plt.plot(X_train,model.predict(X_train),color='b')
    plt.title('Salary Vs Experience (Training set')
    plt.xlabel('Years of Experience')
    plt.xlabel('Salary')
    plt.show()
    return


@app.cell
def _(mo):

    # Setup the input slider
    my_slider = mo.ui.slider(
        start=0, 
        stop=75, 
        step=5, 
        value=20,          # Default starting value
        label="Select Input Value: "
    )
    my_slider
    return (my_slider,)


@app.cell
def _(model, my_slider):
    # new_X_values_df=pd.DataFrame({'YearsExperience':[1,3,5,8,12,15,20,25,100]})
    new_y_pred=model.predict([[my_slider.value]])
    print('\nTredictions for new Experience values: \n',new_y_pred)
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


app._unparsable_cell(
    r"""
    from sklearn import 
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
