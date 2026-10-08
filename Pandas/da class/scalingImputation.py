import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import seaborn as sns
    from sklearn.preprocessing import OneHotEncoder,StandardScaler
    from sklearn.impute import SimpleImputer

    return OneHotEncoder, SimpleImputer, StandardScaler, pd, sns


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(OneHotEncoder, pd, sns):
    df=sns.load_dataset('titanic')
    df.head()
    selected_columns=['survived','pclass','sex','age','sibsp','parch','embarked']
    X=df.drop('survived',axis=1)
    df=df[selected_columns]
    # one hot encoding 
    Categorical_features=['pclass','sex','sibsp','parch','embarked']
    encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    # sparse output false returns an standard np arrray, not scipy array(which is the default)
    # ignore unknown values during transform
    encoded_data=encoder.fit_transform(df[Categorical_features]) # fill encoder on the catagorical features
    feature_names=encoder.get_feature_names_out(Categorical_features) # get teh new frature names (optional)
    # create new data frame with the encoded data and featurre names 
    encoded_df=pd.DataFrame(encoded_data,columns=feature_names)

    return X, df, encoded_df


@app.cell
def _(SimpleImputer, X):
    missing_values_age=X['age'].isna().any()
    if missing_values_age:
        print('Handilling nulls............')
        imputer=SimpleImputer(strategy='mean')
        imputer.fit(X[['age']])
        X['age']=imputer.transform(X[['age']])[:, 0] 
    print(X.age)

    
    return


@app.cell
def _(StandardScaler, X):
    # standard scale the age features(after imputations)
    scaler=StandardScaler()
    scaler.fit(X[['age']])
    X_scaled=X.copy()# create copy to avoide the modifying original data
    print(X_scaled)

    return X_scaled, scaler


@app.cell
def _(X, X_scaled, encoded_df, pd, scaler):
    def _():
        X_scaled.loc[:,'age']=scaler.transform(X[['age']])
        print(X_scaled)
        df_merged= pd.concat([encoded_df,X_scaled['age']],axis=1)
        return print(df_merged.head())


    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
