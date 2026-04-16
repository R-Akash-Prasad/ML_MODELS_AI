import numpy as np 
import streamlit as st
import pandas as pd 
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,RandomForestRegressor,
                              GradientBoostingClassifier,GradientBoostingRegressor)
from sklearn.metrics import (mean_squared_error,r2_score,accuracy_score,precision_score
                             ,recall_score,f1_score)
from analysis import suggest_imporovements , generate_summary

st.set_page_config('ML Automation',page_icon='֎')
st.title('ML Automation')
st.subheader('Streamlit app to get CSV and Target as input and performs ml algo like trees , ensemble internally using AI to compare results and for suggestions ')
file = st.file_uploader('Upload file here',type=['csv','xls'])
if file:
    st.markdown('## Preview')
    df = pd.read_csv(file)
    st.dataframe(df.head())
    
    target = st.selectbox(':blue[select target]',df.columns)
    st.write(f':red[Target variable : ]{target}')
    
    if target:
        x = df.drop(columns=[target]).copy()
        y = df[target].copy()
        
        
        # PREPROCESSING
        
        num_cols = x.select_dtypes(include=np.number).columns.to_list()
        cat_cols = x.select_dtypes(include=object).columns.to_list()
        
        # MiSSING VALUE TREATMENT
        
        x[num_cols] = x[num_cols].fillna(x[num_cols].median())
        x[cat_cols] = x[cat_cols].fillna('Missing data')
        
        # Encoding 
        
        x = pd.get_dummies(data=x, columns=cat_cols,drop_first=True,dtype=int)
        
        # for categoric target 
        if y.dtype == 'object':
            label = LabelEncoder()
            y = label.fit_transform(y)
            
        # Detect the Problem Type
        if df[target].dtype == 'object' or len(np.unique(y)) <=10:
            problem_type = 'Classification'
        else:
            problem_type = 'Regression'
            
        st.write(f'## problem_type : {problem_type}')        
        
        #Train test Split
        xtrain , xtest , ytrain , ytest = train_test_split(x,y,random_state=42,test_size=0.2)
        
        for i in xtrain.columns:
            s = StandardScaler()
            xtrain[i] = s.fit_transform(xtrain[[i]])
            xtest[i] = s.transform(xtest[[i]])
            
        results = []
        if problem_type == 'Regression':
            models = {'Linear Regression' : LinearRegression() ,
                      'Random Forest' : RandomForestRegressor(random_state=42),
                      'Gradiant Bossting': GradientBoostingRegressor(random_state=42)}
            
            for name , model in models.items():
                model.fit(xtrain,ytrain)
                ypred = model.predict(xtest)
                
                results.append({'model Name' : name,
                               'R2 Score':round(r2_score(ytest,ypred),3),
                               'MSE':round(mean_squared_error(ytest,ypred),3),
                               'RMSE':round(np.sqrt(mean_squared_error(ytest,ypred)),3)})
                
        else:
             models = {'Logistic Regression' : LogisticRegression() ,
                      'Random Forest' : RandomForestClassifier(random_state=42),
                      'Gradiant Bossting': GradientBoostingClassifier(random_state=42)}
             
             for name , model in models.items():
                model.fit(xtrain,ytrain)
                ypred = model.predict(xtest)
                
                results.append({'model Name':name,
                               'Accuarcy':round(accuracy_score(ytest,ypred),3),
                               'Precision':round(precision_score(ytest,ypred),3),
                               'Recall':round(recall_score(ytest,ypred),3),
                               'F1':round(f1_score(ytest,ypred),3)})
                
        result_df = pd.DataFrame(results)
        st.write(f'#### :green[Results]')
        st.dataframe(result_df)
        
        if problem_type == 'Regression':
            st.bar_chart(result_df.set_index('model Name')['R2 Score'])
            st.bar_chart(result_df.set_index('model Name')['RMSE'])
        else:
            st.bar_chart(result_df.set_index('model Name')['Accuarcy'])
            st.bar_chart(result_df.set_index('model Name')['F1'])
            
        
        if st.button('Generate Summary'):
            summary = generate_summary(result_df)
            st.write(summary)
            
        if st.button('Suggest Improvement'):
            suggest = suggest_imporovements(result_df)
            st.write(suggest)