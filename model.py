#Import Libraries
import os
import joblib
from glob import glob
from data import Data
import numpy as np
import pandas as pd
import plotly.express as px
#Model Building
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from category_encoders import OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score,classification_report,confusion_matrix

class Model:
    """Methods for Training the model"""
    
    def __init__(self,data=None):
        """init
        
        Parameter
        ----------
        data : Data, optional
        Data source, by default None
        """
        self.data=data if data is not None else Data()
        self.fitted_model = None

    def get_train_test_data(self):
        """Splitting the data into target and feature, with 80% for training"""

        #Get Data
        df = self.data.wrangle_data()
        
        # Separate features and target
        target_col="loan_status"
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split into train and test sets
        return train_test_split(X, y, test_size=0.2, random_state=42)
        
               
        
    def GradientModel(self):
        """Building the Model and save to self.fitted_model""" 
        
        #Extracting the Training data
        X_train,X_test,y_train,y_test=self.get_train_test_data()
        
        #Making a pipeline with the GradientBoostingClassifier,SimpleInmputer and One Hot Encoding
        clf=make_pipeline(OrdinalEncoder(),SimpleImputer(),GradientBoostingClassifier(random_state=42))
        
        #Perfoming a Grid Search
        params={"simpleimputer__strategy":["mean","median"],
        "gradientboostingclassifier__max_depth":range(2,5),
        "gradientboostingclassifier__n_estimators":range(25,31,5)}
        
        #Compute class weights for y_train for class balance
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        grid_search=GridSearchCV(clf,param_grid=params,cv=5,n_jobs=-1)
        
        #Fit the model
        grid_search.fit(X_train,y_train,gradientboostingclassifier__sample_weight=sample_weights)
        
        #Extract the best estimator and save to self.fitted_model
        self.fitted_model=  grid_search
        return self
        
        
    def dump(self):
        """Save Model in the current directory with timestamp e.g......data/model_20260823.pkl"""
        
        #create timestamp
        timestamp=pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        #Create a path
        filepath=os.path.join("data",f"{timestamp}_credit_risk.pkl")
        
        #Save model
        joblib.dump(self.fitted_model,filepath)
        
        #Return filepath
        return filepath
        
    def load(self):
        """
        Load Model from the data directory, to make predictions
        attaches to self.model
        returns None
        """
        #create search path
        search_path = os.path.join("data", "*credit_risk*.pkl")
        model_path=sorted(glob(search_path))[-1]
        
        #load model and assign to model.self
        self.fitted_model=joblib.load(model_path)
        return self

    def predict_proba(self, X):
        """Delegates prediction to the underlying scikit-learn model"""
        if self.fitted_model is None:
           self.load()
        return self.fitted_model.predict_proba(X)   
            

class ModelBuilder:
    def __init__(self,X_test,y_test,model=Model()):
        self.model = model   
        self.X_test=X_test
        self.y_test=y_test

    def make_cnf_matrix(self,threshold):
        """Creates Confusion Matrix Heatmap and Classification Report Display"""
        
        y_pred_proba=self.model.load().predict_proba(self.X_test)[:,1]
        y_pred=(y_pred_proba>threshold).astype(int)
        
        #Create a Confusion Matrix
        cm=confusion_matrix(self.y_test,y_pred)
        
        #Create a confusion heatmap
        fig = px.imshow(
            cm,
            x=["Not Approved (False)", "Approved (True)"],
            y=["Not Approved (False)", "Approved (True)"],
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
            title=f"Confusion Matrix (Threshold = {threshold})"
        )

        #Generate Classification Report as a Dictionary
        report_dict = classification_report(self.y_test, y_pred, target_names=["Not Approved", "Approved"],output_dict=True,zero_division=0)
        
        #Convert report dictionary into a DataFrame (excluding accuracy and support)
        report_df = pd.DataFrame(report_dict).T
        metrics_df = report_df.loc[["Not Approved", "Approved", "macro avg", "weighted avg"], ["precision", "recall", "f1-score"]]
        
        #Plot Classification Report Heatmap
        fig_report = px.imshow(
            metrics_df,
            text_auto=".2",              
            color_continuous_scale="Viridis",
            zmin=0.0,
            zmax=1.0,
            labels=dict(x="Metric", y="Class / Average", color="Score"),
            title=f"Classification Report Heatmap (Accuracy: {round(report_dict['accuracy'],2)})"
        )
        
        
        #return fig and classification_report
        return fig,fig_report
        
    def feature_importance(self):
        """Extracts feature importances from the fitted model and returns a plot of the Gini Impunity"""
        
        #Extracting the feature importances of the model's best estimator, converting to a series
        pipeline=self.model.fitted_model.best_estimator_
        importances=pipeline.named_steps["gradientboostingclassifier"].feature_importances_
        feat_imp=pd.Series(importances,index=self.X_test.columns).sort_values()
        
        #Plotting a horizontal bar plot
        fig=px.bar(x=feat_imp.values,y=feat_imp.index,orientation="h",title="Gini Impunity")
        fig.update_layout(xaxis_title="Gini Importance",yaxis_title="Feature")
        
        #Return fig
        return fig
        
    def make_predictions(self):
        """Generates predictions on X_test
        Returns DataFrame
        """
        #Generating predictions from the Loaded Model
        prediction=self.model.load().predict(self.X_test)
        
        # Create DataFrame with Index (Applicant ID / Row Number) and Prediction
        df_pred = pd.DataFrame({
        "Applicant Index": self.X_test.index,
        "Predicted Loan Status": np.where(prediction, "Approved", "Not Approved")
        })
        
        #Return DataFrame
        return df_pred
        
        
        