#import libraries

#Data Handling
import pandas as pd
import numpy as np
#Data Visualization
import plotly.express as px


class Data:
    """For importing and interracting with the data"""
    def __init__(self,filepath="data/credit_risk_dataset.csv"):
        self.filepath=filepath

    def wrangle_data(self,filepath=None):
        """Importing data from a filepath to Dataframe
        parameters
        ------
        filepath:str
        returns:DataFrame
        """
        #import data from csv file to a dataframe and clean
        path =filepath if filepath is not None else self.filepath
        df=pd.read_csv(path)
        
        #Converting the loan_status 
        df["loan_status"]=df.loan_status.astype(bool)
        
        #Clustering the ages in the person_age column
        #Define bin boundaries and corresponding labels
        bins = [0, 25, 29,34,39, 44, 49, 54,59,64,69, float('inf')]
        labels = [
            "Under 25", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54","55-59","60-64","65-69","70 and Above"
        ]
        
        # Create an age_group column
        df["age_group"] = pd.cut(df["person_age"], bins=bins, labels=labels, right=True)
        
        #Drop the person_age column
        df.drop(columns="person_age",inplace=True)
        
        #return DataFrame
        return df

class GraphBuilder:
    """Methods for building Graphs."""
    def __init__(self,data=None):
        """init
        
        Parameters
        ----------
        data : Data, optional
        Data source, by default None
        """
        self.data = data if data is not None else Data()
        
    def build_age_hist(self):
        """Creates Age by Approval Status Bar.
        Returns
        -------
        Figure
        """     
        # Get data from wrangle data method
        df_age=self.data.wrangle_data()
        
        # Create Figure   
        fig=px.histogram(data_frame=df_age,x="age_group",color="loan_status",barmode="group",title="Loan Application Distribution :Age vs Loan Status")
        
        #Axes label
        fig.update_layout(xaxis_title="Age Group",yaxis_title="Frequency(count)")
        
        # Return Figure
        return fig 
        
    def build_home_hist(self):
        """Creates Home Ownership by Approval Status Bar.
        Returns
        -------
        Figure
        """     
        # Get data from wrangle data method
        df_home=self.data.wrangle_data()
        
        # Create Figure   
        fig=px.histogram(data_frame=df_home,x="person_home_ownership",color="loan_status",barmode="group",title="Home Ownership Distribution by Loan Status")
        
        #Axes label
        fig.update_layout(xaxis_title="Home Ownership",yaxis_title="Frequency(count)")
        
        # Return Figure
        return fig 

    def build_income_box(self):
        """Creates Income by Approval Status Boxplot.
        Returns
        -------
        Figure
        """     
        # Get data from wrangle data method
        df_income=self.data.wrangle_data()
        
        #Trim Data,to remove outliers
        q1,q9=df_income["person_income"].quantile([0.1,0.9])
        mask=df_income["person_income"].between(q1,q9)
        
        # Create Figure   
        fig=px.box(data_frame=df_income[mask],x="loan_status",y="person_income",title="Income Distribution by Loan Approval Status")
        
        #Axes label
        fig.update_layout(xaxis_title="Loan Approval Status",yaxis_title="Income")
        
        # Return Figure
        return fig 

         
    def build_intent_bar(self):
        """Creates Loan Usage by Approval Status horizontal Bar.
        Returns
        -------
        Figure
        """     
        # Get data from wrangle data method
        df=self.data.wrangle_data()
        
        # Create Figure   
        fig=px.histogram(data_frame=df,y="loan_intent",color="loan_status",barmode="group",title="Distribution of Loan Application Usage by Approval Status")
        
        #Axes label
        fig.update_layout(yaxis_title="Loan Intention",xaxis_title="Frequency(count)")
        
        # Return Figure
        return fig 
        
    def build_employment_bar(self):
        """Creates employment duration by Approval Status Bar.
        Returns
        -------
        Figure
        """     
        # Get data from wrangle data method
        df=self.data.wrangle_data()
        
        # Create Figure   
        fig=px.histogram(data_frame=df,x="person_emp_length",color="loan_status",barmode="group",title="Loan Approval by Employment Length")
        
        #Axes label
        fig.update_layout(xaxis_title="Employment Length",yaxis_title="Frequency(count)")
        fig.update_xaxes(range=[0, 25])
        
        # Return Figure
        return fig
        
    def build_amount_bar(self):
        """Creates Principal Loan Amount Applied Distribution plot.
        Returns
        -------
        Figure
        """     
        # Get data from wrangle data method
        df=self.data.wrangle_data()
        
        # Create Figure   
        fig=px.histogram(data_frame=df,x="loan_amnt",nbins=13,title="Distribution of the Principal Loan Applied")
        
        #Axes label
        fig.update_layout(xaxis_title="Principal Loan Amount",yaxis_title="Frequency(count)")
        fig.update_yaxes(range=[0, 12000],tick0=1000,dtick=1000)
        
        # Return Figure
        return fig 
        
    def build_grade_bar(self):
        """Creates loan grade by Approval Status.
        Returns
        -------
        Figure
        """     
        # Get data from wrangle data method
        df=self.data.wrangle_data()
        
        #Calculating the proportions of the class distribution
        majority,minority=df.loan_status.value_counts(normalize=True)

        #Creating a pivot table aggregating by the mean of the classes
        df_grade=pd.pivot_table(df,index="loan_grade",values="loan_status",aggfunc=np.mean)
        
        # Create Figure   
        fig=px.bar(data_frame=df_grade,x="loan_status",title="proportion of Loan Grade vs Loan Approval Status")
        
        #Add Vertical Lines
        fig.add_vline(
        x=majority, 
        line_dash="dash",
        line_color="red",
        annotation_text="Majority Class",
        annotation_position="bottom left"
        )
        
        fig.add_vline(
            x=minority,
            line_dash= "dot",
            line_color="yellow",
            annotation_text="Minority Class",
            annotation_position="bottom left"
            )

        #Axes label
        fig.update_layout(xaxis_title="Loan Approval Status",yaxis_title="Loan Grade")
        
        # Return Figure
        return fig 
    
    def build_history_bar(self):
        """Creates Credit History by Approval Status Bar.
        Returns
        -------
        Figure
        """     
        # Get data from wrangle data method
        df=self.data.wrangle_data()
        credit_history=df.groupby("cb_person_cred_hist_length")["loan_status"].mean().sort_index().to_frame() 
        
        # Create Figure   
        fig=px.bar(data_frame=credit_history,y="loan_status",x=credit_history.index,title="Approval Rate by Credit History Length(Years)")
        
        #Axes label
        fig.update_layout(xaxis_title="Credit Hstory Length(Years)",yaxis_title="Loan Approval")
        
        # Return Figure
        return fig 
        
    def build_default_bar(self):
        """Creates prior Loan Default by Approval Status Bar.
        Returns
        -------
        Figure
        """     
        # Get data from wrangle data method
        df=self.data.wrangle_data()
     
        # Create Figure   
        fig=px.bar(data_frame=df,x="cb_person_default_on_file",color="loan_status",title="Prior Loan Default")
        
        #Axes label
        fig.update_layout(xaxis_title="Prior Default",yaxis_title="Frequency")
        
        # Return Figure
        return fig 
        
    def build_loan_status(self):
        """Creates Loan Status bar plot.
        Returns
        -------
        Figure
        """     
        # Get data from wrangle data method
        df=self.data.wrangle_data()
        loan_count=df.loan_status.value_counts(normalize=True)  
     
        # Create Figure   
        fig=px.bar(data_frame=loan_count,title="Class Distribution: Approval Status ")
        
        #Axes label
        fig.update_layout(xaxis_title="Approval Status",yaxis_title="Relative Frequency")
        
        #Return Figure
        return fig    
        
    

  
        
    
    