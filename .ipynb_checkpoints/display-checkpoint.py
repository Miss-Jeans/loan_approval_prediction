#import libraries
from dash import Input, Output, State, dcc, html, Dash,dash_table
from data import GraphBuilder
from model import ModelBuilder
from model import Model

#Instantiate GraphBuilder
gb=GraphBuilder()
#Instantiate model
model=Model().load()
X_train, X_test, y_train, y_test = model.get_train_test_data()
#Instantiate ModelBuilder with the test split
mb = ModelBuilder(X_test=X_test, y_test=y_test,model=model)
predictions_df = mb.make_predictions()

#instantiate app
app=Dash(__name__)
app.layout=html.Div(
    [
        html.H1("Applicant Demographics"),
        dcc.Dropdown(options=["Age","Income","Loan Intent","Prior Default","Credit History","Employment Duration","Home Ownership","Loan Amount","Loan Grade","Loan Status"],
                     value="Loan Status",
                     id="demo-plots-dropdown"),
        html.Div(id="demo-plots-display"),
        html.H1("The Model"),
        html.H2("Choose Threshold"),
        dcc.Slider(min=0.0,max=1.0,value=0.0,step=0.1,id="confusion_matrix_slider"),
        html.Div(id="confusion_matrix_display"),
        html.Div(id="classification_report_display"),
        html.H2("Feature Importances"),
        dcc.Graph(figure=mb.feature_importance(),id="gini_importance_display"),
        html.H2("Predictions"),
        dash_table.DataTable(data=predictions_df.to_dict("records"),columns=[{"name": col, "id": col} for col in predictions_df.columns],page_size=10,style_table={"overflowX": "auto"}
        )
           
    ]
)

@app.callback(Output("demo-plots-display","children"),Input("demo-plots-dropdown","value"))
def display_demo_graph(graph_name):
    """Serves applicant demograhic visualization.

    Parameters
    ----------
    graph_name : str
        User input given via 'demo-plots-dropdown'. Name of Graph to be returned.
        Options are "Age","Income","Loan Intent","Prior Default","Credit History","Employment Duration","Home Ownership".

    Returns
    -------
    dcc.Graph
        Plot that will be displayed in 'demo-plots-display' Div.
    """
    if graph_name=="Loan Status":
        fig=gb.build_loan_status()
    elif graph_name=="Age":
        fig=gb.build_age_hist()   
    elif graph_name=="Home Ownership":
        fig=gb.build_home_hist()
    elif graph_name=="Income":
        fig=gb.build_income_box() 
    elif graph_name=="Employment Duration":
        fig=gb.build_employment_bar()
    elif graph_name=="Loan Amount":
        fig=gb.build_amount_bar()
    elif graph_name=="Loan Grade":
        fig=gb.build_amount_bar()
    elif graph_name=="Credit History":
        fig=gb.build_history_bar()  
    elif graph_name=="Prior Default":
        fig=gb.build_default_bar()   
    else:
        graph_name=="Loan Intent"
        fig=gb.build_intent_bar()
    
    return dcc.Graph(figure=fig)
  

@app.callback([Output("confusion_matrix_display","children"),Output("classification_report_display", "children")],Input("confusion_matrix_slider","value"))
def confusion_matrix_plot(threshold_val):
    fig,fig_report=mb.make_cnf_matrix(threshold_val)
    return dcc.Graph(figure=fig),dcc.Graph(figure=fig_report)

    
