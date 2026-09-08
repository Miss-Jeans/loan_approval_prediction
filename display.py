#import libraries
from dash import Input, Output, State, dcc, html, Dash,dash_table
from data import GraphBuilder,Data
from model import ModelBuilder
from model import Model
from database import Repository
from experiment import Experiment
import pandas as pd
import os
from dotenv import load_dotenv

#Instantiate GraphBuilder
gb=GraphBuilder()
#Instantiate model
model=Model().load()
#Get Training and test sets
X_train, X_test, y_train, y_test = model.get_train_test_data()
#Instantiate ModelBuilder with the test split
mb = ModelBuilder(X_test=X_test, y_test=y_test,model=model)
predictions_df = mb.make_predictions()
#Load environment variables 
load_dotenv()       
#Retrieve URI
mongo_uri = os.environ.get("MONGO_URI")
#Instantiate Repository
repo=Repository(mongo_uri,records=Data())
Mdl=Model()
#Instantiate Experiment
Exp=Experiment(repo,Mdl)


NUMERICAL_CONFIG = {
    'input_income': {'col': 'person_income', 'label': 'Annual Income ($):', 'step': 1000, 'curr': True},
    'input_loan_amnt': {'col': 'loan_amnt', 'label': 'Loan Amount ($):', 'step': 500, 'curr': True},
    'input_int_rate': {'col': 'loan_int_rate', 'label': 'Interest Rate (%):', 'step': 0.1, 'pct': True},
    'input_cred_hist': {'col': 'cb_person_cred_hist_length', 'label': 'Credit History (Years):', 'step': 1},
    'input_loan_pct': {'col': 'loan_percent_income', 'label': 'Loan Income (%):', 'step': 0.1, 'pct': True},
    'input_emp_length': {'col': 'person_emp_length', 'label': 'Employment Length (Years):', 'step': 1},
}

CATEGORICAL_CONFIG = {
    'input_age': {'col': 'age_group', 'label': 'Applicant Age Group'},
    'input_home_ownership': {'col': 'person_home_ownership', 'label': 'Home Ownership'},
    'input_loan_intent': {'col': 'loan_intent', 'label': 'Loan Intent'},
    'input_loan_grade': {'col': 'loan_grade', 'label': 'Loan Grade'},
    'input_historical_default': {'col': 'cb_person_default_on_file', 'label': 'Prior Default'}
}

def make_slider(cfg, slider_id):
    col, step = cfg['col'], cfg['step']
    mn, mx = float(X_train[col].min()), float(X_train[col].max())
    val = float(X_train[col].mean())
    val = round(val) if step >= 1 else round(val, 2)
    
    fmt = (lambda x: f"${int(x):,}") if cfg.get('curr') else ((lambda x: f"{x}%") if cfg.get('pct') else str)
    marks = {int(x): fmt(int(x)) for x in X_train[col].quantile([0, 0.5, 1.0])}
    
    return dcc.Slider(min=mn, max=mx, step=step, value=val, marks=marks, id=slider_id)

# Pre-render input elements dynamically
num_inputs = [
    html.Div([
        html.Label(cfg['label'], style={'fontWeight': 'bold'}),
        make_slider(cfg, cid)
    ], style={'marginBottom': '15px'}) 
    for cid, cfg in NUMERICAL_CONFIG.items()
]

cat_inputs = [
    html.Div([
        html.Label(cfg['label'], style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id=cid, 
            options=sorted(X_train[cfg['col']].unique()), 
            value=sorted(X_train[cfg['col']].unique())[0],
            clearable=False
        )
    ], style={'marginBottom': '15px'}) 
    for cid, cfg in CATEGORICAL_CONFIG.items()
]

#instantiate app
app=Dash(__name__,suppress_callback_exceptions=True)

#Instantiate Server to run on Render
server=app.server

app.layout=html.Div(
    [
        html.H1("Predictive Modeling and A/B Testing"),
        html.H2("Choose your Option from the dropdown menu"),
        dcc.Dropdown(options=["Applicant Demographics","The Model","The Experiment"],value="Applicant Demographics",id="Project_phase_dropdown",clearable=False),
        html.Hr(),
        html.Div(id="Project-output-display")
    ]
) 

@app.callback(Output("Project-output-display", "children"),Input("Project_phase_dropdown", "value"))
def render_project_phase(selected_phase):
    """Dynamically returns the layout phases based on the dropdown selection."""
    
    # APPLICANT DEMOGRAPHICS
    if selected_phase == "Applicant Demographics":
        return html.Div([
            html.H1("Applicant Demographics"),
            dcc.Dropdown(
                options=["Age", "Income", "Loan Intent", "Prior Default", 
                         "Credit History", "Employment Duration", "Home Ownership", 
                         "Loan Amount", "Loan Grade", "Loan Status"],
                value="Loan Status",
                id="demo-plots-dropdown",
                clearable=False
            ),
            html.Div(id="demo-plots-display")
        ])
    
    # THE MODEL
    elif selected_phase == "The Model":
        return html.Div([
            html.H1("The Model"),
            html.H2("Select Prediction Mode"),
            dcc.RadioItems(
                id="prediction_mode_radio",
                options=[
                    {"label": " View Batch Test Predictions", "value": "batch"},
                    {"label": " Predict Single Applicant", "value": "single"}
                ],
                value="batch",
                inline=True,
                style={"marginBottom": "20px", "fontSize": "16px"}
            ),
            
            # Batch View Container
            html.Div(id="batch_predictions_container", children=[
                html.H2("Choose Threshold"),
                dcc.Slider(min=0.0, max=1.0, value=0.5, step=0.05, id="confusion_matrix_slider"),
                html.Div(id="model_evaluation_results"),
                html.H2("Feature Importances"),
                dcc.Graph(figure=mb.feature_importance(), id="gini_importance_display"),
                html.H2("Predictions"),
                dash_table.DataTable(
                    data=predictions_df.to_dict("records"),
                    columns=[{"name": col, "id": col} for col in predictions_df.columns],
                    page_size=10,
                    style_table={"overflowX": "auto"}
                )
            ]),
            
            # Single Applicant View Container
            html.Div(id="single_prediction_container", style={"display": "none"}, children=[
                html.H2("Single Applicant Risk Assessment"),
                html.Div([
                    html.Div(num_inputs, style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
                    html.Div(cat_inputs + [
                        html.Br(),
                        html.Button("Predict Risk", id="btn_predict_single", n_clicks=0, 
                                    style={"backgroundColor": "#007bff", "color": "white", "padding": "10px 20px", "border": "none", "borderRadius": "4px", "cursor": "pointer"})
                    ], style={"width": "48%", "display": "inline-block", "marginLeft": "4%", "verticalAlign": "top"})
                ]),
                html.Hr(),
                html.Div(id="single_prediction_output", style={"marginTop": "20px"})
            ])
        ])
    
    # THE EXPERIMENT
    else:
        return html.Div([
            html.H1("The Experiment"),
            html.H2("Choose Risk Threshold"),
            dcc.Slider(min=0.00, max=0.50, value=0.03, step=0.01, id="Experiment_risk_slider"),
            html.Div(id="Experiment_results")
        ])    

# Demographics Plot Callback
@app.callback(Output("demo-plots-display", "children"),Input("demo-plots-dropdown", "value"),prevent_initial_call=False)
def display_demo_graph(graph_name):
    if graph_name == "Loan Status":
        fig = gb.build_loan_status()
    elif graph_name == "Age":
        fig = gb.build_age_hist()   
    elif graph_name == "Home Ownership":
        fig = gb.build_home_hist()
    elif graph_name == "Income":
        fig = gb.build_income_box() 
    elif graph_name == "Employment Duration":
        fig = gb.build_employment_bar()
    elif graph_name == "Loan Amount":
        fig = gb.build_amount_bar()
    elif graph_name == "Loan Grade":
        fig = gb.build_grade_bar()
    elif graph_name == "Credit History":
        fig = gb.build_history_bar()  
    elif graph_name == "Prior Default":
        fig = gb.build_default_bar()   
    else:
        fig = gb.build_intent_bar()
    
    return dcc.Graph(figure=fig)


# Model Evaluation Callback
@app.callback(Output("model_evaluation_results", "children"),Input("confusion_matrix_slider", "value"))
def confusion_matrix_plot(threshold_val):
    result = mb.make_cnf_matrix(threshold_val)
    
    return html.Div([
        dcc.Graph(figure=result["fig"]),
        dcc.Graph(figure=result["fig_report"]),
        html.H3(f"Model ROC-AUC: {round(result['Roc'], 2)}")
    ])

# Prediction Mode Toggle Callback
@app.callback(
    [Output("batch_predictions_container", "style"),
     Output("single_prediction_container", "style")],
    Input("prediction_mode_radio", "value")
)
def toggle_prediction_mode(selected_mode):
    if selected_mode == "single":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


# Single Applicant Prediction Inference Callback
@app.callback(
    Output("single_prediction_output", "children"),
    Input("btn_predict_single", "n_clicks"),
    [State("input_income", "value"),
     State("input_loan_amnt", "value"),
     State("input_int_rate", "value"),
     State("input_cred_hist", "value"),
     State("input_loan_pct", "value"),
     State("input_emp_length", "value"),
     State("input_age", "value"),   
     State("input_home_ownership", "value"),
     State("input_loan_intent", "value"),
     State("input_loan_grade", "value"),
     State("input_historical_default", "value")],
    prevent_initial_call=True
)
def predict_single_applicant(n_clicks, income, loan_amnt, int_rate, cred_hist, loan_pct_income, duration, age, home, intent, grade, default_rec):
    input_dict = {
    'person_income': income,
    'loan_amnt': loan_amnt,
    'loan_int_rate': int_rate,
    'cb_person_cred_hist_length': cred_hist,
    'loan_percent_income': loan_pct_income,   
    'person_emp_length': duration,
    'age_group': age,
    'person_home_ownership': home,
    'loan_intent': intent,
    'loan_grade': grade,
    'cb_person_default_on_file': default_rec
}
    
    input_data = pd.DataFrame([input_dict])[X_train.columns]
    p_default = float(model.predict_proba(input_data)[:, 1][0])
    
    decision = "REJECT / HIGH RISK" if p_default > 0.15 else "APPROVE / LOW RISK"
    card_color = "#f8d7da" if p_default > 0.15 else "#d4edda"
    text_color = "#721c24" if p_default > 0.15 else "#155724"
    
    return html.Div([
        html.H3(f"Predicted Default Probability: {p_default:.2%}"),
        html.H3(f"Recommendation (Optimal Cutoff = 0.15): {decision}")
    ], style={
        "padding": "20px", 
        "backgroundColor": card_color, 
        "color": text_color, 
        "borderRadius": "8px", 
        "textAlign": "center"
    })    

@app.callback(Output("Experiment_results", "children"), Input("Experiment_risk_slider", "value"))
def exp_results(threshold):
    if threshold is None:
        return html.Div("Adjust the slider to evaluate risk threshold.")

    # Get Contingency Table DataFrame
    ct_df = Exp.get_contingency_table(risk_threshold=threshold)

    # Format index for display
    ct_df = ct_df.reset_index()

    # Convert DataFrame to Dash DataTable
    contingency_table_component = html.Div(
        [
            html.H4("A/B Group Contingency Table"),
            dash_table.DataTable(
                data=ct_df.to_dict("records"),
                columns=[{"name": str(i), "id": str(i)} for i in ct_df.columns],
                style_cell={"textAlign": "center", "padding": "8px"},
                style_header={
                    "fontWeight": "bold",
                    "backgroundColor": "#f2f2f2",
                },
            ),
        ],
        style={"marginBottom": "20px"},
    )

    # Get ROI Component
    roi_component = Exp.get_ROI(risk_threshold=threshold)

    # Get Chi_square components
    result=Exp.chi_square(risk_threshold=threshold)
    
    # Return contigency_table and KPI
    return html.Div(
        [
            contingency_table_component,
            roi_component,
            html.Div(
                [
                    html.H3("Chi-Square Test of Independence"),
                    html.P(f"p-value: {result.pvalue:.4f}"),
                    html.P(f"Statistic: {result.statistic:.4f}"),
                    html.P(f"Verdict: {'Statistically Significant!, threshold sigificantly increases loan recovery' if result.pvalue < 0.05 else 'Statistically Insignificant!'}", 
                           style={"fontWeight": "bold", "color": "#007bff"},
                          ),
                ],
                style={"marginTop": "20px"},
            ),
        ]
    )