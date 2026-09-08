import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import Table2x2
from database import Repository
from model import Model
import plotly.express as px
from dash import html

class Experiment:

    def __init__(self, repo=None, Mdl=None):
        # Prevent mutable default argument issues
        self.repo = repo if repo is not None else Repository()
        self.Mdl = Mdl if Mdl is not None else Model()
        self.model = self.Mdl.load()

    def _prepare_experiment_df(self, risk_threshold, records=None):
        """Helper method to load data, predict probabilities, and apply thresholding."""
        if records is None:
            records = self.repo.assign_to_groups(observations=None)

        df = pd.DataFrame(records)

        if df.empty:
            raise ValueError("No records returned from repository.")

        # Ensure we predict probabilities on the current DataFrame's features
        # Extract features used by the model
        feature_cols = self.Mdl.X_train.columns if hasattr(self.Mdl, "X_train") else None
        
        if feature_cols is not None:
            X = df[feature_cols]
        else:
            # Fallback if X_train isn't exposed directly
            X_train, _, _, _ = self.Mdl.get_train_test_data()
            X = df[X_train.columns]

        # Predict probability of rejection 
        df["p_reject"] = self.model.predict_proba(X)[:, 0]

        # Initialize baseline
        df["reconsidered_approval"] = 0

        # Apply risk threshold to Treatment group
        treatment = df["group"] == "Treatment(Adjusted_risk_Threshold)"
        below_risk = df["p_reject"] <= risk_threshold
        df.loc[treatment & below_risk, "reconsidered_approval"] = 1

        return df

    def get_contingency_table(self, risk_threshold, records=None):
        """Generates a 2x2 contingency table comparing Control vs Treatment reconsidered approvals."""
        df = self._prepare_experiment_df(risk_threshold=risk_threshold, records=records)
        contingency_table = pd.crosstab(df["group"], df["reconsidered_approval"])
        contingency_table = contingency_table.reindex(columns=[0, 1], fill_value=0)
        return contingency_table

    
    def chi_square(self, risk_threshold,records=None):
        """Performs a Chi-Square test for nominal association on the contingency table."""
        data = self.get_contingency_table(risk_threshold=risk_threshold,records=records)
        Contingency_table = Table2x2(data.values)
        chisquare_test = Contingency_table.test_nominal_association()
        
        return chisquare_test

    def get_ROI(self, risk_threshold, records=None):
        """Calculates and displays financial projections for recovered loans."""
        df = self._prepare_experiment_df(risk_threshold=risk_threshold, records=records)

        # Recovered loans (applications approved under adjusted threshold)
        recovered_loans = df[df["reconsidered_approval"] == 1].copy()

        if recovered_loans.empty:
            return html.Div("No loans were recovered at this risk threshold.",style={"color": "orange", "padding": "10px"})

        average_loan_amount = (recovered_loans["loan_amnt"].mean() if "loan_amnt" in recovered_loans.columns  else 5000 )
        average_interest_rate = ((recovered_loans["loan_int_rate"].mean() / 100) if "loan_int_rate" in recovered_loans.columns  else 0.12 )

        total_volume = len(recovered_loans) * average_loan_amount
        expected_gross_interest = total_volume * average_interest_rate

        if "loan_amnt" in recovered_loans.columns:
            expected_loss = (
                recovered_loans["loan_amnt"] * recovered_loans["p_reject"]).sum()
        else:
            expected_loss = (average_loan_amount * recovered_loans["p_reject"]).sum()

        net_expected_profit = expected_gross_interest - expected_loss

        # Calculating the KPI
        return html.Div(
            [
                html.H4("Financial Projections (ROI)"),
                html.P(f"Recovered Loans: {len(recovered_loans):,}"),
                html.P(f"Total Portfolio Volume: ${total_volume:,.2f}"),
                html.P(f"Expected Gross Interest: ${expected_gross_interest:,.2f}"),
                html.P(f"Expected Default Losses: ${expected_loss:,.2f}"),
                html.Hr(),
                html.H3(f"Net Expected Profit/Loss: ${net_expected_profit:,.2f}",
                    style={
                        "color": ("green" if net_expected_profit >= 0 else "red")
                    },
                ),
            ],
            style={
                "border": "1px solid #ddd",
                "padding": "15px",
                "borderRadius": "8px",
                "backgroundColor": "#f9f9f9",
            },
        )