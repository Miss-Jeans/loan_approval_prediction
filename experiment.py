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
        self._cached_base_df=None    #Creating a placeholder cache for base data

    def load_and_cache_base_data(self):
        """
        Handles the heavy MongoDB fetching and Model predictions.
        """
        records = self.repo.assign_to_groups(observations=None)
        df = pd.DataFrame(records)
        if df.empty:
            raise ValueError("No records returned from repository.")

        feature_cols = self.Mdl.X_train.columns if hasattr(self.Mdl, "X_train") else None
        if feature_cols is not None:
            X = df[feature_cols]
        else:
            X_train, _, _, _ = self.Mdl.get_train_test_data()
            X = df[X_train.columns]

        # Run GradientBoosting prediction once
        df["p_reject"] = self.model.predict_proba(X)[:, 0]
        self._cached_base_df = df
        return df

    def _prepare_experiment_df(self, risk_threshold, records=None):
        """Faster threshold application."""
        
        if records is not None:
            df = pd.DataFrame(records)
            
        else:
            if self._cached_base_df is None:
                self.load_and_cache_base_data()
            df = self._cached_base_df.copy() # Shallow copy for fast modification

        # Fast assignment using NumPy
        treatment = df["group"] == "Treatment(Adjusted_risk_Threshold)"
        below_risk = df["p_reject"] <= risk_threshold
        
        df["reconsidered_approval"] = np.where(treatment & below_risk, 1, 0)
        return df

    def get_contingency_table(self, risk_threshold, records=None):
        df = self._prepare_experiment_df(risk_threshold=risk_threshold, records=records)
        contingency_table = pd.crosstab(df["group"], df["reconsidered_approval"])
        return contingency_table.reindex(columns=[0, 1], fill_value=0)

    def chi_square(self, risk_threshold, records=None):
        data = self.get_contingency_table(risk_threshold=risk_threshold, records=records)
        return Table2x2(data.values).test_nominal_association()

    def get_ROI(self, risk_threshold, records=None):
        df = self._prepare_experiment_df(risk_threshold=risk_threshold, records=records)
        recovered_loans = df[df["reconsidered_approval"] == 1]

        if recovered_loans.empty:
            return html.Div("No loans were recovered at this risk threshold.", style={"color": "orange", "padding": "10px"})

        # Vectorized Math computations
        avg_loan = recovered_loans["loan_amnt"].mean() if "loan_amnt" in recovered_loans.columns else 5000
        avg_int = (recovered_loans["loan_int_rate"].mean() / 100) if "loan_int_rate" in recovered_loans.columns else 0.12

        total_volume = len(recovered_loans) * avg_loan
        expected_gross_interest = total_volume * avg_int
        
        expected_loss = (recovered_loans["loan_amnt"] * recovered_loans["p_reject"]).sum() if "loan_amnt" in recovered_loans.columns else (avg_loan * recovered_loans["p_reject"]).sum()
        net_expected_profit = expected_gross_interest - expected_loss

        return html.Div([
            html.H4("Financial Projections (ROI)"),
            html.P(f"Recovered Loans: {len(recovered_loans):,}"),
            html.P(f"Total Portfolio Volume: ${total_volume:,.2f}"),
            html.P(f"Expected Gross Interest: ${expected_gross_interest:,.2f}"),
            html.P(f"Expected Default Losses: ${expected_loss:,.2f}"),
            html.Hr(),
            html.H3(f"Net Expected Profit/Loss: ${net_expected_profit:,.2f}", style={"color": "green" if net_expected_profit >= 0 else "red"}),
        ], style={"border": "1px solid #ddd", "padding": "15px", "borderRadius": "8px", "backgroundColor": "#f9f9f9"})