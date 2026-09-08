import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from data import Data 


class Repository:
    """MongoDB Repository that retrieves and processes data for A/B testing."""

    def __init__(self, mongo_uri, records):
        # Load environment variables 
        load_dotenv()
        
        #Retrieve URI
        self.mongo_uri = os.environ.get("MONGO_URI")
        self.client = MongoClient(self.mongo_uri)

        # Access database and collection
        self.db = self.client["credit_risk_ab_test"]
        self.collection = self.db["loan_applicants"]

        # Instantiate Data class
        self.records = Data()

    def transform_db(self, input_df=None):
        """Wrangled data via Data.wrangle_data(), enriches risk tiers, and loads into MongoDB."""
        if input_df is not None:
            df = input_df.copy()
        else:
            df = self.records.wrangle_data(filepath=None)

        # Map Loan Grades to Risk Tiers
        grade_dict = {
            'A': 'Low Risk', 'B': 'Low Risk',
            'C': 'Marginal', 'D': 'Medium Risk',
            'E': 'Medium Risk', 'F': 'High Risk', 'G': 'High Risk'
        }

        if 'loan_grade' in df.columns:
            df['risk_tier'] = df['loan_grade'].map(grade_dict).fillna('Medium Risk')
        else:
            df['risk_tier'] = 'Medium Risk'

        # Insert documents into MongoDB
        records_dict = df.to_dict(orient="records")
        if records_dict:
            self.collection.insert_many(records_dict)

        return self

    def find_by_tier(self, risk_tier="Medium Risk", loan_status=False):
        """Finds non-approved applicant records in PyMongo collection for a given risk tier."""
        query = {"risk_tier": risk_tier, "loan_status": loan_status}
        return list(self.collection.find(query))

    def assign_to_groups(self, observations=None):
        """Randomly assigns observations into Control and Treatment groups."""
        if observations is None:
            observations = self.find_by_tier()

        if not observations:
            return []

        # Reproducible random assignment
        np.random.seed(42)
        np.random.shuffle(observations)

        idx = len(observations) // 2

        # Assign Control and Treatment
        for doc in observations[:idx]:
            doc["InExperiment"] = True
            doc["group"] = "Control(Fixed_risk_Threshold)"

        for doc in observations[idx:]:
            doc["InExperiment"] = True
            doc["group"] = "Treatment(Adjusted_risk_Threshold)"

        return observations

    def update_applicants(self, assigned_records=None):
        """Executes bulk update operations for experiment group assignments."""
        if assigned_records is None:
            assigned_records = self.assign_to_groups()

        if not assigned_records:
            return {"n_modified": 0, "n": 0}

        # Create list of Bulk UpdateOne operations
        operations = [UpdateOne(filter={"_id": doc["_id"]}, update={"$set": doc} ) for doc in assigned_records ]

        # Execute bulk write
        result = self.collection.bulk_write(operations)

        return {
            "n_modified": result.modified_count,
            "n": result.matched_count
        }