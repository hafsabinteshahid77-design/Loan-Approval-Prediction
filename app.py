import streamlit as st
import pickle
import pandas as pd

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
# Load Model
model = pickle.load(open("loan_model.pkl", "rb"))

st.title("🏦 Loan Approval Prediction")

no_of_dependents = st.number_input("Number of Dependents", 0, 10)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term")
cibil_score = st.number_input("CIBIL Score")
residential_assets_value = st.number_input("Residential Assets Value")
commercial_assets_value = st.number_input("Commercial Assets Value")
luxury_assets_value = st.number_input("Luxury Assets Value")
bank_asset_value = st.number_input("Bank Asset Value")

if st.button("Predict"):

    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0

    input_data = pd.DataFrame({
        "no_of_dependents":[no_of_dependents],
        "education":[education],
        "self_employed":[self_employed],
        "income_annum":[income_annum],
        "loan_amount":[loan_amount],
        "loan_term":[loan_term],
        "cibil_score":[cibil_score],
        "residential_assets_value":[residential_assets_value],
        "commercial_assets_value":[commercial_assets_value],
        "luxury_assets_value":[luxury_assets_value],
        "bank_asset_value":[bank_asset_value]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success(" Loan Approved")
    else:
        st.error(" Loan Rejected")


st.markdown("---")
st.header("🤖 AI Loan Assistant")

# Groq API Key from Streamlit Secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

# Load LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key,
    temperature=0.5
)

# User Question
question = st.text_input("Ask any question about Bank Loans")

if st.button("Ask Chatbot"):

    if question:

        prompt = f"""
You are an AI Bank Loan Assistant.

Answer only loan-related questions in simple English.

User Question:
{question}
"""

        response = llm.invoke([HumanMessage(content=prompt)])

        st.success(response.content)
