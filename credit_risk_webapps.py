from joblib import load
import numpy as np
import pandas as pd
import streamlit as st
import datetime

st.write("# Loan Credit Risk Prediction")
st.markdown("**This is a web app to predict the loan credit risk based on the borrower's set of attributes.** ")
st.markdown('The model is built using Random Forest ML Algorithm with the performance of 97% Accuracy, 98% Recall, 97% Precision.')
st.markdown('Author:   [@Ichfan Kurniawan](https://www.linkedin.com/in/ichfan-kurniawan/)')
st.markdown('[Github Repo](https://github.com/IchfanKurniawan/loan-credit-risk-streamlit)')
st.markdown('---')
st.write("## Input Data")
st.write("Hover over to the question mark sign to get a description of the borrower's attributes below.")


# help markdown
help_term = 'The number of payments on the loan.'.strip()
help_home = 'The home ownership status provided by the borrower.'.strip()
help_verif = 'Verification status of a borrower.'.strip()
help_initial_list = 'The initial listing status of the loan.'.strip()
help_int_rate = 'Interest rate on the loan'.strip()
help_annual_inc = 'The self-reported annual income provided by the borrower during registration.'.strip()
help_dti = 'A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.'.strip()
help_inq_last_6mths = 'The number of inquiries in past 6 months (excluding auto and mortgage inquiries).'.strip()
help_revol_util = 'Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.'.strip()
help_total_rec_prncp = 'Principal received to date.'.strip()
help_out_prncp = 'Remaining outstanding principal for total amount funded.'.strip()
help_total_rec_late_fee = 'Late fees received to date.'.strip()
help_recoveries = 'Recoveries'.strip()
help_collection_recovery_fee = 'Post charge off collection fee'.strip()
help_last_pymnt_amnt = 'Last total payment amount received.'.strip()
help_emp_length = 'Employment length in years.'.strip()
help_issue_d = 'The month which the loan was funded.'.strip()
help_last_credit_pull_d = 'The most recent month LC pulled credit for this loan.'.strip()



# categorical vars
term = st.selectbox('Term (term)', (' 36 months', ' 60 months'), help=help_term)
home_ownership = st.selectbox('Home Ownership (home_ownership)', ('RENT', 'OWN', 'MORTGAGE', 'OTHER'), help=help_home)
verification_status = st.selectbox('verification Status (verification_status)', ('Verified', 'Source Verified', 'Not Verified'), help=help_verif)
initial_list_status = st.selectbox('Initial List Status (initial_list_status)', ('f', 'w'), help=help_initial_list)

# numerical vars
int_rate = st.number_input('Interest Rate (int_rate)', 5.0, 30.0, 10.36, help=help_int_rate)
annual_inc = st.number_input('Annual Income (annual_inc)', 1000.0, 8000000.0, 24000.0, help=help_annual_inc)
dti = st.number_input('dti Ratio (dti)', 0.0, 50.0, 27.65, help=help_dti)
inq_last_6mths = st.number_input('Inquiries in Past 6 Months (inq_last_6mths)', 0.0, 8.0, 1.0, help=help_inq_last_6mths)
revol_util = st.number_input('Revolving Line Utilization Rate (revol_util)', 0.0, 900.0, 83.7, help=help_revol_util)
total_rec_prncp = st.number_input('Principal Received (total_rec_prncp)', 0.0, 36000.0, 5000.0, help=help_total_rec_prncp)
out_prncp = st.number_input('Outstanding Principal (out_prncp)', 0.0, 35000.0, 0.0, help=help_out_prncp)
total_rec_late_fee = st.number_input('Late Fees Received (total_rec_late_fee)', 0.0, 360.0, 0.0, help=help_total_rec_late_fee)
recoveries = st.number_input('Recoveries', 0.0, 34000.0, 0.0, help=help_recoveries)
collection_recovery_fee = st.number_input('Post Charge Off Collection Fee (collection_recovery_fee)', 0.0, 7100.0, 0.0, help=help_collection_recovery_fee)
last_pymnt_amnt = st.number_input('Last Payment Amount (last_pymnt_amnt)', 0.0, 37000.0, 1000.0, help=help_last_pymnt_amnt)

# date related datatype
emp_length = st.selectbox('Employement length (emp_length)', ('< 1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10 +'), help=help_emp_length)
issue_d = st.date_input("Issue Date (issue_d)", datetime.date(2010, 1, 1), datetime.date(2007, 1, 1), datetime.date(2015, 1, 1), help=help_issue_d)
last_credit_pull_d = st.date_input("Last Credit Pull  Date (last_credit_pull_d)", datetime.date(2010, 1, 1), datetime.date(2007, 1, 1), datetime.date(2015, 1, 1), help=help_last_credit_pull_d)




# User Input - Feature Engineering
# Date to numeric
quarter_dict = {
    'Jan':0,
    'Feb':0,
    'Mar':0,
    'Apr':1/4,
    'May':1/4,
    'Jun':1/4,
    'Jul':2/4,
    'Aug':2/4,
    'Sep':2/4,
    'Oct':3/4,
    'Nov':3/4,
    'Dec':3/4
}

def month_to_quarter(x):
    try: return float(x[-2:]) + (quarter_dict[x[:3]])
    except Exception as e: return np.nan
    
issue_d = month_to_quarter(str(issue_d.strftime('%b-%y')))
last_credit_pull_d = month_to_quarter(str(last_credit_pull_d.strftime('%b-%y')))

# Employement length
def emp_length_encode(x):
    try:
        if '+' in x:
            return 10
        elif '<' in x:
            return 0
        else:
            return int(x[0])
    except Exception as e: return np.nan
    
emp_length = emp_length_encode(emp_length)





# dataframe of user input data
input_dict ={
    'term':term,
    'int_rate':int_rate,
    'emp_length':emp_length,
    'home_ownership':home_ownership,
    'annual_inc':annual_inc,
    'verification_status':verification_status,
    'issue_d':issue_d,
    'dti':dti,
    'inq_last_6mths':inq_last_6mths,
    'revol_util':revol_util,
    'initial_list_status':initial_list_status,
    'out_prncp':out_prncp,
    'total_rec_prncp':total_rec_prncp,
    'total_rec_late_fee':total_rec_late_fee,
    'recoveries':recoveries,
    'collection_recovery_fee':collection_recovery_fee,
    'last_pymnt_amnt':last_pymnt_amnt,
    'last_credit_pull_d':last_credit_pull_d
}

input_dict = pd.DataFrame(input_dict, index=[0])
df = pd.read_csv('webapp_data_creation.csv')
df = pd.concat([input_dict, df], axis=0)

# split categorical & numerical
df_cat = df.select_dtypes(include='object')
df_num = df.select_dtypes(exclude='object')

# get_dummies of categorical
df_cat = pd.get_dummies(df_cat)
drop_col_dummies = ['term_ 36 months', 'home_ownership_MORTGAGE', 
                    'verification_status_Not Verified', 'initial_list_status_f']
df_cat.drop(drop_col_dummies, axis=1, inplace=True)

# concat categorical & numerical after encoding
df_dummies = pd.concat([df_cat, df_num], axis=1)
data_test = df_dummies[:1]



# Load & Process Model
model = load('final_model_credit.joblib')

if st.button('Get the Prediction!'):
    # Processing
    st.markdown('---')
    st.write('## Result')
    y_hat = model.predict(data_test)

    type_y_hat = type(y_hat)
    if y_hat[0] == 1:
        st.write("""#### We are so sorry, Your Loan Credit is Rejected!""")
    else:
        st.write("""#### Congratulations, Your Loan Credit is Approved!""")

    pred_proba = model.predict_proba(data_test)
  
    st.write('The Loan is Approved Probability = ', str(np.round(pred_proba[0][0]*100,2)),'%')
    st.write('The Loan is Rejected Probability = ', str(np.round(pred_proba[0][1]*100,2)),'%')

