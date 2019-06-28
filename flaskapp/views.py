from flask import render_template
from flask import request
from flaskapp import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flaskapp.model_tfidf import get_competitors

# Python code to connect to Postgres
# You may need to modify this based on your OS,
# as detailed in the postgres dev setup materials.
user = 'zuzuluo' #add your Postgres username here
host = 'localhost'
dbname = 'patent_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)

@app.route('/')
@app.route('/input')
def patent_input():
   return render_template("input.html")

@app.route('/output')
def patent_output():
    #pull 'patent_month' from input field and store it
    query = request.args.get('description')
    #just select the Cesareans  from the patent dtabase for the month that the user inputs
    df_cmpt = get_competitors(query)

    # convert dataframe to dictionary
    cmpt_dic = []
    for i in range(0, df_cmpt.shape[0]):
        index = i+1
        company_name = df_cmpt.assignee_name.values[i]
        country = df_cmpt.applicant_country.values[i]
        priority_yr = df_cmpt.patent_date.values[i]
        patent_title = df_cmpt.title.values[i]
        cos_sim = df_cmpt.cos_sim.values[i]
        cmpt_dic.append(dict(index=index, assignee_name=company_name,
                            applicant_country=country, patent_date=priority_yr,
                            title=patent_title, cos_sim=cos_sim))

    return render_template('output.html', tables=cmpt_dic, titles=df_cmpt.columns.values)
