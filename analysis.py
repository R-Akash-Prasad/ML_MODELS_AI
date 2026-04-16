import os 
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)

model = genai.GenerativeModel('gemini-2.5-flash-lite')

def generate_summary(result_df):
    prompt = f'''you are a data scientist expert 
            here are the model results : 
    {result_df.to_string()}
            
    1.Identiy the best model
    2.Explain why  it is best 
    3. summarize the performance of the models 
    '''
    
    response = model.generate_content(prompt)
            
    return response.text

def suggest_imporovements(result_df):
    prompt = f''' you are a data scientist expert.
    here are the model result :
    
    {result_df.to_string()}
    
    
    suggest:
    - way to improve the model performance 
    - hyperparameter tunning and give range of values in each parameter
    - better sutiable algorithms for the given data 
    - Data perprocessing improvement 
    '''
    
    response = model.generate_content(prompt)
    return response.text