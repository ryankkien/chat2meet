import gradio as gr  # Importing Gradio for creating web UI
from langchain import PromptTemplate, LLMChain  # Importing necessary classes from langchain
from langchain.chat_models import ChatOpenAI  # Importing OpenAI chat model
import pandas as pd  # Pandas for data manipulation
import requests  # For making HTTP requests
from bs4 import BeautifulSoup  # For parsing HTML
import re  # For regular expressions
from datetime import datetime, timedelta  # For handling date and time
from langchain.llms import OpenAI

# Function to scrape data from when2meet link
def scrape_when2meet(link):
    response = requests.get(link)  # Making a GET request to the link
    soup = BeautifulSoup(response.content, 'html.parser')  # Parsing the HTML content
    script = str(soup.findAll('script'))  # Finding all script tags

    # Nested function to extract data using regular expressions
    def extract_data(regex):
        matches = re.findall(regex, script)  # Finding all matches
        return pd.DataFrame(matches)  # Returning a DataFrame of matches

    # Extracting slots, users, and availability data
    slots = extract_data("TimeOfSlot\[(\d+)\]=(\d+);")
    users = extract_data("PeopleNames\[(\d+)\] = '([^']+)';PeopleIDs\[\d+\] = (\d+);")
    avails = extract_data("AvailableAtSlot\[(\d+)]\.push\((\d+)\);")

    # Creating dictionaries for users and slots
    userdict = users.set_index(2)[1].to_dict()
    avails.columns = ['timeslot', 'id']
    avails['available'] = True  # Marking all slots as available
    pivot_avails = avails.pivot_table(index='id', columns='timeslot', fill_value=False)
    reshaped_avails = pivot_avails.stack().reset_index()  # Reshaping the DataFrame

    # Renaming columns and merging DataFrames
    users.columns = ['id', 'name', 'user_id']
    slots.columns = ['timeslot', 'time']
    slots['time'] = pd.to_datetime(slots['time'].astype(int), unit='s').dt.strftime("%a %H:%M")
    slotsdict = slots.set_index('timeslot')['time'].to_dict()
    avails['timeslot'] = avails['timeslot'].map(slotsdict)
    avails['id'] = avails['id'].map(userdict)
    avails = avails[['timeslot', 'id']]
    avails[['day_of_week', 'time']] = avails['timeslot'].str.split(' ', expand=True)
    avails = avails[['id', 'day_of_week', 'time']]
    return avails  # Returning the final DataFrame

# Function to convert minutes to time
def minutes_to_time(minutes):
    time = timedelta(minutes=minutes)  # Creating a timedelta object
    hours = str(time.seconds//3600).zfill(2)  # Extracting hours
    minutes = str((time.seconds//60)%60).zfill(2)  # Extracting minutes
    return hours + ':' + minutes  # Returning time in HH:MM format

# Function to combine times
def combine_times(newdf):
    for index, row in newdf.iterrows():  # Iterating over DataFrame rows
        for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:  # Iterating over days of the week
            if row[day] != '-':  # If the day is not empty
                schedlist = [int(i) for i in row[day].split(",")]  # Splitting the schedule into a list
                denseschedlist = []  # Initializing the list for dense schedule
                if schedlist:  # If the schedule list is not empty
                    i = 0
                    while i < len(schedlist):  # Iterating over the schedule list
                        if i+1 < len(schedlist) and schedlist[i+1] - schedlist[i] == 15:  # If the next slot is 15 minutes away
                            j = i
                            while j+1 < len(schedlist) and schedlist[j+1] - schedlist[j] == 15:  # While the next slot is 15 minutes away
                                j += 1
                            denseschedlist.append(minutes_to_time(schedlist[i]) + '-' + minutes_to_time(schedlist[j] + 15))  # Adding the time range to the dense schedule list
                            i = j + 1
                        else:
                            i += 1
                newdf.at[index, day] = ','.join(denseschedlist)  # Joining the dense schedule list into a string
    return newdf  # Returning the updated DataFrame

# Function to convert schedule to a different format
def format_schedule(df):
    df['time'] = pd.to_datetime(df['time'], format='%H:%M')  # Converting time to datetime format
    df['minutes_from_midnight'] = df['time'].dt.hour * 60 + df['time'].dt.minute  # Calculating minutes from midnight
    df = df.groupby(['id', 'day_of_week', 'minutes_from_midnight']).size().reset_index(name='counts')  # Grouping by id, day of week, and minutes from midnight
    df = df.groupby(['id', 'day_of_week']).apply(lambda x: ','.join([str(minutes_from_midnight) for minutes_from_midnight in x['minutes_from_midnight']])).unstack().fillna('-')  # Joining the minutes from midnight into a string for each id and day of week
    return combine_times(df)  # Combining times and returning the final DataFrame

# Function to convert DataFrame to text chart
def df_to_text_chart(df):
    text_chart = []  # Initializing the text chart list

    # Adding the header and separator rows
    text_chart.append("Name          | Mon    | Tue    | Wed     | Thu    | Fri     | Sat   | Sun")
    text_chart.append("--------------|--------|--------|---------|--------|---------|-------|-----")

    for index, row in df.iterrows():  # Iterating over DataFrame rows
        row_values = [index]  # Initializing the row values list with the index

        for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:  # Iterating over days of the week
            if day in row and row[day] != '-':  # If the day is in the row and is not empty
                row_values.append(row[day])  # Adding the day's schedule to the row values
            else:
                row_values.append('-')  # Adding a '-' for empty schedule

        text_chart.append(' | '.join(row_values))  # Joining the row values with '|' and adding the row to the text chart

    text_chart = '\n'.join(text_chart)  # Joining the rows of the text chart with '\n'

    return text_chart  # Returning the final text chart


# Function to answer the question using OpenAI's GPT-3 or GPT-4
def answer_question(when2meet_link, openai_key, model_choice, question):
    # Define the role
    role = 'You are an intelligent secretary with great reasoning skills that helps users with scheduling. You will be given a calendar of availabilities. When asked a question you will answer a question step-by-step, proving your logic. Answers should be specific. When asked for a time, give the the time and day. If you believe there is no correct solution to a question, simply state so.'

    # Define the prompt template
    chart = df_to_text_chart(format_schedule(scrape_when2meet(when2meet_link)))
    template = f"""
    Role: {role}

    Question: {{question}}

    Chart: {chart}

    Answer: """
    prompt = PromptTemplate(
        template=template,
        input_variables=['question']
    )
    if model_choice == 'gpt-4':  # If the model choice is GPT-4
        llm = ChatOpenAI(openai_api_key = openai_key, temperature = 0, model_name='gpt-4')  # Creating a ChatOpenAI object with GPT-4
    else:
        llm = ChatOpenAI(openai_api_key = openai_key, temperature = 0, model_name='gpt-3.5-turbo')  # Creating a ChatOpenAI object with GPT-3.5 Turbo
    llm_chain = LLMChain(prompt=prompt, llm=llm)  # Creating a LLMChain object
    question += "Explain your reasoning step-by-step."  # Adding a request for explanation to the question
    return llm_chain.run(question)  # Running the LLMChain and returning the result

# Creating the Gradio interface
iface = gr.Interface(fn=answer_question, 
                     inputs=["text", "text", gr.inputs.Radio(['gpt-4', 'gpt-3.5-turbo']), "text"], 
                     outputs="text",
                     allow_flagging=False,
                     allow_screenshot=False,
                     title="chat2meet",
                     description="""This is a chatbot that uses OpenAI's GPT-3 or GPT-4 to schedule people from a when2meet link. Please enter a when2meet link, openai key, model, and question.
                    Your when2meet link should cover a full week (this may change later)
                    If you have any issues email: ryankien@ucla.edu
                    Here are some samples you can request:
                    Give me a couple of times where X and X could meet.
                    Who is available on Monday at 10 AM?
                    Find a time slot where the most participants are available.
                    Create a a full shift schedule for the week""")

# Launching the Gradio interface
iface.launch()
