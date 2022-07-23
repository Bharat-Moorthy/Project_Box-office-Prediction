import streamlit as st
import pickle
import requests
import base64
import numpy as np

st.set_page_config(
     page_title="Box office Predictor",
     page_icon="🍿",
     layout="centered",
          menu_items={
         'About': 'https://www.kaggle.com/competitions/tmdb-box-office-prediction/data',
         
     }
 )
xgb_model=pickle.load(open('HypertuneXGB.pkl','rb'))
@st.cache()


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp{
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('/color-money-background-stationery-739106.png')

def prediction(budget,crew_male_count,cast_male_count,cast_count,crew_count,crew_female_count,people_count_dept_editing,people_count_dept_production,
        people_count_job_producer,people_count_job_editor,people_count_job_casting,cast_female_count,people_count_dept_VisualEffects,people_count_job_ArtDirection,
        count_of_tagline_words,Keywords_count,production_companies_count,people_count_job_Screenplay,part_of_collection,people_count_job_ProductionDesign,
        people_count_job_ExecutiveProducer,Production_country_USA):
    log_budget=np.log(budget)
    if part_of_collection == "Yes":
        part_of_collection=1
    else:
        part_of_collection=0

    if Production_country_USA =="Yes":
        Production_country_USA=1
    else:
        Production_country_USA=0

    features=np.array([log_budget,crew_male_count,cast_male_count,cast_count,crew_count,crew_female_count,people_count_dept_editing,people_count_dept_production,
        people_count_job_producer,people_count_job_editor,people_count_job_casting,cast_female_count,people_count_dept_VisualEffects,people_count_job_ArtDirection,
        count_of_tagline_words,Keywords_count,production_companies_count,people_count_job_Screenplay,part_of_collection,people_count_job_ProductionDesign,
        people_count_job_ExecutiveProducer,Production_country_USA])

    log_revenue=xgb_model.predict(features.reshape(1,-1))
    revenue=np.exp(log_revenue)
    return revenue

def main():
    st.markdown("<h1 style ='text-align:center;color:Brown;'>MOVIE BOX OFFICE PREDICTION</h1>",unsafe_allow_html=True)
    movie_name=st.text_input("Enter the movie name","")
    st.write('The movie title is ',movie_name)
    budget=st.number_input("Enter the movie budget in dollars",min_value=100,max_value=100000000,step=1)

    crew_male_count=st.number_input("Enter the male crew count",min_value=0,max_value=1000,step=1,help="Enter the count of male gender in total crew")
    cast_male_count=st.number_input("Enter the count of male actors",min_value=0,max_value=1000,step=1,help="Enter the count of male gender actors")
    cast_count=st.number_input("Enter the total cast count",min_value=1,max_value=1000,step=1)
    crew_count=st.number_input("Enter the total crew count",min_value=1,max_value=1000,step=1)
    crew_female_count=st.number_input("Enter the female crew count",min_value=0,max_value=1000,step=1,help="Enter the count of female gender in total crew")
    people_count_dept_editing=st.number_input("Enter the editing department people count",min_value=1,max_value=100,step=1,help="Enter the total count of people in editing department")
    people_count_dept_production=st.number_input("Enter the production department people count",min_value=1,max_value=100,step=1,help="Enter the total count of people in production department")
    people_count_job_producer=st.number_input("Enter the producers count",min_value=1,max_value=50,step=1,help="Enter the total count of people in producer role only")
    people_count_job_editor=st.number_input("Enter the editors count",min_value=1,max_value=50,step=1,help="Enter the total count of people in editor role only")
    people_count_job_casting=st.number_input("Enter the casting people count",min_value=1,max_value=50,step=1,help="Enter the total count of people in casting role only")
    cast_female_count=st.number_input("Enter the count of female actors",min_value=0,max_value=1000,step=1,help="Enter the count of female gender actors")
    people_count_dept_VisualEffects=st.number_input("Enter the Visual Effects department people count",min_value=1,max_value=100,step=1,help="Enter the total count of people in visual effects department")
    people_count_job_ArtDirection=st.number_input("Enter the Art Direction people count",min_value=1,max_value=50,step=1,help="Enter the total count of people in art direction role only")
    count_of_tagline_words = st.number_input("Enter the count of tagline words if any ",min_value=0,max_value=50,step=1,help="Enter the count of words of tagline. If no tagline enter zero")
    Keywords_count  = st.number_input("Enter the count of keywords if any ",min_value=0,max_value=50,step=1,help="Enter the count of words of keywords. If no keywords enter zero")
    production_companies_count = st.number_input("Enter the count production companies ",min_value=1,max_value=50,step=1,help="Enter the count of production companies involved in production")
    people_count_job_Screenplay=st.number_input("Enter the Screenplay people count",min_value=0,max_value=50,step=1,help="Enter the total count of people in screeplay writer role only")
    part_of_collection = st.selectbox(
        'Whether the movie is part of collection of movies released before',
        ('Yes', 'No'),help="If the movie is a part of previously released movies ,select yes else no")
    people_count_job_ProductionDesign=st.number_input("Enter the Production Design people count",min_value=1,max_value=50,step=1,help="Enter the total count of people in production design role only")
    people_count_job_ExecutiveProducer=st.number_input("Enter the Executive Producer people count",min_value=1,max_value=50,step=1,help="Enter the total count of people in Executive producer role only")
    Production_country_USA = st.selectbox(
        'Whether the production country is USA or not',
        ('Yes', 'No'),help="If the production company belong to US ,select Yes else no")
    if st.button("Predict Box Office"):
        result = prediction(budget,crew_male_count,cast_male_count,cast_count,crew_count,crew_female_count,people_count_dept_editing,people_count_dept_production,
                people_count_job_producer,people_count_job_editor,people_count_job_casting,cast_female_count,people_count_dept_VisualEffects,people_count_job_ArtDirection,
                count_of_tagline_words,Keywords_count,production_companies_count,people_count_job_Screenplay,part_of_collection,people_count_job_ProductionDesign,
                people_count_job_ExecutiveProducer,Production_country_USA)
        st.success("Predicted Revenue of the Movie: {} US Dollars".format(result.item()))


if __name__ == "__main__":
    main()
    


     
