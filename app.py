import streamlit as st

from PIL import Image
from datetime import datetime  # Import datetime
import pickle

model = pickle.load(open('./pipeline_important.pkl', 'rb'))

def run():
    st.set_page_config(layout="wide")

    
   
    st.markdown("""
    <head>
        <meta charset="UTF-8">
        <title>All Navigation Menu Hover Animation | CodingLab</title> 
        <style>
            body {
                background-color: #000000;  /* Black background color */
                
            }
            .stApp {
                background-color: #000000;  /* Set background to black for the app */
                
            }
            h1, h2, h3, h4, h5, h6, p, li, a {
                color: #ffffff;  /* Set font color to white for all text elements */
            }
            .nav-links {
                list-style: none;
                display: flex;
                justify-content: space-around;
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
                font-size: 20px;
            }
            .nav-links li {
                position: relative;
                padding: 10px 20px;
                cursor: pointer;
                transition: transform 0.3s ease;
            }
            .nav-links li a {
                color: #fff;  /* Change link text color to white for contrast */
                text-decoration: none;
            }
            .center::after, .upward::after, .forward::after {
                content: '';
                position: absolute;
                width: 100%;
                height: 2px;
                bottom: -5px;
                left: 0;
                background-color: #fff;  /* Change underline color to white */
                transform: scaleX(0);
                transition: transform 0.3s ease;
            }
            .center::after {
                transform-origin: center;
            }
            .upward::after {
                transform-origin: bottom;
            }
            .forward::after {
                transform-origin: top right;
            }
            .nav-links li:hover::after {
                transform: scaleX(1);
            }
            .button {
                color: black; /* Button text color */
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                
            }
                
        </style>
    </head>
    """, unsafe_allow_html=True)


    st.markdown("""
    <body>
        <ul class="nav-links">
            <li class="center"><a href="https://zafrafarhan461.wixsite.com/accidentseverity">HOME</a></li>
            <li class="center"><a href="https://zafrafarhan461.wixsite.com/accidentseverity/traffic-apis">Accident APIs</a></li>
            <li class="upward"><a href="https://zafrafarhan461.wixsite.com/accidentseverity/copy-of-traffic-apis">Accident Stats</a></li>
            <li class="forward"><a href="https://zafrafarhan461.wixsite.com/accidentseverity/copy-2-of-traffic-apis">Route Monitoring</a></li>
            <li class="forward"><a href="https://zafrafarhan461.wixsite.com/accidentseverity/copy-3-of-traffic-apis">O/D Monitoring</a></li>
            <li class="forward"><a href="#">Severity Prediction</a></li>
        </ul>
    </body>
    """, unsafe_allow_html=True)
    
    image_path = "./bg.png"  # Assuming bg.jpg is in the same directory as your script
    img1 = Image.open(image_path)
    #img1 = img1.resize((80, 50), Image.BICUBIC)  # Use BICUBIC filter for better quality
    st.image(img1, use_column_width=True)
    
    st.header("Accident Details")

    # Speed input using a slider for better interactivity
    Speed_limit = st.slider(
        "Average Speed (km/h):",
        min_value=0,
        max_value=200,
        step=5,
        value=60
    )
    
    # Vehicle type selection using radio buttons
    vehicle = st.radio(
        "Select the Vehicle Type:",
        ('Car', 'Van', 'Bus', 'Motorcycle', 'Pedal cycle', 'Other')
    )

    # Number of casualties in the accident
    Number_of_Casualties = st.number_input(
        "Number of Casualties:",
        min_value=0,
        max_value=100,
        step=1,
        value=0
    )
    
    
    # Number of vehicles involved in the accident
    Number_of_Vehicles = st.slider(
        "Number of Vehicles Involved:",
        min_value=1,
        max_value=5,
        step=1,
        value=1,
        format="%d"  # Ensures that the displayed value is an integer
    )
    
    ## Road type
    road_display = ('Single carriageway','Dual carriageway','Roundabout','One way street','Slip road')
    road_options = list(range(len(road_display)))
    Road_Type = st.selectbox("Road Type ",road_options, format_func=lambda x: road_display[x])

    ## For road surface condition
    road_surface_display = ('Dry','Wet or damp','Snow')
    surface_options = list(range(len(road_surface_display)))
    Road_Surface_Conditions= st.selectbox("Road Surface Condition :", surface_options, format_func=lambda x: road_surface_display[x])

    ## Junction Detail
    junction_display = ('T or staggered junction', 'Crossroads',
       'Not at junction or within 20 metres', 'Roundabout',
       'Mini-roundabout', 'More than 4 arms (not roundabout)',
       'Private drive or entrance', 'Slip road', 'Other junction')
    junction_options = list(range(len(junction_display)))
    Junction_Detail = st.selectbox("Road Type ",junction_options, format_func=lambda x: junction_display[x])

    ## For urban rural status
    urban_display = ('Rural','Urban')
    urban_options = list(range(len(urban_display)))
    Urban_or_Rural_Area = st.selectbox("Urban-Rural Status :",urban_options, format_func=lambda x: urban_display[x])


    if st.button('Predict Severity'):
        features = [[Number_of_Vehicles, Speed_limit, Urban_or_Rural_Area, Number_of_Casualties, Junction_Detail, Road_Surface_Conditions, Road_Type]]
        prediction = model.predict(features)
        
        if prediction == 0:
            st.write("The accident severity is predicted to be low.")
        else:
            st.write("The accident severity is predicted to be medium/ high")
    
run()
