import os
import re
import spacy
import pandas as pd
import http.client
import json
from datetime import datetime, timedelta
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import dateparser
from fuzzywuzzy import process

# Load environment variables
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load and filter cities data
df1 = pd.read_csv(r'G:\HamsafarAI\data\worldcities.csv')
df1 = df1[df1['country'] == 'India']
cities = df1['city_ascii'].dropna().unique().tolist()

# Initialize Google Generative AI model
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Precompute city vectors for similarity search
def precompute_city_vectors(original_cities):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([city.lower() for city in original_cities])
    return vectorizer, vectors

vectorizer, city_vectors = precompute_city_vectors(cities)

# Function to extract date
def extract_date(query):
    doc = nlp(query)
    today = datetime.today()
    days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    for ent in doc.ents:
        if ent.label_ == "DATE":
            date_text = ent.text.lower()
            if "tomorrow" in date_text:
                return (today + timedelta(days=1)).strftime('%Y-%m-%d')
            elif "day after tomorrow" in date_text:
                return (today + timedelta(days=2)).strftime('%Y-%m-%d')
            elif "next" in date_text:
                for day in days_of_week:
                    if day in date_text:
                        day_num = days_of_week.index(day)
                        days_ahead = (day_num - today.weekday() + 7) % 7
                        days_ahead = days_ahead if days_ahead > 0 else 7
                        return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
            match = re.search(r"(\d{1,2})(st|nd|rd|th)?", date_text)
            if match:
                day = int(match.group(1))
                month_match = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)", date_text)
                if month_match:
                    month_text = month_match.group(1)
                    parsed_date = dateparser.parse(f"{day} {month_text} {today.year}")
                    if parsed_date and parsed_date >= today:
                        return parsed_date.strftime('%Y-%m-%d')
                if "next month" in date_text:
                    next_month = today.month % 12 + 1
                    year = today.year if next_month > today.month else today.year + 1
                    parsed_date = datetime(year, next_month, day)
                    if parsed_date:
                        return parsed_date.strftime('%Y-%m-%d')
                parsed_date = dateparser.parse(f"{day} {today.strftime('%B')} {today.year}")
                if parsed_date and parsed_date >= today:
                    return parsed_date.strftime('%Y-%m-%d')
            parsed_date = dateparser.parse(date_text, settings={'PREFER_DATES_FROM': 'future'})
            if parsed_date:
                return parsed_date.strftime('%Y-%m-%d')
    return None

# Find closest city with both cosine similarity and fuzzy matching
def find_closest_city(entity, original_cities, vectorizer, city_vectors, threshold=0.8):
    best_match, fuzzy_score = process.extractOne(entity, original_cities)
    if fuzzy_score >= threshold * 100:
        return best_match
    entity_vector = vectorizer.transform([entity.lower()])
    cosine_sim = cosine_similarity(entity_vector, city_vectors)
    closest_city_index = cosine_sim.argmax()
    return original_cities[closest_city_index]

def extract_cities(query, cities):
    query = query.lower()
    origin, destination = None, None

    # Attempt to find "from" and "to" cities explicitly
    from_match = re.search(r"from\s+(\w+)", query)
    to_match = re.search(r"to\s+(\w+)", query)

    if from_match:
        origin_candidate = from_match.group(1).strip()
        origin = find_closest_city(origin_candidate, cities, vectorizer, city_vectors)
    
    if to_match:
        destination_candidate = to_match.group(1).strip()
        destination = find_closest_city(destination_candidate, cities, vectorizer, city_vectors)

    if not origin or not destination or origin == destination:
        doc = nlp(query)
        extracted_entities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        matched_cities = [find_closest_city(entity, cities, vectorizer, city_vectors) for entity in extracted_entities]
        
        if len(matched_cities) >= 2:
            origin, destination = matched_cities[0], matched_cities[1]
        elif len(matched_cities) == 1:
            origin = origin or matched_cities[0]

    if origin == destination:
        destination = None  # Reset destination to prompt the user to specify it

    return origin, destination

def convert_time_to_datetime(time_str, day_offset, journey_date):
    base_date = datetime.strptime(journey_date, "%Y-%m-%d")
    return base_date + timedelta(days=day_offset, hours=int(time_str.split(":")[0]), minutes=int(time_str.split(":")[1]))

def get_trains_between_stations(from_station, to_station, date_of_journey):
    from_station = from_station.strip() if from_station else ""
    to_station = to_station.strip() if to_station else ""
    conn = http.client.HTTPSConnection("irctc1.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "b23f4ed7b9msh80d0c1957ac19f2p19c71djsn936ce06cd38a",
        'x-rapidapi-host': "irctc1.p.rapidapi.com"
    }
    url = f"/api/v3/trainBetweenStations?fromStationCode={from_station}&toStationCode={to_station}&dateOfJourney={date_of_journey}"
    conn.request("GET", url, headers=headers)
    res = conn.getresponse()
    data = res.read()
    conn.close()
    return json.loads(data.decode("utf-8"))

# Streamlit Interface
st.title("Travel Planner")

# Single user query input
user_query = st.text_input("Enter your travel query", "I have some time so I wanna go on a trip from Chennai to Prayagraj next month 22nd")

if st.button("Process Query"):
    origin, destination = extract_cities(user_query, cities)
    journey_date = extract_date(user_query)
    
    # Display extracted details
    st.write(f"**Origin**: {origin if origin else 'Not specified'}")
    st.write(f"**Destination**: {destination if destination else 'Not specified'}")
    st.write(f"**Journey Date**: {journey_date if journey_date else 'Not specified'}")

    if origin and destination and journey_date:
        response1 = model.generate_content(f"find train from {origin} to {destination} I just wanna know how to reach there direct more preferrably direct if possible or using connecting trains ok , just wanna know station names that I will have to change(if required) , ok just telll me that , i dont neeed a train , just station names, give me name of stations seperated by comma, dont give space after comma")
        st.write("Stations on the route:", response1.text)

        response2 = model.generate_content(f"find me station codes for {response1.text} stations, just give codes seperated by comma, just give codes ")
        st.write("Station Codes:", response2.text)
        stations = list(response2.text.split(', '))
        
        # Display journey path with rectangles and arrows
        st.markdown("""
        <style>
            .station-box {
                display: inline-block;
                padding: 10px 15px;
                margin: 5px;
                border: 2px solid #4CAF50;
                border-radius: 10px;
                font-weight: bold;
                font-size: 16px;
                color: white;
                background-color: #4CAF50;
            }
            .arrow {
                display: inline-block;
                margin: 5px;
                font-size: 24px;
                color: #4CAF50;
            }
        </style>
        """, unsafe_allow_html=True)

        for i, station in enumerate(stations):
            st.markdown(f'<div class="station-box">{station}</div>', unsafe_allow_html=True)
            if i < len(stations) - 1:
                st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)

        # Fetch and display train details with layover logic
        st.write("Fetching train details...")
        min_layover = timedelta(hours=1)
        max_layover = timedelta(hours=6)
        current_date = journey_date

        for i in range(len(stations) - 1):
            from_station, to_station = stations[i], stations[i + 1]
            train_data = get_trains_between_stations(from_station, to_station, current_date)
            found_connection = False
            
            if 'data' in train_data and train_data['data']:
                for train in train_data['data']:
                    departure_time = convert_time_to_datetime(train['from_std'], train['from_day'], current_date)
                    arrival_time = convert_time_to_datetime(train['to_std'], train['to_day'], current_date)

                    # Display train details
                    st.write(f"Train from {train['from_station_name']} to {train['to_station_name']}")
                    st.write(f"  Train Number: {train['train_number']}")
                    st.write(f"  Train Name: {train['train_name']}")
                    st.write(f"  Departure: {train['from_std']}, Arrival: {train['to_std']}")

                    # Layover check for next leg
                    if i < len(stations) - 2:
                        next_station_code = stations[i + 2]
                        next_leg_date = arrival_time.strftime('%Y-%m-%d')
                        next_trains = get_trains_between_stations(to_station, next_station_code, next_leg_date)

                        valid_next_train = None
                        for next_train in next_trains['data']:
                            next_departure_time = convert_time_to_datetime(next_train['from_std'], next_train['from_day'], next_leg_date)
                            layover_time = next_departure_time - arrival_time
                            if min_layover <= layover_time <= max_layover:
                                valid_next_train = next_train
                                break

                        if valid_next_train:
                            st.write("  Valid connection found:")
                            st.write(f"    Connecting Train Number: {valid_next_train['train_number']}")
                            st.write(f"    Connecting Train Name: {valid_next_train['train_name']}")
                            st.write(f"    Departure Time: {valid_next_train['from_std']} after layover of {layover_time}")
                            current_date = next_departure_time.strftime('%Y-%m-%d')
                            found_connection = True
                            break
                        else:
                            st.write("  No valid connecting train found.")
                            found_connection = False
                    else:
                        found_connection = True
                        break

            if not found_connection:
                st.write(f"No trains found from {from_station} to {to_station} on {current_date}.")
                break
