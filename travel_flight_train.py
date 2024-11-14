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
import subprocess
import collections.abc

# Load environment variables
load_dotenv()
api_key = st.secrets['GEMINI_API_KEY']

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load and filter cities data
df1 = pd.read_csv(r'Indian_cities.csv')
df1 = df1[df1['country'] == 'India']
cities = df1['city_ascii'].dropna().unique().tolist()

# Initialize Google Generative AI model
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")
def get_fare_details(train_no, from_station_code, to_station_code, class_type, quota, journey_date):
    conn = http.client.HTTPSConnection("irctc1.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': 
st.secrets['x-rapidapi-key']
,
        'x-rapidapi-host': "irctc1.p.rapidapi.com"
    }
    
    # Prepare the API request URL with the provided parameters
    request_url = f"/api/v1/checkSeatAvailability?classType={class_type}&fromStationCode={from_station_code}&quota={quota}&toStationCode={to_station_code}&trainNo={train_no}&date={journey_date}"
    
    # Make the API request
    conn.request("GET", request_url, headers=headers)
    res = conn.getresponse()
    data = res.read()
    conn.close()
    
    # Parse the JSON response
    seat_data = json.loads(data.decode("utf-8"))
    if not seat_data["status"]:
        print("Error fetching seat availability:", seat_data["message"])
        return None
    
    # Extract and format seat availability details
    availability_details = []
    for availability in seat_data["data"]:
        availability_info = {
            "ticket_fare": availability["ticket_fare"],
            "catering_charge": availability["catering_charge"],
            "total_fare": availability["total_fare"],
            "date": availability["date"],
            "status": availability["current_status"]
        }
        availability_details.append(availability_info)
    
    return availability_details
    
   
    
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
        'x-rapidapi-key': 
st.secrets['x-rapidapi-key']
,
        'x-rapidapi-host': "irctc1.p.rapidapi.com"
    }
    url = f"/api/v3/trainBetweenStations?fromStationCode={from_station}&toStationCode={to_station}&dateOfJourney={date_of_journey}"
    conn.request("GET", url, headers=headers)
    res = conn.getresponse()
    data = res.read()
    conn.close()
    return json.loads(data.decode("utf-8"))
def get_city_identifiers(city):
    conn = http.client.HTTPSConnection("sky-scrapper.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': 
st.secrets['x-rapidapi-key']
,
        'x-rapidapi-host': "sky-scrapper.p.rapidapi.com"
    }
    conn.request("GET", f"/api/v1/flights/searchAirport?query={city}&locale=en-US", headers=headers)
    res = conn.getresponse()
    data = res.read()
    response = json.loads(data.decode("utf-8"))

    if response.get("status") and response.get("data"):
        skyId = response["data"][0]["skyId"]
        entityId = response["data"][0]["entityId"]
        return skyId, entityId
    else:
        print(f"SkyId or EntityId not found for {city}.")
        return None, None
def get_flight_details(origin_skyId, destination_skyId, origin_entityId, destination_entityId, date):
    conn = http.client.HTTPSConnection("sky-scrapper.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': 
st.secrets['x-rapidapi-key']
,
        'x-rapidapi-host': "sky-scrapper.p.rapidapi.com"
    }
    conn.request("GET", f"/api/v2/flights/searchFlightsComplete?originSkyId={origin_skyId}&destinationSkyId={destination_skyId}&originEntityId={origin_entityId}&destinationEntityId={destination_entityId}&date={date}&cabinClass=economy&adults=1&sortBy=best&currency=INR&market=en-US&countryCode=US", headers=headers)
    res = conn.getresponse()
    data = res.read()
    data_json = json.loads(data.decode("utf-8"))

    if data_json.get("status") and data_json.get("data"):
        itineraries = data_json["data"]["itineraries"]
        flight_info=[]
        flight_count=0
        for itinerary in itineraries:
            price = itinerary["price"]["formatted"]
            for leg in itinerary["legs"]:
                flight_info.append({
                    "airline": leg["carriers"]["marketing"][0]["name"],
                    "price": price,
                    "origin": leg["origin"]["name"],
                    "origin_code": leg["origin"]["displayCode"],
                    "destination": leg["destination"]["name"],
                    "destination_code": leg["destination"]["displayCode"],
                    "departure_time": leg["departure"],
                    "arrival_time": leg["arrival"],
                    "duration": leg["durationInMinutes"]
                })

                # Print the details
                # st.write(f"Airline: {airline}")
                # st.write(f"Price: {price}")
                # st.write(f"From: {origin} ({origin_code})")
                # st.write(f"To: {destination} ({destination_code})")
                # st.write(f"Departure Time: {departure_time}")
                # st.write(f"Arrival Time: {arrival_time}")
                # st.write(f"Duration: {duration} minutes")
                # st.write("="*30)
                flight_count+=1
                if flight_count>=3:
                    break
        flight_text = "\n".join(
            f"Flight {i+1}: {flight['airline']} - {flight['price']}\n"
            f"From {flight['origin']} ({flight['origin_code']}) to {flight['destination']} ({flight['destination_code']})\n"
            f"Departure: {flight['departure_time']} | Arrival: {flight['arrival_time']} | Duration: {flight['duration']} minutes\n"
            for i, flight in enumerate(flight_info)
        )

        # Use the model to generate a user-friendly itinerary summary
        response = model.generate_content(
            f"Here are some flight options for your trip. Please format the following information into a friendly, readable plan:\n{flight_text}."
        )

        # Display the model's response as the formatted itinerary
        st.write(response.text)
def parse_travel_plan(response_text):
    route = response.text

# Split the route by " *** " to separate different segments
    segments = route.split("   ")

    # Initialize lists for flight and train cities
    flight_cities = []
    train_cities = []

    # Process each segment
    for segment in segments:
        # Split the segment by " to " to get departure and arrival cities
        cities = segment.split(" to ")

        for city in cities:
            # Check if city is lowercase (flight city) or uppercase (train city)
            city=city.strip()
            if city.islower():
                flight_cities.append(city)
            elif city.isupper():
                train_cities.append(city)
    print(flight_cities,train_cities)
    return flight_cities, train_cities
# Streamlit Interface
st.title("Travel Planner")

# Initialize session state variables if not already set
if "origin" not in st.session_state:
    st.session_state.origin = ""
if "destination" not in st.session_state:
    st.session_state.destination = ""
if "journey_date" not in st.session_state:
    st.session_state.journey_date = ""
if "details_confirmed" not in st.session_state:
    st.session_state.details_confirmed = False

# Query input
user_query = st.text_input("Enter your travel query", "I have some time so I wanna go on a trip from Chennai to Prayagraj next month on the 22nd")

# Process the query when the button is clicked
if st.button("Process Query"):
    # Extract travel details
    origin, destination = extract_cities(user_query, cities)
    journey_date = extract_date(user_query)
    
    # Set session state variables for extracted information
    st.session_state.origin = origin
    st.session_state.destination = destination
    st.session_state.journey_date = journey_date
    st.session_state.details_confirmed = False  # Reset confirmation flag on new query processing

# Step 1: Display extracted information and allow confirmation or edits
if st.session_state.origin or st.session_state.destination or st.session_state.journey_date:
    st.write("### Please confirm or edit your trip details:")
    origin = st.text_input("Origin", value=st.session_state.origin if st.session_state.origin else "Not specified")
    destination = st.text_input("Destination", value=st.session_state.destination if st.session_state.destination else "Not specified")
    journey_date = st.text_input("Journey Date (YYYY-MM-DD)", value=st.session_state.journey_date if st.session_state.journey_date else "Not specified")
    
    # Confirmation button to finalize details
    if st.button("Confirm Details"):
        # Update session state with confirmed details
        st.session_state.origin = origin
        st.session_state.destination = destination
        st.session_state.journey_date = journey_date
        st.session_state.details_confirmed = True  # Set confirmation flag

# Step 2: Proceed to generate itinerary if details are confirmed
if st.session_state.details_confirmed:
    st.write(f"**Origin**: {st.session_state.origin}")
    st.write(f"**Destination**: {st.session_state.destination}")
    st.write(f"**Journey Date**: {st.session_state.journey_date}")
    
    # Ensure all essential details are provided before generating the itinerary
    if st.session_state.origin and st.session_state.destination and st.session_state.journey_date:
            response = model.generate_content(
                f"I wanna know if I am going from {origin} to {destination}, then I want it such that I can save time like mix of flight and train, can u give me the name of cities where i need to take flight and after that train, separated by a 3 space between 2 sections, flight and train, you don't need to minimize time, just give me mix of flight and train, I will check myself if that's optimal, just print flight and train where to where such that the cities in all lower are flight and cities in all caps are train, include where to where also include the origin city, make sure you stick to the format I gave you lower for flight cities and caps for train cities."
            )
            print(response.text)
            flight_cities, train_cities = parse_travel_plan(response.text)
        # print(flight_cities,train_cities)
        # Flight Segment
            if flight_cities:
                st.write("Fetching flight details...")
                flight_count=0
                for i in range(len(flight_cities) - 1):
                    origin_city = flight_cities[i]
                    destination_city = flight_cities[i + 1]
                    origin_skyId, origin_entityId = get_city_identifiers(origin_city)
                    destination_skyId, destination_entityId = get_city_identifiers(destination_city)
                    
                    if origin_skyId and destination_skyId:
                        st.write(f"\nFlights from {origin_city.title()} to {destination_city.title()} on {journey_date}:")
                        get_flight_details(origin_skyId, destination_skyId, origin_entityId, destination_entityId, journey_date)
                        


    # Train Segment
            if train_cities:
                st.write("Fetching station codes for train cities...")
                station_codes_response = model.generate_content(f"find station codes for {', '.join(train_cities)} cities, separated by commas with no spaces in between.")
                station_codes = station_codes_response.text.strip().split(",")

                if len(station_codes) != len(train_cities):
                    st.write("Failed to fetch all station codes. Please try again or adjust the station codes manually.")
                else:
                    # Train Segment
                    st.write("Fetching train details...")
                    current_date = journey_date
                    min_layover = timedelta(hours=1)
                    max_layover = timedelta(hours=6)
                    train_info_for_model = []  # To collect details to pass to model

                    for i in range(len(station_codes) - 1):
                        from_station, to_station = station_codes[i], station_codes[i + 1]
                        train_data = get_trains_between_stations(from_station, to_station, current_date)
                        found_connection = False
                        
                        if 'data' in train_data and train_data['data']:
                            for train in train_data['data']:
                                departure_time = convert_time_to_datetime(train['from_std'], train['from_day'], current_date)
                                arrival_time = convert_time_to_datetime(train['to_std'], train['to_day'], current_date)
                                
                                # Collect train details
                                train_details = {
                                    "from_station_name": train['from_station_name'],
                                    "to_station_name": train['to_station_name'],
                                    "train_number": train['train_number'],
                                    "train_name": train['train_name'],
                                    "departure": train['from_std'],
                                    "arrival": train['to_std']
                                }
                                
                                # Fetch fare details
                                fare_details = get_fare_details(train['train_number'], from_station, to_station,'2A','GN',journey_date)
                                # st.write(fare_details)
                                train_details["fare_details"] = fare_details  # Add fare details to train info

                                # Add train and fare info to list for model formatting
                                train_info_for_model.append(train_details)

                                # Layover check for connecting leg
                                if i < len(station_codes) - 2:
                                    next_station_code = station_codes[i + 2]
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
                                        st.write("  Valid connection found.")
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

                    # Send the compiled train and fare details to the model for formatting
                    formatted_train_info = model.generate_content(f"Please format the following train journey details and fares in a user-friendly way: {train_info_for_model}")
                    st.write(formatted_train_info.text)
                    


