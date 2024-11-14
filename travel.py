import os
import re
import spacy
import pandas as pd
import http.client
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import dateparser  # Additional import for date parsing
from fuzzywuzzy import process
def extract_date(query):
    doc = nlp(query)
    today = datetime.today()
    
    # Predefined days of the week for reference
    days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    
    # Process each entity in the query to check for dates
    for ent in doc.ents:
        if ent.label_ == "DATE":
            date_text = ent.text.lower()
            
            # 1. Handle relative terms like "tomorrow" and "day after tomorrow"
            if "tomorrow" in date_text:
                return (today + timedelta(days=1)).strftime('%Y-%m-%d')
            elif "day after tomorrow" in date_text:
                return (today + timedelta(days=2)).strftime('%Y-%m-%d')
            
            # 2. Handle specific weekday terms like "next Monday"
            elif "next" in date_text:
                for day in days_of_week:
                    if day in date_text:
                        day_num = days_of_week.index(day)
                        days_ahead = (day_num - today.weekday() + 7) % 7
                        days_ahead = days_ahead if days_ahead > 0 else 7
                        return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
            
            # 3. Handle expressions like "21st," "21st of next month"
            match = re.search(r"(\d{1,2})(st|nd|rd|th)?", date_text)
            if match:
                day = int(match.group(1))
                
                # Check if month is mentioned (e.g., "21st November")
                month_match = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)", date_text)
                if month_match:
                    month_text = month_match.group(1)
                    parsed_date = dateparser.parse(f"{day} {month_text} {today.year}")
                    if parsed_date and parsed_date >= today:
                        return parsed_date.strftime('%Y-%m-%d')
                
                # Handle "next month on 21st"
                if "next month" in date_text:
                    next_month = today.month % 12 + 1
                    year = today.year if next_month > today.month else today.year + 1
                    parsed_date = datetime(year, next_month, day)
                    if parsed_date:
                        return parsed_date.strftime('%Y-%m-%d')
                
                # Handle standalone day of the current month
                parsed_date = dateparser.parse(f"{day} {today.strftime('%B')} {today.year}")
                if parsed_date and parsed_date >= today:
                    return parsed_date.strftime('%Y-%m-%d')
            
            # 4. Parse expressions like "Nov 21" or "21st Nov" using dateparser
            parsed_date = dateparser.parse(date_text, settings={'PREFER_DATES_FROM': 'future'})
            if parsed_date:
                return parsed_date.strftime('%Y-%m-%d')
    
    # If no specific date found, return None or a default date
    return None
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

# Query preprocessing
def preprocess_query(query):
    query = re.sub(r"[\(\)]", "", query)  # Remove parentheses
    query = re.sub(r"\s+", " ", query)    # Remove extra spaces
    return query.strip()

# Find the closest matching city based on cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process  # Add this package for fuzzy matching

# Precompute vector representations of the cities
def precompute_city_vectors(original_cities):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([city.lower() for city in original_cities])
    return vectorizer, vectors

# Find closest city with both cosine similarity and fuzzy matching
def find_closest_city(entity, original_cities, vectorizer, city_vectors, threshold=0.8):
    # Fuzzy matching for direct name similarity
    best_match, fuzzy_score = process.extractOne(entity, original_cities)
    
    # If fuzzy match is above a certain threshold, return it
    if fuzzy_score >= threshold * 100:  # fuzzy_score is from 0 to 100
        return best_match
    
    # Otherwise, fall back to cosine similarity
    entity_vector = vectorizer.transform([entity.lower()])
    cosine_sim = cosine_similarity(entity_vector, city_vectors)
    closest_city_index = cosine_sim.argmax()
    return original_cities[closest_city_index]

# Precompute vectors once
vectorizer, city_vectors = precompute_city_vectors(cities)

# Extract budget from query
def extract_budget(doc):
    for ent in doc.ents:
        if ent.label_ == "MONEY" and ("inr" in ent.text.lower() or "rupee" in ent.text.lower() or "₹" in ent.text):
            return re.sub(r'[^\d]', '', ent.text)
    for token in doc:
        if token.like_num:
            return token.text
    return None

# Extract preferences from query
def extract_preferences(doc):
    preferences = {'cheapest': False, 'fastest': False, 'direct': False}
    for token in doc:
        if token.text in preferences:
            preferences[token.text] = True
    return preferences
def extract_cities(query, cities):
    query = query.lower()
    origin, destination = None, None
    
    # Attempt to find "from" and "to" cities directly
    from_match = re.search(r"from\s+(\w+)", query)
    to_match = re.search(r"to\s+(\w+)", query)

    if from_match:
        origin_candidate = from_match.group(1)
        origin = find_closest_city(origin_candidate, cities, vectorizer, city_vectors)
    
    if to_match:
        destination_candidate = to_match.group(1)
        destination = find_closest_city(destination_candidate, cities, vectorizer, city_vectors)
        print(destination)
    
    # Fall back to spaCy if keywords "from" or "to" aren't used
    if not origin or not destination:
        doc = nlp(query)
        extracted_entities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]  # GPE for Geo-Political Entities (places)
        matched_cities = [find_closest_city(entity, cities, vectorizer, city_vectors) for entity in extracted_entities]
        
        # Assign based on position in the query, if available
        if len(matched_cities) >= 2:
            origin, destination = matched_cities[0], matched_cities[1]
        elif len(matched_cities) == 1:
            origin = origin or matched_cities[0]

    return origin, destination
# Process query
query = "I have some time so I wanna go on a trip from Chennai to Prayagraj next month 22nd"
cleaned_query = preprocess_query(query)
doc = nlp(cleaned_query)
# matched_cities = [find_closest_city(entity, cities, vectorizer, city_vectors) for entity in extracted_entities if entity.lower()]
# origin = matched_cities[0] if matched_cities else None
# destination = matched_cities[1] if len(matched_cities) > 1 else None
origin,destination=extract_cities(query,cities)
budget = extract_budget(doc)
preferences = extract_preferences(doc)
date=extract_date(cleaned_query)
print(origin,destination)
# Generate response for stations and station codes
response1 = model.generate_content(f"find train from {origin} to {destination} I just wanna know how to reach there direct more preferrably direct if possible or using connecting trains ok , just wanna know station names that I will have to change(if required) , ok just telll me that , i dont neeed a train , just station names, give me name of stations seperated by comma, dont give space after comma")
print(response1.text)
response2 = model.generate_content(f"find me station codes for {response1.text} stations, just give codes seperated by comma, just give codes ")
print(response2.text)
stations = list(response2.text.split(', '))
stations = [station.strip() for station in stations]
print(stations)
# Setup IRCTC API connection
conn = http.client.HTTPSConnection("irctc1.p.rapidapi.com")
headers = {
    'x-rapidapi-key': "b23f4ed7b9msh80d0c1957ac19f2p19c71djsn936ce06cd38a",
    'x-rapidapi-host': "irctc1.p.rapidapi.com"
}
initial_date_of_journey = date if not None else "2024-10-30"

# Convert time string to datetime object
def convert_time_to_datetime(time_str, day_offset, journey_date):
    base_date = datetime.strptime(journey_date, "%Y-%m-%d")
    return base_date + timedelta(days=day_offset, hours=int(time_str.split(":")[0]), minutes=int(time_str.split(":")[1]))

# Fetch trains between two stations
def get_trains_between_stations(from_station, to_station, date_of_journey):
    conn.request("GET", f"/api/v3/trainBetweenStations?fromStationCode={from_station}&toStationCode={to_station}&dateOfJourney={date_of_journey}", headers=headers)
    res = conn.getresponse()
    data = res.read()
    return json.loads(data.decode("utf-8"))

# Minimum and maximum layover times
min_layover = timedelta(hours=1)
max_layover = timedelta(hours=6)
current_date = initial_date_of_journey

# Search for trains with valid connections
for i in range(len(stations) - 1):
    start_station_code = stations[i]
    end_station_code = stations[i + 1]
    print(f"Searching for trains from {start_station_code} to {end_station_code} on {current_date}...")
    train_data = get_trains_between_stations(start_station_code, end_station_code, current_date)

    if 'data' in train_data and train_data['data']:
        trains = train_data['data']
        found_connection = False

        for train in trains:
            departure_time = convert_time_to_datetime(train['from_std'], train['from_day'], current_date)
            arrival_time = convert_time_to_datetime(train['to_std'], train['to_day'], current_date)
            
            # Display train details
            print(f"Train from {train['from_station_name']} to {train['to_station_name']}:")
            print(f"  Train Number: {train['train_number']}")
            print(f"  Train Name: {train['train_name']}")
            print(f"  Departure Time: {train['from_std']} on {departure_time.strftime('%Y-%m-%d')}")
            print(f"  Arrival Time: {train['to_std']} on {arrival_time.strftime('%Y-%m-%d')}")
            print(f"  Duration: {train['duration']}")

            # Check for the next leg
            if i < len(stations) - 2:
                next_station_code = stations[i + 2]
                next_leg_date = arrival_time.strftime('%Y-%m-%d')
                next_trains = get_trains_between_stations(end_station_code, next_station_code, next_leg_date)

                valid_next_train = None
                for next_train in next_trains['data']:
                    next_departure_time = convert_time_to_datetime(next_train['from_std'], next_train['from_day'], next_leg_date)
                    layover_time = next_departure_time - arrival_time
                    if min_layover <= layover_time <= max_layover:
                        valid_next_train = next_train
                        break

                if valid_next_train:
                    print("  Valid connection found:")
                    print(f"    Connecting Train Number: {valid_next_train['train_number']}")
                    print(f"    Connecting Train Name: {valid_next_train['train_name']}")
                    print(f"    Departure Time: {valid_next_train['from_std']} after layover of {layover_time}")
                    current_date = next_departure_time.strftime('%Y-%m-%d')
                    found_connection = True
                    break
                else:
                    print("  No valid connecting train found.")
                    found_connection = False
            else:
                found_connection = True
                break

        if not found_connection:
            print(f"No trains found from {start_station_code} to {end_station_code} on {current_date}.")
            break
    else:
        print(f"No trains found from {start_station_code} to {end_station_code} on {current_date}.")
        break

# Close the IRCTC API connection
conn.close()
