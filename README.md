# Travel Planner: Mixed Flight and Train Itinerary Generator

This project is a smart travel planning tool that generates optimized travel itineraries, combining flights and trains to save time on long journeys. The tool uses natural language processing to parse a casual travel query and automatically provides a travel plan with a mix of flights and trains. The output is formatted to be user-friendly, making it easy to understand and visualize travel options.

## Features

- **Natural Language Query Parsing**: Users can input a casual travel query (e.g., "Plan a trip from Chennai to Prayagraj next month on the 22nd"), and the tool will interpret it to find origin, destination, and date.
- **Optimal Route Generation**: The tool combines flights and trains based on origin and destination, with flights used for longer segments to save time.
- **Fare and Time Information**: Provides detailed information for each segment, including fares for different classes on trains and pricing for flights.
- **Formatted Itinerary**: The output is generated as a well-organized, readable itinerary using Google Generative AI, giving a friendly overview of each part of the journey.
  
## Requirements

- Python 3.7+
- [Streamlit](https://streamlit.io/) for the web interface
- NLP libraries and APIs:
  - [spaCy](https://spacy.io/) for language processing
  - [Google Generative AI](https://developers.generativeai.google/) for text formatting
  - [RapidAPI](https://rapidapi.com/) access for IRCTC and flight details
- CSV file with Indian cities (`Indian_cities.csv`)

## Setup Instructions

1. **Clone the Repository** and navigate to the project directory:
   ```bash
   git clone https://github.com/your-username/travel-planner.git
   cd travel-planner
