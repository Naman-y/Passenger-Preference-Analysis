# Passenger Preference Analysis in the Domestic Airline Industry

## Project Overview
This project examines passenger preferences in the domestic airline industry using survey data. The analysis focuses on the factors that influence airline selection, including demographics, travel habits, price sensitivity, loyalty program participation, and service-related preferences.

The project also includes a Streamlit dashboard for presenting the analysis in an interactive format.

## Objectives
- Identify the main factors influencing airline choice
- Analyze passenger demographics and travel behavior
- Study the relationship between price sensitivity and loyalty
- Segment passengers based on preference patterns
- Compare airline-specific drivers of customer choice
- Present the findings through a dashboard

## Project Structure
```text
Passenger-Preference-Analysis/
|-- Dashboard.py
|-- combined_dataset_cleaned.csv
|-- airline_segmentation_ready.csv
|-- airline_model_ready.csv
|-- requirements.txt
|-- Group_3_Airline.pdf
|-- README.md
`-- ML_Engg/
    |-- descriptive_analytics.ipynb
    |-- predictive_modelling.ipynb
    `-- segmentation_ready.ipynb
```

## Dataset
The project uses cleaned and processed survey data stored in CSV format.

### Files Used
- `combined_dataset_cleaned.csv` for the main dashboard analysis
- `airline_segmentation_ready.csv` for customer segmentation
- `airline_model_ready.csv` for airline-specific driver analysis

### Dataset Size
- Main dataset: `2099` rows and `28` columns

### Key Attributes
- Gender
- Age
- Occupation
- Purpose of travel
- Travel frequency
- Travel class
- Flight preference
- Booking mode
- Price sensitivity
- Loyalty program participation
- Reward preference
- Schedule preference
- In-flight priority
- Airline last flown

## Methodology
The project follows the steps below:

1. Data collection through survey responses
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Feature preparation
5. Customer segmentation
6. Airline-specific analysis
7. Dashboard development

## Dashboard Sections
The Streamlit dashboard includes the following sections:

- `Overview`
- `Demographics`
- `Travel Behavior`
- `Price & Loyalty`
- `Airline & Sentiment`
- `Customer Segmentation`
- `Airline-Specific Drivers`

## Tools and Technologies
- Python
- Pandas
- NumPy
- Plotly
- Streamlit
- Jupyter Notebook

## How to Run
Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard from the `Passenger-Preference-Analysis` folder:

```bash
streamlit run Dashboard.py
```

## Notebook Files
The `ML_Engg` folder contains the notebooks used for different stages of the analysis:

- `descriptive_analytics.ipynb`
- `predictive_modelling.ipynb`
- `segmentation_ready.ipynb`

## Deployment
The dashboard for this project has been deployed online and can be accessed using the link below:

[Passenger Preference Dashboard](https://naman-y-passenger-preference-analysis-dashboard-r9ow1a.streamlit.app/)


## Summary
This project presents a business analytics study of passenger behavior in the domestic airline industry. It combines data analysis, customer segmentation, and dashboard-based visualization to support interpretation of passenger preferences and airline-related decision factors.
