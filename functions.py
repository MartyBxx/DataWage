import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry as pyc

def job_category(job_title):
    data_engineering = ["Data Engineer", "Data Analyst", "Analytics Engineer",
                        "BI Data Analyst", "Business Data Analyst", "BI Developer",
                        "BI Analyst", "Business Intelligence Engineer",
                        "BI Data Engineer", "Power BI Developer"]
    
    data_scientist = ["Data Scientist", "Applied Scientist", "Research Scientist",
                        "3D Computer Vision Researcher", "Deep Learning Researcher",
                        "AI/Computer Vision Engineer"]
    
    machine_learning = ["Machine Learning Engineer", "ML Engineer",
                        "Lead Machine Learning Engineer", "Principal Machine Learning Engineer"]
    
    data_architecture = ["Data Architect", "Big Data Architect", "Cloud Data Architect", "Principal Data Architect"]
    
    management = ["Data Science Manager", "Director of Data Science",
                  "Head of Data Science", "Data Scientist Lead", "Head of Machine Learning",
                  "Manager Data Management", "Data Analytics Manager"]
    
    if job_title in data_engineering:
        return "Data Engineering"
    elif job_title in data_scientist:
        return "Data Science"
    elif job_title in machine_learning:
        return "Machine Learning"
    elif job_title in data_architecture:
        return "Data Architecture"
    elif job_title in management:
        return "Management"
    else:
        return "Other"
    
def emp_residence(emp_res):

    north_america = ['US', 'CA', 'MX']

    central_america = ['HN', 'CR', 'PR', 'DO']

    south_america = ['EC', 'PE', 'CO', 'BO', 'BR', 'CL', 'AR']

    europe = ['AT', 'ES', 'FR', 'IT', 'MD', 
              'NL', 'EE', 'HU', 'NO', 'DK',
              'LT', 'LV', 'DE', 'FI', 'HR',
              'CH', 'GR', 'PL', 'UA', 'PT',
              'BA', 'IE', 'MT', 'RO',
              'CZ', 'SI', 'BE', 'AD', 'SE',
              'BG', 'RS', 'LU', 'GB', 'RU']
    
    middle_east = ['GE', 'IL', 'AE',
                   'SA', 'TR', 'OM', 'LB',
                   'PK', 'AM', 'QA', 'UZ',
                   'KW', 'CY', 'IR',
                   'IQ']
    
    asia = ['IN', 'PH', 'VN', 'TH',
            'KR', 'JP', 'SG', 'CN',
            'ID', 'MY']
    
    africa = ['ZA', 'KE', 'EG', 'NG', 'UG',
              'MU', 'TN', 'GH', 'CF',
              'DZ']
    
    oceania = ['AU', 'NZ', 'AS']

    if emp_res in north_america:
        return 'North America'
    elif emp_res in central_america:
        return 'Central America'
    elif emp_res in south_america:
        return 'South America'
    elif emp_res in europe:
        return 'Europe'
    elif emp_res in middle_east:
        return 'Middle East'
    elif emp_res in asia:
        return 'Asia'
    elif emp_res in africa:
        return 'Africa'
    elif emp_res in oceania:
        return 'Oceania'
    else:
        return 'Other'
    
def standort(standort):

    north_america = ['US', 'CA', 'MX']

    central_america = ['HN', 'CR', 'PR', 'DO']

    south_america = ['EC', 'PE', 'CO', 'BO', 'BR', 'CL', 'AR']

    europe = ['AT', 'ES', 'FR', 'IT', 'MD', 
              'NL', 'EE', 'HU', 'NO', 'DK',
              'LT', 'LV', 'DE', 'FI', 'HR',
              'CH', 'GR', 'PL', 'UA', 'PT',
              'BA', 'IE', 'MT', 'RO',
              'CZ', 'SI', 'BE', 'AD', 'SE',
              'BG', 'RS', 'LU', 'GB', 'RU']
    
    middle_east = ['GE', 'IL', 'AE',
                   'SA', 'TR', 'OM', 'LB',
                   'PK', 'AM', 'QA', 'UZ',
                   'KW', 'CY', 'IR',
                   'IQ']
    
    asia = ['IN', 'PH', 'VN', 'TH',
            'KR', 'JP', 'SG', 'CN',
            'ID', 'MY']
    
    africa = ['ZA', 'KE', 'EG', 'NG', 'UG',
              'MU', 'TN', 'GH', 'CF',
              'DZ']
    
    oceania = ['AU', 'NZ', 'AS']

    if standort in north_america:
        return 'North America'
    elif standort in central_america:
        return 'Central America'
    elif standort in south_america:
        return 'South America'
    elif standort in europe:
        return 'Europe'
    elif standort in middle_east:
        return 'Middle East'
    elif standort in asia:
        return 'Asia'
    elif standort in africa:
        return 'Africa'
    elif standort in oceania:
        return 'Oceania'
    else:
        return 'Other'

def country_con(code):
    """
    Konvertiert einen ISO 3166-1 Alpha-2 oder Alpha-3 Ländercode in den entsprechenden Ländernamen.
    
    Args:
        code (str): Der ISO 3166-1 Alpha-2 oder Alpha-3 Ländercode.
        
    Returns:
        str: Der entsprechende Ländername oder eine Fehlermeldung, wenn der Code ungültig ist.
    """
    try:
        country = pyc.countries.get(alpha_2=code.upper()) or pyc.countries.get(alpha_3=code.upper())
        if country:
            return country.name
        else:
            return f"Ungültiger Ländercode: {code}"
    except Exception as e:
        return str(e)
