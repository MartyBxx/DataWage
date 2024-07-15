import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from functions import standort, job_category, emp_residence

# Laden der Daten
data_path = "dataset_with_coordinates.csv"
data = pd.read_csv(data_path)

# Datenbereinigung & Preprocessing
data_cleaned = data.dropna(subset=['employee_residence_latitude', 'employee_residence_longitude',
                                   'company_location_latitude', 'company_location_longitude'])

data_cleaned['job_category'] = data_cleaned['job_title'].apply(job_category)
data_cleaned['residence'] = data_cleaned['employee_residence'].apply(emp_residence)
data_cleaned['company_continent'] = data_cleaned['company_location'].apply(standort)

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


# Streamlit-Konfiguration
st.set_page_config(page_title="DataWage: das Tool für Ihr zukünftiges Data-Gehalt", layout="wide")
st.image('banner.png')

# Dashboard Titel
st.title("DataWage: das Tool für Ihr zukünftiges Data-Gehalt")
st.subheader('Interaktives Dashboard zur Explorativen und Prädiktiven Datenanalyse')

# Sidebar für die Navigation
# st.sidebar.image('dtwg.png')
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seite auswählen", ["EDA", "Predictions", "Remote"])
data_snippet = st.sidebar.checkbox('Datensatz Überblick')
if data_snippet:
    st.subheader('Datensatz Überblick')
    st.write(data.head(5))

sns.set_palette('mako')

if page == "EDA":
    st.header("Explorative Datenanalyse")

    # Auswahl für die Visualisierung
    vis_option = st.selectbox("Wählen Sie die Visualisierung:", 
                              ["Gehaltsverteilung nach Job-Titel", "Gehaltsverteilung nach Erfahrungsstufe", 
                               "Geografische Verteilung", "Gehaltsverteilung nach Job-Kategorie"])

    if vis_option == "Gehaltsverteilung nach Job-Titel":
        sel_job = st.selectbox('Wählen Sie eine Job-Bezeichnung:', data_cleaned['job_title'].unique())
        filter_jobs = data_cleaned[data_cleaned['job_title'] == sel_job]
        # Boxplot nach Job-Titel

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='salary_in_usd', data=filter_jobs)
        plt.title(f'Gehaltsverteilung für {sel_job}')
        plt.xlabel('Gehalt in USD')
        plt.grid(True)

        st.pyplot(plt.gcf())

        # plt.figure(figsize=(12, 8))
        # sns.boxplot(x='salary_in_usd', y= 'job_title', data=data_cleaned, showfliers=False)
        # plt.title('Gehaltsverteilung nach Job-Titel')
        # plt.xlabel('Gehalt in USD')
        # plt.ylabel('Job-Titel')
        # plt.grid(True)
        # st.pyplot(plt.gcf())

    elif vis_option == "Gehaltsverteilung nach Job-Kategorie":
        # Boxplot nach Job-Titel
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='salary_in_usd', y='job_category', data=data_cleaned)
        plt.title('Gehaltsverteilung nach Job-Kategorie')
        plt.xlabel('Gehalt in USD')
        plt.ylabel('Job-Kategorie')
        plt.grid(True)
        st.pyplot(plt.gcf())

    elif vis_option == "Gehaltsverteilung nach Erfahrungsstufe":
        # Boxplot nach Erfahrungsstufe
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='salary_in_usd', y='experience_level', data=data_cleaned)
        plt.title('Gehaltsverteilung nach Erfahrungsstufe')
        plt.xlabel('Gehalt in USD')
        plt.ylabel('Erfahrungsstufe')
        plt.grid(True)
        st.pyplot(plt.gcf())

    elif vis_option == "Geografische Verteilung":
        # Interaktive Karte
        map_option = st.selectbox("Wählen Sie die Ansicht:", ["Wohnorte der Mitarbeiter", "Standorte der Unternehmen"])
        
        if map_option == "Wohnorte der Mitarbeiter":
            fig = px.scatter_geo(data_cleaned, lat='employee_residence_latitude', lon='employee_residence_longitude',
                                 hover_name='employee_residence', size='salary_in_usd',
                                 title='Geografische Verteilung der Wohnorte der Mitarbeiter')
        else:
            fig = px.scatter_geo(data_cleaned, lat='company_location_latitude', lon='company_location_longitude',
                                 hover_name='company_location', size='salary_in_usd',
                                 title='Geografische Verteilung der Unternehmensstandorte')
        
        st.plotly_chart(fig)

elif page == "Predictions":
    st.header("Prädiktive Modellierung")

    # Features auswählen
    st.sidebar.subheader("Features für das Modell auswählen:")
    selected_features = st.sidebar.multiselect("Wählen Sie die Features:", 
                                               ["experience_level", "employment_type", "job_title", 
                                                "remote_ratio", "company_size", "company_location"])
    
    st.sidebar.subheader("Wählen Sie den Algorithmus:")
    algorithm = st.sidebar.selectbox("Algorithmus", 
                                     ["Linear Regression", "Logistic Regression","Decision Tree Class", "Decision Tree Regr", "Random Forest Class", "Random Forest Regr", "KNN Class", "KNN Regr"])

    # Datenaufbereitung für das Modell
    if selected_features:
        X = pd.get_dummies(data_cleaned[selected_features])
        y = data_cleaned['salary_in_usd']
        
        # Train-Test-Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modellauswahl
        if algorithm == "Linear Regression":
            model = LinearRegression()
        elif algorithm == "Logistic Regression":
            model = LogisticRegression(max_iter= 100)
        elif algorithm == "Decision Tree Class":
            model = DecisionTreeClassifier(random_state=42)
        elif algorithm == "Decision Tree Regr":
            model = DecisionTreeRegressor(random_state=42)
        elif algorithm == "Random Forest Class":
            model = RandomForestClassifier(random_state=42)
        elif algorithm == "Random Forest Regr":
            model = RandomForestRegressor(random_state=42)
        elif algorithm == "KNN Class":
            model = KNeighborsClassifier()
        elif algorithm == "KNN Regr":
            model = KNeighborsRegressor()
        
        # Modelltraining
        
        model.fit(X_train, y_train)
        
        # Vorhersagen
        y_pred = model.predict(X_test)
        
        # Metriken berechnen
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        show_metrics = st.sidebar.checkbox("Metriken anzeigen")

        if show_metrics:
        
        # Anzeige der Metriken
            st.write(f"**Algorithmus**: {algorithm}")
            st.write(f"**Mean Absolute Error (MAE)**: {mae:.2f}")
            st.write(f"**Mean Squared Error (MSE)**: {mse:.2f}")
            st.write(f"**Root Mean Squared Error (RMSE)**: {rmse:.2f}")
            st.write(f"**R²-Score**: {r2:.2f}")
        
        # Eingaben für die Vorhersage
        st.subheader("Gehalt Predictions")
        prediction_input = {}
        for feature in selected_features:
            unique_values = data_cleaned[feature].unique()
            prediction_input[feature] = st.selectbox(f"Wählen Sie {feature}:", unique_values)
        
        if st.button("Predict"):
            input_data = pd.DataFrame([prediction_input])
            input_data_encoded = pd.get_dummies(input_data).reindex(columns=X.columns, fill_value=0)
            salary_prediction = model.predict(input_data_encoded)
            st.write(f"Vorhergesagtes Gehalt in USD: {salary_prediction[0]:.2f}")
    else:
        st.write("Bitte wählen Sie mindestens ein Feature aus, um das Modell zu trainieren.")

elif page == "Remote":
    st.header("Remote")

    # Filtermöglichkeiten
    st.sidebar.subheader("Filter:")
    job_title_selected = st.sidebar.multiselect("Job-Bezeichnung auswählen:", data_cleaned['job_title'].unique())
    experience_level_selected = st.sidebar.multiselect("Erfahrungsstufe auswählen:", data_cleaned['experience_level'].unique())
    employment_type_selected = st.sidebar.multiselect("Art der Anstellung auswählen:", data_cleaned['employment_type'].unique())
    company_location_selected = st.sidebar.multiselect("Unternehmensstandort auswählen:", data_cleaned['company_location'].unique())
    company_size_selected = st.sidebar.multiselect("Unternehmensgröße auswählen:", data_cleaned['company_size'].unique())

    

    # Anwenden der Filter
    filtered_data = data_cleaned
    if job_title_selected:
        filtered_data = filtered_data[filtered_data['job_title'].isin(job_title_selected)]
    if experience_level_selected:
        filtered_data = filtered_data[filtered_data['experience_level'].isin(experience_level_selected)]
    if employment_type_selected:
        filtered_data = filtered_data[filtered_data['employment_type'].isin(employment_type_selected)]
    if company_location_selected:
        filtered_data = filtered_data[filtered_data['company_location'].isin(company_location_selected)]
    if company_size_selected:
        filtered_data = filtered_data[filtered_data['company_size'].isin(company_size_selected)]

    # Scatter Plot: Gehalt vs. Remote-Arbeit
    st.subheader("Scatter Plot: Gehalt vs. Remote-Arbeit")
    fig = px.scatter(filtered_data, x='remote_ratio', y='salary_in_usd', color='experience_level',
                     hover_data=['job_title'], title='Gehalt in USD vs. Anteil der Remote-Arbeit', height= 600)
    st.plotly_chart(fig)

    # Heatmap für Korrelationen
    # st.subheader("Korrelationen zwischen numerischen Variablen")
    # correlation_matrix = filtered_data[['salary_in_usd', 'remote_ratio']].corr()
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    # st.pyplot(plt.gcf())
