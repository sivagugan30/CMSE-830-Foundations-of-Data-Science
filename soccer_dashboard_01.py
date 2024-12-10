import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_classif
import matplotlib.pyplot as plt 

# Set Streamlit app config for a wider layout and light theme
st.set_page_config(layout="wide", page_title="", initial_sidebar_state="expanded")

# Set background image or color dynamically based on the page
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )




# Load data
df = pd.read_csv('cleaned_df.csv')

skill_columns = [
                                'finishing', 'dribbling', 'curve', 'crossing', 'heading_accuracy',
                                'long_shots', 'shot_power', 'short_passing', 'vision', 'ball_control',
                                'standing_tackle', 'sliding_tackle', 'interceptions', 'defensive_awareness',
                                'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes'
                            ]

# Page navigation state
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a Section", ["Home","Instructions", "Hypothesis Testing", "Types of Players", "Analyse a Player", "What Player to Buy?","Data Handling"])


# Function to go home
if options == "Home":
    st.session_state.page = 'home'
elif options == "Instructions":
    st.session_state.page = 'Instructions'
elif options == "Hypothesis Testing":
    st.session_state.page = 'hypothesis_testing'
elif options == "Types of Players":
    st.session_state.page = 'types_of_players'
elif options == "Analyse a Player":
    st.session_state.page = 'individual_player_analysis'
elif options == "What Player to Buy?":
    st.session_state.page = 'what_player_to_buy'
elif options == "Data Handling":
    st.session_state.page = 'data_handling'


# Homepage
if st.session_state.page == 'home':
    set_background(image_path='garnacho.jpeg')
    st.header("")
    st.markdown('<h1 style="color: white;">Soccer Analytics Dashboard 2024</h1>', unsafe_allow_html=True)

elif st.session_state.page == 'Instructions':

    # Description
    st.subheader("About the Dashboard :")
    st.write(
        "This dashboard is designed to provide insights into player performance, categories, "
        "and recommendations using tools like **Streamlit**, **Plotly**, and **Pandas**. "
        "Analyze and interact with soccer player data seamlessly."
    )
    
    # Features
    st.subheader("Sections : ")
    st.write("- **Data Collection & Preparation:** Gathering, Cleaning, and Preprocessing Data..")
    st.write("- **Hypothesis Testing:** Statistical analysis of player performance.")
    st.write("- **Types of Players:** Displays top players across categories like Creative, Athletic, Defensive, and Goalkeeping with visualizations.")
    st.write("- **Individual Player Analysis:** Detailed insights and comparisons for selected players.")
    st.write("- **What Player to Buy?:** Personalized player recommendations based on selected attributes.")
    st.write("")
    # Closing Note
    st.write("Explore through the dashboard using the Navigation bar to explore, and enjoy the world of soccer analytics!")



# Types of Players Section
elif st.session_state.page == 'types_of_players':
    st.title("Explore Soccer Section")
    
    # Creating the layout within the section to avoid scrolling
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Explore Categories")
            categories = {
                "Creative": ['vision', 'short_passing', 'dribbling', 'long_passing'],
                "Athletic": ['sprint_speed', 'stamina', 'strength', 'acceleration'],
                "Defensive": ['standing_tackle', 'interceptions', 'defensive_awareness', 'sliding_tackle'],
                "Goalkeeping": ['gk_diving', 'gk_reflexes', 'gk_handling', 'gk_positioning'],
                "Attacking": ['finishing', 'shot_power', 'volleys', 'crossing']
            }

            selected_category = st.selectbox("Select a Category", list(categories.keys()))

            # Get relevant columns for the selected category
            selected_columns = categories[selected_category]

            # Calculate category-specific creativity score
            score_name = f"{selected_category.lower()}_score"
            df[score_name] = df[selected_columns].mean(axis=1)

            # Get top 10 players based on the creativity score
            top_players = df.sort_values(by=score_name, ascending=False).head(10)

            # Display top players
            st.write(f"Top 10 Players - {selected_category}")
            st.write(top_players[['player_name', 'overall_rating'] + selected_columns])

        with col2:
            st.subheader("3D Visualization of Players in Selected Category")
            # Creating a 3D plot for the selected category
            fig_3d = px.scatter_3d(
                df,
                x=selected_columns[0],
                y=selected_columns[1],
                z=selected_columns[2],
                color=score_name,  # Use creativity score for color
                hover_name='player_name',
                #size=selected_columns[3],
                title=f"3D Plot of Players based on {selected_category} Skills",
                height=500,
                template= 'plotly_dark',  # Set Plotly chart to light mode
                color_continuous_scale = 'Blackbody'
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        # 2D Plots for Top 5 Players
        st.subheader("2D Visualizations of Skills")

        # Plot 1
        fig_2d_1 = px.scatter(
            df,
            x=selected_columns[0],
            y=selected_columns[1],
            color=score_name,  # Use creativity score for color
            hover_name='player_name',
            title=f"Joint Plot : {selected_columns[0].replace('_', ' ').title()} vs {selected_columns[1].replace('_', ' ').title()}",
            template='plotly_dark',
            marginal_x = 'histogram',
            marginal_y = 'histogram'
        )
        fig_2d_1.update_traces(
            selector = dict(type='histogram'),
            nbinsx = 10,
            nbinsy = 10,
            marker_color='rgba(255, 0, 0, 0.8)'
        )
        st.plotly_chart(fig_2d_1, use_container_width=True)

        # Plot 2
        fig_2d_2 = px.scatter(
            df,
            x=selected_columns[2],
            y=selected_columns[3],
            color=score_name,  # Use creativity score for color
            hover_name='player_name',
            title=f"Joint Plot : {selected_columns[2].replace('_', ' ').title()} vs {selected_columns[3].replace('_', ' ').title()}",
            template='plotly_dark',
            marginal_x = 'histogram',
            marginal_y = 'histogram'
        )
        fig_2d_2.update_traces(
            selector = dict(type='histogram'),
            nbinsx = 10,
            nbinsy = 10,
            marker_color='rgba(255, 0, 0, 0.8)'
        )

        st.plotly_chart(fig_2d_2, use_container_width=True)










elif st.session_state.page == 'individual_player_analysis':
    st.sidebar.title("Filters")
    team_selected = st.sidebar.selectbox("Select a Team", df['team'].unique())
    position_selected = st.sidebar.selectbox("Select a Position", df['core_position'].unique())
    
    # Filter players based on selected team and position
    players_filtered = df[(df['team'] == team_selected) & (df['core_position'] == position_selected)]

    # Set Mbappe as the default player selection if he is available in the filtered list
    player_selected = st.sidebar.selectbox("Select a Player", players_filtered['player_name'].unique())

    # Get player data
    player_data = df[df['player_name'] == player_selected]

    if not player_data.empty:
        # Display player's name as the title
        st.title(f"{player_selected}")

        # Create DataFrame for player personal details in a custom format
        # Create DataFrame for player personal details in a row-wise format
        player_details = pd.DataFrame({
            'Attribute': [
                'Height',
                'Weight',
                'Foot',
                'Age',
                'Overall Rating',
                'Best Position',
                'Contract Start',
                'Contract End',
                'No of Positions',
                'Wage',
                'Market Value',
                'Position'
            ],
            'Value': [
                player_data['height'].values[0],
                player_data['weight'].values[0],
                player_data['foot'].values[0],
                player_data['age'].values[0],
                player_data['overall_rating'].values[0],
                player_data['best_position'].values[0],
                player_data['contract_start'].values[0],
                player_data['contract_end'].values[0],
                player_data['no_of_playable_positions'].values[0],
                player_data['wage'].values[0],
                player_data['market_value'].values[0],
                player_data['core_position'].values[0],
            ]
        })


        # Create two columns for layout
        col1, col2 = st.columns([1, 2])  # Adjust the width ratio as needed

        # Display the DataFrame in the first column
        with col1:
            st.subheader("Player Details")
            st.write(player_details)  # This will fit the left side of the screen.

        # Define categories and calculate scores
        categories = {
            "Creative": ['vision', 'short_passing', 'dribbling', 'long_passing'],
            "Athletic": ['sprint_speed', 'stamina', 'strength', 'acceleration'],
            "Defensive": ['standing_tackle', 'interceptions', 'defensive_awareness', 'sliding_tackle'],
            "Goalkeeping": ['gk_diving', 'gk_reflexes', 'gk_handling', 'gk_positioning'],
            "Attacking": ['finishing', 'shot_power', 'volleys', 'crossing']
        }

        radar_values = []
        for category, skills in categories.items():
            score = player_data[skills].mean(axis=1).values[0]
            radar_values.append(score)

        radar_categories = list(categories.keys())

        # Calculate average scores for players in the same position
        position_data = df[df['core_position'] == position_selected]
        avg_category_scores = {category: position_data[skills].mean(axis=1).mean() for category, skills in categories.items()}
        avg_radar_values = list(avg_category_scores.values())

        fig_radar_compare = go.Figure()

        # Add trace for the selected player
        fig_radar_compare.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=radar_categories,
            fill='toself',
            name=f"{player_selected}",
            marker=dict(size=8, color='rgba(0, 123, 255, 0.7)')
        ))

        # Add trace for average performance
        fig_radar_compare.add_trace(go.Scatterpolar(
            r=avg_radar_values,
            theta=radar_categories,
            fill='toself',
            name=f'Avg {position_selected}',
            marker=dict(size=8, color='rgba(255, 0, 0, 0.5)')
        ))

        fig_radar_compare.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100]),
            ),
            showlegend=True,
            template='plotly_dark',
            title=f"{player_selected} vs average {position_selected.lower()}s (Benchmarking)"
        )

        # Display the radar plot in the second column
        with col2:
            st.plotly_chart(fig_radar_compare, use_container_width=True)

        # Filter skills excluding goalkeeping skills
        non_gk_skills = [
            'finishing', 'dribbling', 'curve', 'crossing', 'heading_accuracy',
            'long_shots', 'shot_power', 'short_passing', 'vision', 'ball_control',
            'standing_tackle', 'sliding_tackle', 'interceptions', 'defensive_awareness',
            'sprint_speed', 'stamina', 'strength', 'acceleration', 'balance'
        ]

        # Get player skills
        player_skills = player_data[non_gk_skills].T.reset_index()
        player_skills.columns = ['Skill', 'Value']

        # Get top 5 and bottom 5 skills
        top_skills = player_skills.nlargest(5, 'Value')
        bottom_skills = player_skills.nsmallest(5, 'Value')

        # Create two columns for side-by-side layout
        col1, col2 = st.columns(2)

        # Top Skills Bar Plot
        with col1:
            fig_top = px.bar(
                top_skills,
                x='Skill',
                y='Value',
                title=f'Top Skills of {player_selected}',
                template='plotly_dark',
                color_discrete_sequence=['#1f77b4']  # Color for top skills (blue)
            )
            fig_top.update_layout(
                yaxis=dict(range=[0, 100])  # Set y-axis range from 0 to 100
            )
            st.plotly_chart(fig_top, use_container_width=True)
        
        # Bottom Skills Bar Plot
        with col2:
            fig_bottom = px.bar(
                bottom_skills,
                x='Skill',
                y='Value',
                title=f'Bottom Skills of {player_selected}',
                template='plotly_dark',
                color_discrete_sequence=['#ff7f0e']  # Color for bottom skills (orange)
            )
            fig_bottom.update_layout(
                yaxis=dict(range=[0, 100])  # Set y-axis range from 0 to 100
            )
            st.plotly_chart(fig_bottom, use_container_width=True)
       
        # Scatter Plot for User-Selected Skills
        st.subheader("Scatter Plot: Compare Skills")

        # Set default values for X and Y skills
        skill_x = st.selectbox("Select Skill for X-axis", non_gk_skills, index=non_gk_skills.index('shot_power'))
        skill_y = st.selectbox("Select Skill for Y-axis", non_gk_skills, index=non_gk_skills.index('ball_control'))

        fig_scatter = px.scatter(
            df,
            x=skill_x,
            y=skill_y,
            hover_name='player_name',
            title=f"Scatter Plot: {skill_x.replace('_', ' ').title()} vs {skill_y.replace('_', ' ').title()}",
            template='plotly_dark'
        )

        # Highlight the selected player in red
        fig_scatter.add_trace(go.Scatter(
            x=[player_data[skill_x].values[0]],
            y=[player_data[skill_y].values[0]],
            mode='markers',
            marker=dict(color='red', size=20, symbol='star'),
            name=f"{player_selected}"
        ))

        st.plotly_chart(fig_scatter, use_container_width=True)


# Hypothesis Testing Page
if st.session_state.page == 'hypothesis_testing':
    st.title("Hypothesis Testing")

    # Sidebar for picking a hypothesis
    hypothesis_options = [
        "1. Lefties vs. Righties",
        "2. Age vs Market Value",
        "3. Position and Height",
        "4. Wage vs Market Value",
        "5. Branded Players and Market Value",
        "6. Weight vs Agility"
    ]

    selected_hypothesis = st.sidebar.selectbox("Pick a Hypothesis", hypothesis_options)

   # Display full hypothesis description based on selection
    if selected_hypothesis == "1. Lefties vs. Righties":
        st.header("Lefties vs. Righties: Who's Got the Better Skills?")
        st.write("Do left-footed players have a unique advantage in core football skills? Let’s analyze this with some fun comparisons!")

        # Skills to compare
        skills = ['dribbling', 'passing', 'curve', 'shot_power', 'finishing', 'aggression', 'balance']

        # Create a box plot for each skill
        for skill in skills:
        # Create a box plot using Plotly Express
            fig = px.box(
                df,
                x='foot',
                y=skill,
                color='foot',
                title=f"{skill.capitalize()}",
                category_orders={'foot': ['Left', 'Right']},  # Ensure Left is on the left side
                color_discrete_sequence=['blue', 'orange']  # Set colors for left and right
            )

            # Update layout for better visualization
            fig.update_layout(
                yaxis_title=skill.capitalize(),
                height=600
            )

            st.plotly_chart(fig)
            
    elif selected_hypothesis == "2. Age vs Market Value":
        st.header("Age vs Market Value")
        st.write("Do players age like fine wine, enhancing their market value as time goes on? Let's find out!")

        # Plotting code for Age vs Market Value
        fig = px.scatter(df, x='age', y='market_value', hover_name='player_name', title="Age vs Market Value")
        st.plotly_chart(fig)

    elif selected_hypothesis == "3. Position and Height":
        st.header("Position and Height")
        st.write("Do player heights differ by position on the pitch? Let’s take a closer look at how height varies among various core roles!")

        # Box plot for Position and Height
        fig = px.box(df, x='core_position', y='height', title="Height by Core Position", width=800, height=600)
        st.plotly_chart(fig)

    elif selected_hypothesis == "4. Wage vs Market Value":
        st.header("Wage vs Market Value")
        st.write("Is there a significant relationship between a player’s wage and their market value, indicating that higher wages reflect higher perceived value? Let’s explore!")

        # Plotting code for Wage vs Market Value
        fig = px.scatter(df, x='wage', y='market_value', hover_name='player_name', title="Wage vs Market Value")
        st.plotly_chart(fig)


    elif selected_hypothesis == "5. Branded Players and Market Value":
        st.header("Branded Players and Market Value")
        st.write("Are players with higher international reputation more valuable in the market, reflecting their brand power? Let's find out!")

        # Bar plot for average market value by international reputation
        fig2 = px.bar(df.groupby('international_reputation')['market_value'].mean().reset_index(), 
                      x='international_reputation', y='market_value', 
                      title="Average Market Value by International Reputation", 
                      #range_y=[1, 5] 
                      )
        st.plotly_chart(fig2)

    elif selected_hypothesis == "6. Weight vs Agility":
        st.header("Weight and Athletic Metrics")
        st.write("Does a player's weight affect their athleticism? Let’s dive into how weight influences speed and stamina, minus the heavy lifting of confusion!")

        # Box plots for Weight vs Athletic Metrics
        fig_acceleration = px.box(df, x='weight', y='acceleration', title="Weight vs Acceleration", width=800, height=600)
        st.plotly_chart(fig_acceleration)

        fig_sprint_speed = px.box(df, x='weight', y='sprint_speed', title="Weight vs Sprint Speed", width=800, height=600)
        st.plotly_chart(fig_sprint_speed)

        fig_stamina = px.box(df, x='weight', y='stamina', title="Weight vs Stamina", width=800, height=600)
        st.plotly_chart(fig_stamina)

    # Conclusion Section
    st.subheader("Conclusion")
    if selected_hypothesis == "1. Lefties vs. Righties":
        st.write("As we can see, left-footed players tend to excel in curves, while right-footed players shine in finishing. It seems each foot has its own flair for the beautiful game!")
    
    elif selected_hypothesis == "2. Age vs Market Value":
        st.write("As players age, their market value can vary significantly. Clubs often seek players they can groom and develop, hoping that these investments will appreciate in market value over time. Younger players typically hold more potential for growth, while older players may bring experience and immediate impact but are seen as less likely to increase in value.")

    elif selected_hypothesis == "3. Position and Height":
        st.write("Height varies by position, with goalkeepers being the tallest due to their need to cover the entire goalpost and make crucial saves. Defenders also tend to be tall, which aids in clearing balls during corner kicks and long passes.")

    elif selected_hypothesis == "4. Wage vs Market Value":
     st.write("Yes, there is a positive correlation, as clubs typically pay players according to their perceived worth. When players feel underpaid relative to their market value, they may seek opportunities at other clubs where they believe they will be fairly compensated for their contributions. This dynamic ensures that players are motivated and feel valued, fostering a competitive environment.")

    elif selected_hypothesis == "5. Branded Players and Market Value":
        st.write("Players with a higher international reputation often have inflated market values, reflecting their global recognition, fan engagement, and commercial appeal. This brand power not only boosts their value in the transfer market but also enhances merchandise sales and sponsorship opportunities for clubs, making them key assets beyond just their on-field contributions.")

    elif selected_hypothesis == "6. Weight vs Agility":
        st.write("Weight significantly impacts athletic performance. Finding the right balance can be key to maximizing speed and agility while maintaining stamina.")







































elif st.session_state.page == 'what_player_to_buy':
    st.title("What Player to Buy?")

    # Get only numerical features
    numerical_features = df.select_dtypes(include=['number']).columns.to_list()

    # Prepare data for model
    X = df[numerical_features].drop(columns=['market_value'], errors='ignore')
    y = df['market_value']
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model Evaluation Function
    def evaluate_model(model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mape, r2
    
    # Train and evaluate models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    # Predefined results
    data = {
        'Model': ['Linear Regression', 'Random Forest Regressor', 'Gradient Boosting'],
        'MAPE': [0.124, 0.076, 0.082],
        'R²': [0.974906, 0.978413, 0.979113]
    }
    
    # Create DataFrame for results
    df_results = pd.DataFrame(data)
    
    df1 = df.copy()

    # Input filters for player recommendations
    st.subheader("Player Criteria Filters")

    position = st.selectbox("Position", df1['core_position'].unique(), key="position")
    age = st.slider("Age", 18, 40, (22, 30), step=1, key="age")
    budget = st.slider("Budget (Market Value)", 0, 200000000, (10000000, 100000000), step=500000, key="budget")

    skill_columns = ['finishing', 'dribbling', 'defending', 'passing']
    skill1 = st.selectbox("Important Skill 1", skill_columns, index=0, key="skill1")
    skill2 = st.selectbox("Important Skill 2", skill_columns, index=1, key="skill2")

    # Filter players based on criteria
    filtered_players = df1[
        (df1['core_position'] == position) &
        (df1['age'] >= age[0]) & (df1['age'] <= age[1]) &
        (df1['market_value'] >= budget[0]) & (df1['market_value'] <= budget[1])
    ]

    import streamlit as st

    st.subheader("Model Selection")
    
    # Display model evaluation metrics
    st.write("Based on the Model Evaluation metrics, please choose your desired model:")
    
    st.write(df_results)
    
    selected_model_name = st.selectbox("Choose Model for Market Value Prediction", list(models.keys()), key="model_selection")
    # Make the button larger using custom CSS
    st.markdown("""
        <style>
        .big-button {
            font-size: 40px;
            padding: 15px 40px;
            width: 250px;
            height: 100px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Display the button with the custom class
    if st.button("Predict", key="predict_button", help="Click to predict player values", use_container_width=True):
        selected_model = models[selected_model_name]
        
        # Fit the model and predict on filtered players
        if not filtered_players.empty:
            X_filtered = filtered_players[numerical_features].drop(columns=['market_value'], errors='ignore')
            X_filtered_scaled = scaler.transform(X_filtered)
    
            selected_model.fit(X_train_scaled, y_train)  # Fit model on training data
            filtered_players['predicted_value'] = selected_model.predict(X_filtered_scaled)  # Predict for filtered players
            filtered_players['predicted_value'] = filtered_players['predicted_value'].astype(int)  # Convert predicted values to int
            filtered_players = filtered_players.sort_values(by='predicted_value', ascending=False)
    
            # Display predicted market value with the selected skills
            st.subheader("Top players to buy based on your criteria")
            st.write(filtered_players[['player_name', 'predicted_value', skill1, skill2]].head(20))  # Use skill1 and skill2 directly
        else:
            st.write("No players found based on your criteria.")
    
        # Plot skill comparison
        fig_skill_comparison = px.scatter(
            filtered_players.head(20),
            x=skill1,
            y=skill2,
            color='market_value',
            hover_name='player_name',
            title=f"Joint Plot: {skill1.replace('_', ' ').title()} vs {skill2.replace('_', ' ').title()}",
            labels={
                skill1: skill1.replace('_', ' ').title(),
                skill2: skill2.replace('_', ' ').title()
            },
            template='plotly_dark',
            marginal_x='histogram',
            marginal_y='histogram'
        )
        
        # Update the histogram colors and bin size
        fig_skill_comparison.update_traces(
            selector=dict(type='histogram'),
            marker_color='rgba(255, 0, 0, 0.8)',  # Set the histogram color to a semi-transparent red
            nbinsx=10,
            nbinsy=10
        )
        
        # Display the plot
        st.plotly_chart(fig_skill_comparison, use_container_width=True)

    
    
    
    
    
    
    
    
    











if st.session_state.page == 'data_handling':
    
    tab1, tab2, tab3, tab4 = st.tabs(["Data Cleaning & Pre-processing", "Data Processing & Feature Engineering", "Feature Selection", "Modelling"])

    st.write("")
    
    with tab1:
        st.write('')
        st.write('Github Repository : https://github.com/sivagugan30/CMSE-830-Foundations-of-Data-Science')
    
        st.subheader('1. Data Collection')
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        import plotly.subplots as sp
        import plotly.graph_objects as go
        from sklearn.linear_model import LinearRegression
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer, KNNImputer
        from plotly.subplots import make_subplots
    
        # Set display options and warnings
        pd.set_option('display.max_columns', None)
    
        #read df 1
        stats_df = pd.read_csv('stats_df.csv')
    
    
        #read df2 
        personal_df = pd.read_csv('personal_df.csv')
    
        
        st.write(" Data Source 1 : https://www.kaggle.com/datasets/yorkyong/football-player-statistics ")
        st.write(" Data Source 2 : https://www.kaggle.com/datasets/davidcariboo/player-scores ")
        
        # Create two columns in Streamlit for side-by-side display
        col1, col2 = st.columns(2)
        
        # Display stats_df in the first column
        with col1:
            st.subheader("Data Source 1")
            st.dataframe(stats_df.head(3))
        
        # Display personal_df in the second column
        with col2:
            st.subheader("Data Source 2")
            st.dataframe(personal_df.head(3))
    
    
        
        # Merge datasets
        df = pd.merge(personal_df, stats_df, on=['player_name', 'team', 'best_position'])
    
        # Display the first few rows of the dataset
        st.subheader("Merged Dataset")
        st.write(df.head(5))
        
        st.subheader("2. Imputation")
        # Create a heatmap using Plotly

        # Select first 10 columns and 'penalties' column
        columns_to_display = df.columns[:10].tolist() + ['penalties'] 
        
        # Subset the DataFrame with the selected columns
        df_subset = df[columns_to_display]

        fig = px.imshow(
            df_subset.isna().T,  # Transpose to align columns on the y-axis
            color_continuous_scale='viridis',
            aspect="auto",
            labels={'color': 'Missingness'},  # Change the color bar label
            title="Missing Data Heatmap"
        )
        
        # Display the Plotly heatmap in Streamlit
        st.plotly_chart(fig)
    
        # Induce MCAR missingness
        df1 = df.copy()
        missing_percentage = 0.7  # 70% missingness
        num_missing = int(missing_percentage * len(df1))
        missing_indices = np.random.choice(df1.index, num_missing, replace=False)
        df1.loc[missing_indices, 'penalties'] = np.nan
    
        df_subset2 = df1[columns_to_display]
        # Create a heatmap using Plotly
        fig = px.imshow(
            df_subset2.isna().T,  # Transpose to align columns on the y-axis
            color_continuous_scale='viridis',
            aspect="auto",
            labels={'color': 'Missingness'},  # Change the color bar label
            title="Induced Missingness in the Penalties Column"
        )
        
        # Display the Plotly heatmap in Streamlit
        st.plotly_chart(fig)
    
        st.write('Induced Missingness in the Penalties Column is MCAR (Missing Completely at Random) - Missingness is independent of other columns')
    
        
        
        # Correlation analysis
        numeric_columns = df1.select_dtypes(include=[np.number]).columns.tolist()
        correlation_matrix = df1[numeric_columns].corr()
        penalties_correlation = correlation_matrix['penalties']
        highly_correlated_columns = penalties_correlation[penalties_correlation.abs() > 0.5].drop('penalties')
    
        # Create a DataFrame for plotting highly correlated features
        highly_correlated_df = highly_correlated_columns.reset_index()
        highly_correlated_df.columns = ['Feature', 'Correlation']
        # Remove rows where the 'Feature' starts with 'gk_'
        highly_correlated_df = highly_correlated_df[~highly_correlated_df['Feature'].str.startswith('gk_')]
        
        # Reset the index after filtering (optional, if you want a clean index)
        highly_correlated_df.reset_index(drop=True, inplace=True)

        # Create bar chart for highly correlated features
        fig = px.bar(highly_correlated_df, 
                    x='Correlation', 
                    y='Feature', 
                    title='Highly Correlated Features with Penalties',
                    labels={'Correlation': 'Correlation Coefficient'},
                    orientation='h')
        st.plotly_chart(fig)
    
        st.write("Based on the correlation analysis, the 'Finishing' and 'Volleys' features show a strong relationship with 'Penalties'. Since 'Finishing' is more intuitive, it will be used as the primary regressor for various types of imputation.")
    
    
        
    
        # Imputation using Linear Regression
        numeric_columns = ['penalties', 'finishing']
        numeric_columns.remove('penalties')
    
        df_train = df1[df1['penalties'].notna()]
        df_missing = df1[df1['penalties'].isna()]
        X_train = df_train[numeric_columns]
        y_train = df_train['penalties']
    
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
    
        X_missing = df_missing[numeric_columns]
        penalties_pred = lr_model.predict(X_missing)
    
        df1_linear = df1.copy()
        df1_linear.loc[df1_linear['penalties'].isna(), 'penalties'] = penalties_pred
    
        # Imputation using Stochastic Regression
        y_train_pred = lr_model.predict(X_train)
        residuals = y_train - y_train_pred
        mean_residual = residuals.mean()
        std_residual = residuals.std()
    
        random_noise = np.random.normal(mean_residual, std_residual, size=len(penalties_pred))
        penalties_pred_stochastic = penalties_pred + random_noise
    
        df1_stochastic = df1.copy()
        df1_stochastic.loc[df1_stochastic['penalties'].isna(), 'penalties'] = penalties_pred_stochastic
    
        # Imputation using MICE
        df1_mice = df1.copy()
        mice_imputer = IterativeImputer(random_state=42)
        df1_mice[numeric_columns] = mice_imputer.fit_transform(df1_mice[numeric_columns])
    
        # Imputation using KNN
        df1_knn = df1.copy()
        knn_imputer = KNNImputer(n_neighbors=5)
        df1_knn[numeric_columns] = knn_imputer.fit_transform(df1_knn[numeric_columns])
    
    
        # Calculate the mean and mode for the 'penalties' column
        mean_value = df1['penalties'].mean()
        mode_value = df1['penalties'].mode()  # Get the first mode if there are multiple
    
        # Mean Imputation
        df1_mean = df1.copy()
        df1_mean['penalties'] = df1_mean['penalties'].fillna(mean_value)
    
        # Mode Imputation
        df1_mode = df1.copy()
        df1_mode['penalties'] = df1_mode['penalties'].fillna(mode_value)
    
        # Comparison of different Imputation techniques
        imputed_penalties = pd.DataFrame({
            'Mean' : df1_mean['penalties'],
            'Mode' : df1_mode['penalties'],
            'Linear Regression': df1_linear['penalties'],
            'Stochastic Regression': df1_stochastic['penalties'],
            'MICE': df1_mice['penalties'],
            'KNN': df1_knn['penalties']
    
        })
    
        # Create a boxplot using Plotly
        fig = px.box(
            imputed_penalties,  # Your DataFrame with imputed penalties
            title='Comparison of Imputed Penalties',
            labels={'value': 'Penalties', 'variable': 'Imputation Method'}
        )
        
        # Update layout for better readability
        fig.update_layout(
            yaxis_title="Penalties",
            xaxis_title="Imputation Method",
            xaxis_tickangle=45,  # Rotate x-axis labels
            showlegend=False,
            width=900,  # Adjust plot width
            height=600  # Adjust plot height
        )
        
        # Display the boxplot in Streamlit
        st.plotly_chart(fig)
    
        st.write(" In the above box plot, the box for the mean imputed df is compressed due to the imputation of all missing values at the mean. As a result, the mean, median, mode, upper fence, and lower fence all converge to the same value, indicating a lack of variability in the imputed data ")    # Scatter plot comparison of imputation methods
        
        
        def create_combined_scatter_plot(dfs, titles):
            fig = sp.make_subplots(rows=3, cols=2, subplot_titles=titles)
    
            for i, (df, title) in enumerate(zip(dfs, titles)):
                sample_df = df.sample(n=min(100, len(df)), random_state=42)
                fig.add_trace(
                    go.Scatter(x=sample_df['finishing'], y=sample_df['penalties'], mode='markers', name=title,
                            marker=dict(size=10, opacity=0.7)),
                    row=(i // 2) + 1, col=(i % 2) + 1
                )
    
                trendline = np.polyfit(sample_df['finishing'], sample_df['penalties'], 1)
                trendline_func = np.poly1d(trendline)
                x_range = np.linspace(sample_df['finishing'].min(), sample_df['finishing'].max(), 100)
                fig.add_trace(
                    go.Scatter(x=x_range, y=trendline_func(x_range), mode='lines', name='Trendline', line=dict(dash='dash')),
                    row=(i // 2) + 1, col=(i % 2) + 1
                )
    
            fig.update_layout(title_text="Comparison of Imputation Methods", 
                            height=700, width=900, showlegend=False)
            return fig
    
        # Prepare DataFrames for scatter plots
        dfs = [
            df1_mean[['finishing', 'penalties']], 
            df1_mode[['finishing', 'penalties']],
            df1_linear[['finishing', 'penalties']],
            df1_stochastic[['finishing', 'penalties']],
            df1_mice[['finishing', 'penalties']],
            df1_knn[['finishing', 'penalties']]
        ]
    
        titles = [
            'Mean Imputation', 
            'Mode Imputation', 
            'Linear Regression Imputation', 
            'Stochastic Regression Imputation', 
            'MICE Imputation', 
            'KNN Imputation'
        ]
    
        # Create the combined scatter plot
        #st.subheader("Scatter Plot Comparison of Imputation Methods")
        scatter_plot = create_combined_scatter_plot(dfs, titles)
        st.plotly_chart(scatter_plot)
    
        st.write("At first glance, mean and mode imputations significantly distort the data. In contrast, linear and stochastic regression methods maintain the positive correlation essential for modeling, although they lack variance. MICE and KNN, however, strike a balance between variability and correlation")
    
        st.write("Based on the plots above, stochastic regression appears to be the best fit for imputing missing data. It effectively reconstructs missing values while preserving variable relationships, ensuring correlation and variance are maintained")
    
    
    
    
    
        
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        from imblearn.over_sampling import SMOTE

    
        # Assuming 'df' is your DataFrame and 'foot' is the target variable
        st.subheader("3. Class Imbalance")
        st.write("Original Class Distribution")
        y = df['foot']
        st.write(y.value_counts())
    
        # Visualize the original class distribution
        original_count = y.value_counts().reset_index()
        original_count.columns = ['Foot', 'Count']
        fig_original = px.bar(original_count, x='Foot', y='Count', title='Class Imbalance - Foot Column (Original)')
        st.plotly_chart(fig_original)
    
        numeric_columns = df.describe().columns.to_list()
        # Apply SMOTE for class balancing
        X = df[numeric_columns]  # Use numeric_columns for features
        y = df['foot']  # Target
    
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Display new class distribution after SMOTE
        st.write("Class Distribution After SMOTE")
        new_count = pd.Series(y_resampled).value_counts().reset_index()
        new_count.columns = ['Foot', 'Count']
        st.write(new_count)
    
        
    
        
        # Visualize the new class distribution
        fig_after_smote = px.bar(new_count, x='Foot', y='Count', title='Class Balance - Foot Column (After SMOTE)')
        st.plotly_chart(fig_after_smote)
    
        # Convert X_resampled and y_resampled into a DataFrame
        resampled_data = pd.DataFrame(X_resampled, columns=numeric_columns)  # Use numeric_columns for the columns
        resampled_data['foot'] = y_resampled
    
        st.write("SMOTE anslysis")
        
        # Scatter plots for visualizing before and after SMOTE
        fig_scatter = make_subplots(rows=2, cols=2, subplot_titles=('Before SMOTE: Right Foot', 'After SMOTE: Right Foot',
                                                                    'Before SMOTE: Left Foot', 'After SMOTE: Left Foot'))
    
        # Scatter plot for "Right" foot before SMOTE
        fig_scatter.add_trace(
            go.Scatter(x=df[df['foot'] == 'Right']['dribbling'], 
                    y=df[df['foot'] == 'Right']['balance'], 
                    mode='markers', 
                    name='Before SMOTE: Right Foot'),
            row=1, col=1
        )
    
        # Scatter plot for "Right" foot after SMOTE
        fig_scatter.add_trace(
            go.Scatter(x=resampled_data[resampled_data['foot'] == 'Right']['dribbling'], 
                    y=resampled_data[resampled_data['foot'] == 'Right']['balance'], 
                    mode='markers', 
                    name='After SMOTE: Right Foot'),
            row=1, col=2
        )
    
        # Scatter plot for "Left" foot before SMOTE
        fig_scatter.add_trace(
            go.Scatter(x=df[df['foot'] == 'Left']['dribbling'], 
                    y=df[df['foot'] == 'Left']['balance'], 
                    mode='markers', 
                    name='Before SMOTE: Left Foot'),
            row=2, col=1
        )
    
        # Scatter plot for "Left" foot after SMOTE
        fig_scatter.add_trace(
            go.Scatter(x=resampled_data[resampled_data['foot'] == 'Left']['dribbling'], 
                    y=resampled_data[resampled_data['foot'] == 'Left']['balance'], 
                    mode='markers', 
                    name='After SMOTE: Left Foot'),
            row=2, col=2
        )
    
        # Update layout
        fig_scatter.update_layout(title_text='Scatter Plots for Foot Data', height=700)
        st.plotly_chart(fig_scatter)
        
        # Add a comment for the SMOTE plot
        st.write("In this plot, the majority class on the right is untouched, while the minority class (left) has been oversampled using the SMOTE algorithm. This technique generates synthetic samples for the minority class, resulting in a distribution that mirrors the original variable but with an increased number of data points ")






















        

    with tab3:
        #st.title("Feature Selection")
        from scipy.stats import chi2_contingency
        from scipy.stats import ttest_ind
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.preprocessing import StandardScaler
    
    
        st.subheader("1. Categorical Feature Selection")
                # Make a copy of the DataFrame
        df = pd.read_csv('cleaned_df.csv')
        df1 = df.copy()
        
        # Define categorical features
        categorical_features = ['foot', 'best_position', 'core_position', 'age_brackets', 'team']
        
        # Bin 'market_value' into 5 buckets labeled 1 to 5
        df1['market_value_bins'] = pd.qcut(df1['market_value'], q=5, labels=[1, 2, 3, 4, 5])
        
        # Initialize results
        chi_square_results = {}
        
        # Perform Chi-Square test for each feature
        for feature in categorical_features:
            contingency_table = pd.crosstab(df1[feature], df1['market_value_bins'])
            _, p, _, _ = chi2_contingency(contingency_table)
            chi_square_results[feature] = 1 - p  # Store 1 - p-value
        
        # Convert results to a DataFrame
        chi_square_df = pd.DataFrame.from_dict(chi_square_results, orient='index', columns=['1-p'])
        chi_square_df.sort_values(by='1-p', ascending=False, inplace=True)
        
        # Create a new column to categorize the color based on 1-p value
        chi_square_df['Color'] = chi_square_df['1-p'].apply(lambda x: '#228b22' if x > 0.95 else '#d35400')
        
        # Plot results using Plotly
        fig = go.Figure()
        
        # Add bar chart for 1-p values with conditional colors
        for feature, row in chi_square_df.iterrows():
            fig.add_trace(go.Bar(
                x=[feature],
                y=[row['1-p']],
                marker=dict(color=row['Color']),
                name=feature,
                showlegend=False  # Hide legend for each bar
            ))
        
        # Add a horizontal line at 0.95
        fig.add_trace(go.Scatter(
            x=chi_square_df.index,
            y=[0.95] * len(chi_square_df),
            mode='lines',
            line=dict(color='red', width=4),  # Thick red dashed line
            name='Threshold (0.95)',
            showlegend=True  # Hide legend for the threshold line
        ))
        
        # Update layout
        fig.update_layout(
            title='Chi-Square Test for Categorical Features',
            xaxis=dict(title='Features'),
            yaxis=dict(title='1 - p-value'),
            template='plotly_dark',
            showlegend=True,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.write(
        "Chi-square tests were applied to find important categorical features that influence the target variable, 'market_value'. The significant features are highlighted in green."
    )
    
        # Section 2: T-Test
        st.subheader("2. Numerical Feature Selection ")
        
        st.markdown('<h5 style="font-size: 18px;">2.1 T-Test</h5>', unsafe_allow_html=True)
        
        numeric_features = df1.describe().columns.to_list()
        
        # Remove features starting with 'gk_'
        numeric_features_filtered = [col for col in numeric_features if not col.startswith('gk_')]
        numeric_features_filtered.remove('market_value')
        
        # Prepare results for T-Test
        t_test_results = {}
        
        # Perform T-Test for each numerical feature (after removing 'gk_' features)
        for feature in numeric_features_filtered:
            unique_bins = df1['market_value_bins'].unique()
            bin_groups = [df1[df1['market_value_bins'] == bin][feature] for bin in unique_bins]
            
            # Perform pairwise t-test between the first two bins as an example
            t_stat, p_value = ttest_ind(bin_groups[0], bin_groups[1], equal_var=False, nan_policy='omit')
            t_test_results[feature] = 1 - p_value  # Store 1 - p-value
        
        # Convert results to a DataFrame
        t_test_df = pd.DataFrame.from_dict(t_test_results, orient='index', columns=['1-p'])
        t_test_df.sort_values(by='1-p', ascending=False, inplace=True)
        
        # Categorize scores
        def categorize_score(val):
            if val > 0.95:
                return 'High'
            elif 0.5 <= val <= 0.95:
                return 'Medium'
            else:
                return 'Low'
        
        t_test_df['Score Category'] = t_test_df['1-p'].apply(categorize_score)
        
        # Filter 5 features from each category
        high_features = t_test_df[t_test_df['Score Category'] == 'High'].iloc[-10:-1]
        medium_features = t_test_df[t_test_df['Score Category'] == 'Medium'].head(5)
        low_features = t_test_df[t_test_df['Score Category'] == 'Low'].head(5)
        
        # Combine the top features into a single DataFrame
        top_features_df = pd.concat([high_features, medium_features, low_features])
        
        # Plot results using Plotly
        fig = go.Figure()
        
        # Add bar chart for 1-p values with different shades of blue
        fig.add_trace(go.Bar(
            x=high_features.index,
            y=high_features['1-p'],
            #name='High Significance',
            marker=dict(color='#228b22')  # Different shade of blue
            ,showlegend=False
        ))
        
        fig.add_trace(go.Bar(
            x=medium_features.index,
            y=medium_features['1-p'],
            #name='Medium Significance',
            marker=dict(color='#d35400')  # Lighter shade of blue
            ,showlegend=False
        ))
        
        fig.add_trace(go.Bar(
            x=low_features.index,
            y=low_features['1-p'],
            #name='Low Significance',
            marker=dict(color='#d35400')  # Even lighter shade of blue
            ,showlegend=False
        ))
        
        # Add a horizontal red thick dashed line at 0.95
        fig.add_trace(go.Scatter(
            x=top_features_df.index,
            y=[0.95] * len(top_features_df),
            mode='lines',
            line=dict(color='red', width=4),  # Thick red dashed line
            name='Threshold (0.95)'
        ))
        
        # Update layout
        fig.update_layout(
            title='T-Test for Numerical Features',
            xaxis=dict(title='Features'),
            yaxis=dict(title='1 - p-value'),
            template='plotly_dark',
            showlegend=True,
            height=600
        )
    
        st.plotly_chart(fig, use_container_width=True)
        st.write("T-tests were used to compare the average values between different groups and identify which numerical features are most important for predicting 'market_value'. The significant features are highlighted in green.")
        
        st.markdown('<h5 style="font-size: 18px;">2.2 ANOVA & Mutual Information</h5>', unsafe_allow_html=True)
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_classif
    
        # Assuming df is your DataFrame containing the features and target
        
        X = df[numeric_features_filtered]
        y = df['market_value']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Mutual Information calculation
        mi = mutual_info_regression(X_scaled, y)
        mi_series = pd.Series(mi, index=X.columns)
        
        # Perform ANOVA F-test using SelectKBest with f_classif
        anova_selector = SelectKBest(f_classif, k='all')
        anova_selector.fit(X_scaled, y)
        anova_scores = pd.Series(anova_selector.scores_, index=X.columns)
        
        # Sort and select the top 15 features for both metrics
        top_mi_15 = mi_series.sort_values(ascending=False).head(15)
        top_anova_15 = anova_scores.sort_values(ascending=False).head(15)
        
        # Find the common top 10 features between both metrics
        common_features = top_mi_15.index.intersection(top_anova_15.index).tolist()[:10]
        
        # Filter the top 10 common features from both metrics
        top_mi = top_mi_15[common_features]
        top_anova = top_anova_15[common_features]
    
        # Create scatter plot comparing Mutual Information and ANOVA F-Value
        fig = go.Figure()
        
        # Add Mutual Information (x-axis)
        fig.add_trace(go.Scatter(
            x=top_mi.index,
            y=top_mi.values,
            mode='markers+lines',  # Connect adjacent points only
            name='Mutual Information (MI)',
            marker=dict(color='limegreen', size=10),
            yaxis='y1',
            line=dict(shape='spline'),  # Connect adjacent points only
            hovertemplate='<b>Feature:</b> %{x}<br><b>Mutual Information:</b> %{y:.4f}<extra></extra>'  # Hover for Mutual Information
        ))
        
        # Add ANOVA F-Value (y-axis)
        fig.add_trace(go.Scatter(
            x=top_anova.index,
            y=top_anova.values,
            mode='markers+lines',  # Connect adjacent points only
            name='ANOVA F-Value',
            marker=dict(color='royalblue', size=10),
            yaxis='y2',
            line=dict(shape='hv'),  # Connect adjacent points only
            hovertemplate='<b>Feature:</b> %{x}<br><b>ANOVA F-Value:</b> %{y:.4f}<extra></extra>'  # Hover for ANOVA F-Value
        ))
        
        # Add annotations indicating better values
        fig.add_annotation(
            text="Higher MI = more shared information btw feature and target",
            xref="paper", yref="paper",
            x=0.05, y=1.15, showarrow=False,
            font=dict(color="limegreen", size=12),
            xanchor='left'
        )
        
        fig.add_annotation(
            text="Higher ANOVA F-Value = greater feature separation",
            xref="paper", yref="paper",
            x=0.95, y=1.15, showarrow=False,
            font=dict(color="royalblue", size=12),
            xanchor='right'
        )
    
        # Configure layout with increased height and the title
        fig.update_layout(
            xaxis=dict(title='Features'),
            yaxis=dict(
                title='Mutual Information(MI)',
                side='left'
            ),
            yaxis2=dict(
                title='ANOVA F-Value',
                side='right',
                overlaying='y',
                showgrid=False
            ),
            template='plotly_dark',
            legend=dict(x=0.5, y=1.1, orientation='h'),
            height=600,  # Adjusted plot height
        )
        
        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    
        st.write(
        "ANOVA and Mutual Information methods were employed to rank numerical features and assess their relationship with 'market_value'."
    )
    
    
        st.subheader("3. Dimensionality Reduction")
        
        from sklearn.decomposition import PCA
        
        # Select numerical columns from df1
        numerical_df = df1[numeric_features_filtered]
        
        # Standardizing the data before applying PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numerical_df)
        
        # Apply PCA
        pca = PCA()
        pca.fit(scaled_data)
        
        # Explained Variance Ratio (how much variance is captured by each principal component)
        explained_variance = pca.explained_variance_ratio_
        
        # Cumulative explained variance
        cumulative_explained_variance = np.cumsum(explained_variance)
         
        # Create the Plotly figure
        fig = go.Figure()
        
        # Add trace for the cumulative explained variance
        fig.add_trace(go.Scatter(
            x=list(range(1, len(explained_variance) + 1)),
            y=cumulative_explained_variance,
            mode='lines+markers',
            name='Cumulative Explained Variance',
            line=dict(color='blue', width=3),
            marker=dict(symbol='circle', size=8)
        ))
        fig.add_vline(
            x=25, 
            line=dict(color='red', width=4),
            annotation_text="Trade-off point: 95% variance", 
            annotation_position="top right"
        )
    
        # Add titles and labels
        fig.update_layout(
            title='Scree Plot - Principal Component Analysis (PCA)',
            xaxis_title='Principal Components',
            yaxis_title='Variance % Explained',
            template='plotly_dark',
            height=600
        )
        
        # Display the plot in Streamlit
        st.plotly_chart(fig)
        
        st.write(
        "Principal Component Analysis (PCA) was applied to reduce dimensionality, capturing 95% of the variance with the just 25 components. This selection was made to balance accuracy and computational cost."
        )

 




# Footer
st.sidebar.markdown("---")
st.sidebar.write("Created by Sivagugan Jayachandran")

