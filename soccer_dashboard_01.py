import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import base64

import matplotlib.pyplot as plt 

# Set Streamlit app config for a wider layout and light theme
st.set_page_config(layout="wide", page_title="", initial_sidebar_state="expanded")

# Set background image using HTML and CSS
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

pic = 'pic3.jpg'
set_background(pic)

# Load data
df = pd.read_csv('cleaned_df.csv')
df = df.drop('Unnamed: 0', axis=1)

# Check for duplicate columns and log them
duplicates = df.columns[df.columns.duplicated()].unique()
if len(duplicates) > 0:
    st.warning(f"Duplicate columns found: {duplicates.tolist()}")
    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

# Rename the value column to market_value
df.rename(columns={'value': 'market_value'}, inplace=True)  # Ensure 'value' exists in the DataFrame

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
options = st.sidebar.radio("Select a Section", ["Home", "Hypothesis Testing", "Types of Players", "Individual Player Analysis", "What Player to Buy?"])


# Function to go home
if options == "Home":
    st.session_state.page = 'home'
elif options == "Hypothesis Testing":
    st.session_state.page = 'hypothesis_testing'
elif options == "Types of Players":
    st.session_state.page = 'types_of_players'
elif options == "Individual Player Analysis":
    st.session_state.page = 'individual_player_analysis'
elif options == "What Player to Buy?":
    st.session_state.page = 'what_player_to_buy'


# Homepage
if st.session_state.page == 'home':
    st.title("Soccer Analytics Dashboard 2024")
    st.header("")

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
            template='plotly_white',
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
            template='plotly_white',
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
                template='plotly_white'
            )
            st.plotly_chart(fig_top, use_container_width=True)

        # Bottom Skills Bar Plot
        with col2:
            fig_bottom = px.bar(
                bottom_skills,
                x='Skill',
                y='Value',
                title=f'Bottom Skills of {player_selected}',
                template='plotly_white'
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
            template='plotly_white'
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


# What Player to Buy Section
elif st.session_state.page == 'what_player_to_buy':
    st.title("What Player to Buy?")

    # Input filters for player recommendations
    st.sidebar.title("Player Criteria Filters")

    position = st.selectbox("Position", df['core_position'].unique())
    
    # Age slider
    age = st.slider("Age", 18, 40, (22, 30), step=1)

    # Budget slider with default values set to 10M to 100M
    budget = st.slider("Budget (Market Value)", 0, 200000000, (10000000, 100000000), step=500000)

    skill1 = st.selectbox("Important Skill 1", skill_columns, index=skill_columns.index('finishing'))
    skill2 = st.selectbox("Important Skill 2", skill_columns, index=skill_columns.index('dribbling'))

    # Filter players based on criteria
    filtered_players = df[
        (df['core_position'] == position) &
        (df['age'] >= age[0]) & (df['age'] <= age[1]) &
        (df['market_value'] >= budget[0]) & (df['market_value'] <= budget[1])
    ]

    # Check if any players were found
    if filtered_players.empty:
        st.write("No players found based on your criteria.")
    else:
        # Reset the index starting from 1
        filtered_players = filtered_players.reset_index(drop=True)
        filtered_players.index += 1  # Start index from 1 instead of 0

        st.subheader("Top 20 Players to Buy Based on Your Criteria")
        st.write(filtered_players[['player_name', 'market_value', 'age', skill1, skill2]].head(20))

        fig_skill_comparison = px.scatter(
            filtered_players,
            x=skill1,
            y=skill2,
            color='market_value',
            hover_name='player_name',
            title=f"Joint Plot : {skill1.replace('_', ' ').title()} vs {skill2.replace('_', ' ').title()}",
            labels={
                skill1: skill1.replace('_', ' ').title(),
                skill2: skill2.replace('_', ' ').title()
            },
            template='plotly_white',
            marginal_x = 'histogram',
            marginal_y = 'histogram'
        )
        fig_skill_comparison.update_traces(
                selector=dict(type='histogram'),
                marker_color='rgba(255, 0, 0, 0.8)',  # Set the histogram color to a semi-transparent blue.
                nbinsx = 10,
                nbinsy = 10
            )
        st.plotly_chart(fig_skill_comparison, use_container_width=True)























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

        # Scatter plot for International Reputation vs Market Value
        fig1 = px.scatter(df, x='international_reputation', y='market_value', hover_name='player_name', title="International Reputation vs Market Value", range_x=[1, 5])
        st.plotly_chart(fig1)

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






# Footer
st.sidebar.markdown("---")
st.sidebar.write("Created by Sivagugan Jayachandran")

