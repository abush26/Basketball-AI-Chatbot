
import streamlit as st
import requests
import json
import re
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os


API_BASE_URL = "https://www.balldontlie.io/api/v1"
CACHE_DURATION = 300  # 5 minutes cache

# Basketball rules and static information
BASKETBALL_KNOWLEDGE = {
    "rules": {
        "basic_rules": "Basketball is played by two teams of five players. The objective is to score by shooting the ball through the opponent's hoop. A game consists of four quarters, each lasting 12 minutes in the NBA.",
        "scoring": "Field goals are worth 2 points, or 3 points if shot from beyond the three-point line. Free throws are worth 1 point each.",
        "fouls": "Players can commit personal fouls, technical fouls, and flagrant fouls. Six personal fouls result in fouling out of the game.",
        "shot_clock": "Teams have 24 seconds to attempt a shot in the NBA, 30 seconds in college basketball.",
        "overtime": "If the game is tied after regulation, overtime periods of 5 minutes are played until a winner is determined."
    },
    "positions": {
        "point_guard": "The floor general who runs plays and distributes the ball. Usually the shortest player on the team.",
        "shooting_guard": "Primarily responsible for shooting and scoring. Often the team's best perimeter shooter.",
        "small_forward": "Versatile player who can play inside and outside. Often a good all-around player.",
        "power_forward": "Typically plays near the basket, strong rebounder and inside scorer.",
        "center": "Usually the tallest player, plays close to the basket, responsible for rebounds and inside defense."
    },
    "terms": {
        "assist": "A pass that directly leads to a teammate scoring a basket.",
        "rebound": "Gaining possession of the ball after a missed shot.",
        "steal": "Taking the ball away from the opposing team.",
        "block": "Deflecting an opponent's shot attempt.",
        "turnover": "Losing possession of the ball to the opposing team.",
        "double_double": "Recording double digits in two statistical categories in a single game.",
        "triple_double": "Recording double digits in three statistical categories in a single game."
    }
}



class BasketballNLP:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.intent_patterns = {
            'player_stats': [
                'stats for', 'player stats', 'how is', 'performance of', 'statistics',
                'points per game', 'rebounds', 'assists', 'how many points',
                'who is', 'tell me about', 'about player', 'player info',
                'information about', 'profile of'
            ],
            'team_info': [
                'team stats', 'team record', 'standings', 'how is team', 'team performance',
                'wins', 'losses', 'ranking', 'about team', 'team info'
            ],
            'game_results': [
                'game result', 'score', 'who won', 'final score', 'game score',
                'match result', 'latest game', 'recent games'
            ],
            'rules': [
                'rules', 'how to play', 'basketball rules', 'regulation', 'foul',
                'what is', 'define', 'explain', 'meaning of'
            ],
            'schedule': [
                'next game', 'when is', 'schedule', 'upcoming games', 'fixture'
            ]
        }
        self.setup_classifier()

    def setup_classifier(self):
        # Create training data for intent classification
        training_texts = []
        training_labels = []
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                training_texts.append(pattern)
                training_labels.append(intent)
        
        # Vectorize the training data
        self.X_train = self.vectorizer.fit_transform(training_texts)
        self.y_train = training_labels

    def classify_intent(self, text: str) -> str:
        """Classify the intent of user input using cosine similarity"""
        text_vector = self.vectorizer.transform([text.lower()])
        similarities = cosine_similarity(text_vector, self.X_train).flatten()
        
        # Check if the text contains player names - if so, likely player_stats intent
        if self._contains_player_name(text):
            return 'player_stats'
        
        if similarities.max() > 0.1:  # Threshold for classification
            best_match_idx = similarities.argmax()
            return self.y_train[best_match_idx]
        
        return 'general'

    def _contains_player_name(self, text: str) -> bool:
        """Check if text contains a player name pattern"""
        # Look for common player name patterns
        player_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        players = re.findall(player_pattern, text)
        
        # Also check for common basketball player names
        famous_players = [
            'lebron james', 'stephen curry', 'kevin durant', 'giannis antetokounmpo',
            'luka doncic', 'jayson tatum', 'joel embiid', 'nikola jokic',
            'damian lillard', 'james harden', 'russell westbrook', 'chris paul',
            'kobe bryant', 'michael jordan', 'shaq', 'shaquille oneal'
        ]
        
        text_lower = text.lower()
        for player in famous_players:
            if player in text_lower:
                return True
        
        return len(players) > 0

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract relevant entities from the text"""
        entities = {}
        
        # Extract player names (improved pattern matching)
        player_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        players = re.findall(player_pattern, text)
        
        # Check for famous players by name
        famous_players = {
            'lebron james': 'LeBron James',
            'lebron': 'LeBron James',
            'stephen curry': 'Stephen Curry',
            'steph curry': 'Stephen Curry',
            'curry': 'Stephen Curry',
            'kevin durant': 'Kevin Durant',
            'kd': 'Kevin Durant',
            'giannis': 'Giannis Antetokounmpo',
            'giannis antetokounmpo': 'Giannis Antetokounmpo',
            'luka doncic': 'Luka Doncic',
            'luka': 'Luka Doncic',
            'jayson tatum': 'Jayson Tatum',
            'tatum': 'Jayson Tatum',
            'joel embiid': 'Joel Embiid',
            'embiid': 'Joel Embiid',
            'nikola jokic': 'Nikola Jokic',
            'jokic': 'Nikola Jokic'
        }
        
        text_lower = text.lower()
        for key, full_name in famous_players.items():
            if key in text_lower:
                entities['player'] = full_name
                break
        
        if 'player' not in entities and players:
            entities['player'] = players[0]
        
        # Extract team names
        team_names = [
            'Lakers', 'Warriors', 'Celtics', 'Heat', 'Bulls', 'Knicks',
            'Nets', 'Sixers', 'Bucks', 'Raptors', 'Magic', 'Hawks',
            'Hornets', 'Pistons', 'Pacers', 'Cavaliers', 'Wizards',
            'Thunder', 'Trail Blazers', 'Jazz', 'Nuggets', 'Timberwolves',
            'Clippers', 'Kings', 'Suns', 'Mavericks', 'Rockets',
            'Grizzlies', 'Pelicans', 'Spurs'
        ]
        
        for team in team_names:
            if team.lower() in text.lower():
                entities['team'] = team
                break
        
        # Extract numbers (for stats, scores, etc.)
        numbers = re.findall(r'\d+', text)
        if numbers:
            entities['numbers'] = [int(n) for n in numbers]
        
        return entities



class BasketballAPI:
    def __init__(self):
        self.base_url = API_BASE_URL
        self.cache = {}
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with error handling and caching"""
        cache_key = f"{endpoint}_{str(params)}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < CACHE_DURATION:
                return cached_data
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.cache[cache_key] = (data, time.time())
                return data
            else:
                st.error(f"API Error: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")
            return None

    def search_players(self, player_name: str) -> List[Dict]:
        """Search for players by name"""
        data = self._make_request("players", {"search": player_name})
        return data.get('data', []) if data else []

    def get_player_stats(self, player_id: int, season: int = 2023) -> Dict:
        """Get player season averages"""
        data = self._make_request("season_averages", {
            "player_ids[]": player_id,
            "season": season
        })
        return data.get('data', [{}])[0] if data and data.get('data') else {}

    def get_teams(self) -> List[Dict]:
        """Get all NBA teams"""
        data = self._make_request("teams")
        return data.get('data', []) if data else []

    def get_recent_games(self, team_id: int = None, limit: int = 10) -> List[Dict]:
        """Get recent games"""
        params = {"per_page": limit}
        if team_id:
            params["team_ids[]"] = team_id
            
        data = self._make_request("games", params)
        return data.get('data', []) if data else []



class BasketballChatbot:
    def __init__(self):
        self.nlp = BasketballNLP()
        self.api = BasketballAPI()
        
    def generate_response(self, user_input: str) -> str:
        """Generate response based on user input"""
        intent = self.nlp.classify_intent(user_input)
        entities = self.nlp.extract_entities(user_input)
        
        if intent == 'player_stats':
            return self._handle_player_stats(entities, user_input)
        elif intent == 'team_info':
            return self._handle_team_info(entities, user_input)
        elif intent == 'game_results':
            return self._handle_game_results(entities, user_input)
        elif intent == 'rules':
            return self._handle_rules_query(user_input)
        elif intent == 'schedule':
            return self._handle_schedule_query(entities, user_input)
        else:
            return self._handle_general_query(user_input)

    def _handle_player_stats(self, entities: Dict, user_input: str) -> str:
        """Handle player statistics queries"""
        if 'player' not in entities:
            return "Please specify a player name. For example: 'Show me LeBron James stats' or 'Who is Stephen Curry?'"
        
        player_name = entities['player']
        
        # Provide basic info for well-known players even if API fails
        player_info = self._get_basic_player_info(player_name)
        
        players = self.api.search_players(player_name)
        
        if not players:
            if player_info:
                return player_info
            return f"Sorry, I couldn't find any player named '{player_name}'. Please check the spelling."
        
        player = players[0]  # Take the first match
        stats = self.api.get_player_stats(player['id'])
        
        response = f"**{player['first_name']} {player['last_name']}**\n\n"
        
        # Add basic player info
        if player_info:
            response += f"{player_info}\n\n"
        
        response += f"**Team:** {player['team']['full_name']}\n"
        response += f"**Position:** {player.get('position', 'N/A')}\n"
        response += f"**Height:** {player.get('height_feet', 'N/A')}'{player.get('height_inches', '')}\"\n"
        response += f"**Weight:** {player.get('weight_pounds', 'N/A')} lbs\n\n"
        
        if stats:
            response += f"**2023 Season Averages:**\n"
            response += f"• Points: {stats.get('pts', 'N/A')} per game\n"
            response += f"• Rebounds: {stats.get('reb', 'N/A')} per game\n"
            response += f"• Assists: {stats.get('ast', 'N/A')} per game\n"
            response += f"• Field Goal %: {stats.get('fg_pct', 'N/A')}\n"
            response += f"• Games Played: {stats.get('games_played', 'N/A')}\n"
        else:
            response += "Current season stats not available, but this player is active in the NBA."
        
        return response

    def _get_basic_player_info(self, player_name: str) -> str:
        """Get basic information about famous players"""
        player_info = {
            'LeBron James': "LeBron James, often called 'The King', is one of the greatest basketball players of all time. He's a 4-time NBA champion, 4-time Finals MVP, and 4-time regular season MVP. Known for his incredible versatility, basketball IQ, and longevity.",
            'Stephen Curry': "Stephen Curry revolutionized basketball with his incredible three-point shooting. He's a 4-time NBA champion, 2-time MVP, and holds numerous three-point records. Often considered the greatest shooter in NBA history.",
            'Kevin Durant': "Kevin Durant is a prolific scorer and one of the most talented players in NBA history. He's a 2-time NBA champion, 2-time Finals MVP, and former regular season MVP. Known for his incredible scoring ability and versatility.",
            'Giannis Antetokounmpo': "Giannis Antetokounmpo, 'The Greek Freak', is known for his incredible athleticism and versatility. He's an NBA champion, Finals MVP, and 2-time regular season MVP. One of the most dominant players in the modern era.",
            'Nikola Jokic': "Nikola Jokic is a Serbian center known for his exceptional passing and basketball IQ. He's a 2-time NBA MVP and has revolutionized the center position with his playmaking abilities.",
            'Joel Embiid': "Joel Embiid is a dominant center from Cameroon, known for his scoring, rebounding, and defensive abilities. He's a former NBA MVP and one of the league's most skilled big men."
        }
        
        return player_info.get(player_name, "")

    def _handle_team_info(self, entities: Dict, user_input: str) -> str:
        """Handle team information queries"""
        teams = self.api.get_teams()
        
        if 'team' in entities:
            team_name = entities['team']
            team = next((t for t in teams if team_name.lower() in t['full_name'].lower()), None)
            
            if team:
                response = f"**{team['full_name']}**\n\n"
                response += f"• Conference: {team['conference']}\n"
                response += f"• Division: {team['division']}\n"
                response += f"• City: {team['city']}\n"
                response += f"• Abbreviation: {team['abbreviation']}\n"
                
                # Get recent games for this team
                recent_games = self.api.get_recent_games(team['id'], 5)
                if recent_games:
                    response += f"\n**Recent Games:**\n"
                    for game in recent_games[:3]:
                        home_team = game['home_team']['full_name']
                        visitor_team = game['visitor_team']['full_name']
                        response += f"• {visitor_team} @ {home_team}\n"
                
                return response
            else:
                return f"Sorry, I couldn't find information for '{team_name}' team."
        else:
            return "Please specify which team you'd like information about."

    def _handle_game_results(self, entities: Dict, user_input: str) -> str:
        """Handle game results queries"""
        recent_games = self.api.get_recent_games(limit=10)
        
        if not recent_games:
            return "Sorry, I couldn't retrieve recent game results at the moment."
        
        response = "**Recent NBA Games:**\n\n"
        
        for game in recent_games[:5]:
            home_team = game['home_team']['abbreviation']
            visitor_team = game['visitor_team']['abbreviation']
            home_score = game['home_team_score']
            visitor_score = game['visitor_team_score']
            date = game['date'][:10]  # Format date
            
            if home_score and visitor_score:
                winner = home_team if home_score > visitor_score else visitor_team
                response += f"• {visitor_team} {visitor_score} - {home_score} {home_team} ({date}) 🏆 {winner}\n"
            else:
                response += f"• {visitor_team} @ {home_team} ({date}) - Scheduled\n"
        
        return response

    def _handle_rules_query(self, user_input: str) -> str:
        """Handle basketball rules and definitions"""
        user_input_lower = user_input.lower()
        
        # Check for specific rule categories
        for category, rules in BASKETBALL_KNOWLEDGE["rules"].items():
            if category.replace("_", " ") in user_input_lower:
                return f"**{category.replace('_', ' ').title()}:**\n{rules}"
        
        # Check for position definitions
        for position, definition in BASKETBALL_KNOWLEDGE["positions"].items():
            if position.replace("_", " ") in user_input_lower:
                return f"**{position.replace('_', ' ').title()}:**\n{definition}"
        
        # Check for basketball terms
        for term, definition in BASKETBALL_KNOWLEDGE["terms"].items():
            if term in user_input_lower:
                return f"**{term.title()}:**\n{definition}"
        
        # General rules response
        return """**Basic Basketball Rules:**
        
Basketball is played by two teams of five players each. The objective is to score points by shooting the ball through the opponent's hoop while preventing them from scoring in your hoop.

**Key Rules:**
• Game duration: 4 quarters of 12 minutes each (NBA)
• Shot clock: 24 seconds to attempt a shot
• Scoring: 2 points for field goals, 3 points from beyond the arc, 1 point for free throws
• Fouls: 6 personal fouls result in fouling out

Ask me about specific rules, positions, or basketball terms for more detailed information!"""

    def _handle_schedule_query(self, entities: Dict, user_input: str) -> str:
        """Handle schedule and upcoming games queries"""
        # This would typically require a different API endpoint for schedules
        return "I can show you recent games, but I don't have access to upcoming game schedules at the moment. Try asking about recent game results or player stats!"

    def _handle_general_query(self, user_input: str) -> str:
        """Handle general basketball queries"""
        return """I'm your basketball AI assistant! I can help you with:

🏀 **Player Statistics** - "Show me LeBron James stats" or "Who is Stephen Curry?"
🏆 **Team Information** - "Tell me about the Lakers"
🎯 **Game Results** - "Recent NBA games" or "Latest scores"
📖 **Rules & Definitions** - "What is a triple double?" or "Basketball rules"

What would you like to know about basketball?"""



def main():
    st.set_page_config(
        page_title="🏀 Basketball AI Chatbot",
        page_icon="🏀",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #ff6b35;
        margin-bottom: 30px;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #f8f9fa;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: right;
    }
    .bot-message {
        background-color: #28a745;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = BasketballChatbot()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Header
    st.markdown("<h1 class='main-header'>🏀 Basketball AI Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("Ask me anything about basketball - player stats, team info, game results, or rules!")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.subheader("💬 Chat")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
                st.markdown(f"**You:** {user_msg}")
                st.markdown(f"**🏀 Bot:** {bot_msg}")
                st.divider()
        
        # User input
        user_input = st.text_input("Type your basketball question here:", key="user_input")
        
        col_send, col_clear = st.columns([1, 1])
        
        with col_send:
            if st.button("Send 🚀", type="primary"):
                if user_input.strip():
                    with st.spinner("Thinking..."):
                        response = st.session_state.chatbot.generate_response(user_input)
                        st.session_state.chat_history.append((user_input, response))
                    st.rerun()
        
        with col_clear:
            if st.button("Clear Chat 🗑️"):
                st.session_state.chat_history = []
                st.rerun()
    
    with col2:
        # Sidebar with examples and info
        st.subheader("💡 Example Questions")
        
        example_questions = [
            "Who is LeBron James?",
            "Show me Stephen Curry stats",
            "Tell me about the Lakers",
            "Recent NBA games",
            "What is a triple double?",
            "Basketball basic rules",
            "Who is Giannis?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{question}"):
                with st.spinner("Processing..."):
                    response = st.session_state.chatbot.generate_response(question)
                    st.session_state.chat_history.append((question, response))
                st.rerun()
        
        st.subheader("ℹ️ Features")
        st.info("""
        **What I can do:**
        - 📊 Live player statistics
        - 🏆 Team information
        - 🎯 Recent game results
        - 📖 Basketball rules & terms
        - 🤖 AI-powered understanding
        - 👤 Player profiles & info
        
        **Data Source:** NBA API
        **Updated:** Real-time
        """)
        
        st.subheader("🔧 Technical Info")
        st.success("""
        **AI Features:**
        - Neural Network-based NLP
        - Intent Classification
        - Entity Extraction
        - Live Data Integration
        - Caching for Performance
        - Improved Player Recognition
        """)

if __name__ == "__main__":
    main()
