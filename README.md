# GenAI Food Planner Project

## Purpose of the Project

The purpose of this project is to provide personalized recipe suggestions based on user preferences. These preferences include:
- Liked and disliked ingredients
- Dietary preferences (e.g., vegan or vegetarian)
- Preferred cuisines
- Desired dish types

By taking these preferences into account, the system helps users find recipes that align with their tastes and needs.

## Key Features and Approach

### 1. **Categorization and Embedding Usage**
- **Embedding Technology**: User inputs were converted into embeddings using the **SentenceTransformer** model.
- **Ingredient Categorization**: Ingredients were automatically categorized (e.g., fruit, vegetable, protein) using embeddings and **cosine similarity**. This approach ensures accurate classification of user inputs.

### 2. **GPT Integration**
- **Role of GPT**: OpenAI's GPT was utilized solely for generating recipe suggestions.
- **Prompt Engineering**: Custom prompts were designed to include user preferences, dietary restrictions, and other relevant information. These prompts guided GPT to create personalized and relevant recipe suggestions.

### 3. **Dynamic User Interaction**
- **Preference Management**: User preferences (liked/disliked ingredients, cuisines, and dish types) were stored persistently using **ChromaDB**.
- **Reset Preferences**: Users can reset all their preferences with a simple command to start fresh.
- **Vegan and Vegetarian Support**: Users specifying vegan or vegetarian preferences automatically exclude certain ingredient categories (e.g., dairy, meat).

## Tools and Technologies
- **Python**: The primary programming language for the project.
- **ChromaDB**: Database used for storing user preferences.
- **Sentence Transformers**: Used for embedding and categorizing user inputs.
- **Cosine Similarity**: Mathematical method for assigning ingredients to categories.
- **OpenAI GPT**: Generates recipe suggestions based on user preferences.
- **Telegram Bot API**: Provides an interface for user interaction.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env    # fill in OPENAI_API_KEY and TELEGRAM_BOT_TOKEN, then export them
cd backend/app
python foodplaner.py    # starts the Telegram bot
python flask_view.py    # optional: view stored preferences at http://localhost:5000
```

ChromaDB data is created locally under `chromadb/` on first run and is not committed.

## Outcome
This project successfully integrates AI into meal planning by:
- Generating personalized recipe suggestions.
- Creating a dynamic chatbot interface for user interaction.
- Supporting dietary restrictions and user-specific needs.

By combining embedding-based categorization and GPT-powered recipe generation, the system delivers a highly personalized food planning experience.
