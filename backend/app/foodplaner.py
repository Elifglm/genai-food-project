import logging
import chromadb
from sentence_transformers import SentenceTransformer
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters
import re
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import os

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Telegram Bot Token
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# ChromaDB Settings
persist_path = "./chromadb"
chroma_client = chromadb.PersistentClient(persist_path)
collection = chroma_client.get_or_create_collection(name="user_preferences")

# Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Category Definitions
categories = {
    "fruit": ["apple", "banana", "orange", "kiwi", "melon", "strawberry", "pomegranate"],
    "vegetable": ["tomato", "cucumber", "carrot", "spinach", "lettuce", "broccoli", "leek"],
    "protein": ["chicken", "beef", "tofu", "egg", "salmon"],
    "dairy": ["milk", "cheese", "yogurt", "butter"],
    "grain": ["bread", "pasta", "rice", "oat"],
    "dish_type": ["soup", "salad", "pizza", "pasta", "burger", "dessert", "cake", "sandwich"]
}

IGNORE_WORDS = {"i", "but", "like", "dislike", "and", "or"}

# Precompute Embeddings for Categories
category_embeddings = {key: embedding_model.encode(values) for key, values in categories.items()}

def assign_category_with_similarity(item):
    """Determine the category of an item using cosine similarity."""
    item_embedding = embedding_model.encode([item])
    max_similarity = 0
    best_category = "other"

    for category, embeddings in category_embeddings.items():
        similarities = cosine_similarity(item_embedding, embeddings)
        max_category_similarity = np.max(similarities)
        if max_category_similarity > max_similarity:
            max_similarity = max_category_similarity
            best_category = category

    return best_category

def analyze_preferences_with_gpt(text):
    liked_items = []
    disliked_items = []
    dish_likes = []
    dish_dislikes = []
    liked_cuisines = []
    disliked_cuisines = []

    sentences = re.split(r"[.,;]", text.lower())  # Split into sentences
    for sentence in sentences:
        sentence = sentence.strip()

        # Check for "like" and "dislike" in the same sentence
        if "like" in sentence and "dislike" in sentence:
            like_part, dislike_part = sentence.split("dislike", 1)
            if "cuisine" in like_part:
                cuisines = [c for c in re.findall(r"\b\w+\b", like_part.split("like")[-1]) if c != "cuisine"]
                liked_cuisines.extend(cuisines)
            else:
                like_items = re.findall(r"\b\w+\b", like_part.split("like")[-1])
                liked_items.extend(like_items)

            if "cuisine" in dislike_part:
                cuisines = [c for c in re.findall(r"\b\w+\b", dislike_part) if c != "cuisine"]
                disliked_cuisines.extend(cuisines)
            else:
                dislike_items = re.findall(r"\b\w+\b", dislike_part)
                disliked_items.extend(dislike_items)
            continue  # Process next sentence

        # If the sentence only contains "like"
        if "like" in sentence:
            if "cuisine" in sentence:
                cuisines = [c for c in re.findall(r"\b\w+\b", sentence.split("like")[-1]) if c != "cuisine"]
                liked_cuisines.extend(cuisines)
            else:
                like_items = re.findall(r"\b\w+\b", sentence.split("like")[-1])
                liked_items.extend(like_items)

        # If the sentence only contains "dislike"
        if "dislike" in sentence:
            if "cuisine" in sentence:
                cuisines = [c for c in re.findall(r"\b\w+\b", sentence.split("dislike")[-1]) if c != "cuisine"]
                disliked_cuisines.extend(cuisines)
            else:
                dislike_items = re.findall(r"\b\w+\b", sentence.split("dislike")[-1])
                disliked_items.extend(dislike_items)

        # Handle vegan or vegetarian preferences
        if "i am vegan" in sentence:
            disliked_items.extend(categories["protein"] + categories["dairy"])  # Add both protein and dairy items
        elif "i am vegetarian" in sentence:
            disliked_items.extend(categories["protein"])  # Add only protein items

    # Filter ignored words
    liked_items = [item.strip() for item in liked_items if item.strip() and item not in IGNORE_WORDS]
    disliked_items = [item.strip() for item in disliked_items if item.strip() and item not in IGNORE_WORDS]

    return liked_items, disliked_items, dish_likes, dish_dislikes, liked_cuisines, disliked_cuisines

def generate_prompt(liked_items, disliked_items, dish_likes, dish_dislikes, liked_cuisines, disliked_cuisines):
    if not (liked_items or disliked_items or dish_likes or dish_dislikes or liked_cuisines or disliked_cuisines):
        return "Suggest me a general recipe."
    
    prompt = "Suggest me a recipe.\n"
    if liked_cuisines:
        prompt += f"Preferred cuisines: {', '.join(liked_cuisines)}.\n"
    if disliked_cuisines:
        prompt += f"Non-preferred cuisines: {', '.join(disliked_cuisines)}.\n"
    if dish_likes:
        prompt += f"Preferred dish types: {', '.join(dish_likes)}.\n"
    if dish_dislikes:
        prompt += f"Non-preferred dish types: {', '.join(dish_dislikes)}.\n"
    if liked_items:
        prompt += f"Include ingredients: {', '.join(liked_items)}.\n"
    if disliked_items:
        prompt += f"Exclude ingredients: {', '.join(disliked_items)}.\n"
    
    return prompt

def get_gpt_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in recipes."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Unable to fetch response from GPT: {str(e)}"

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.lower()
    user_id = str(update.effective_user.id)

    if "reset preferences" in user_text:
        try:
            collection.delete(where={"user_id": user_id})
            await update.message.reply_text("All your preferences have been reset.")
        except Exception as e:
            await update.message.reply_text(f"An error occurred while resetting preferences: {str(e)}")
        return

    if "suggest recipe" in user_text:
        preferences = collection.get()["metadatas"]
        liked_items = [meta["text"] for meta in preferences if meta["user_id"] == user_id and meta["like_status"] == "like" and meta["category"] != "dish_type"]
        disliked_items = [meta["text"] for meta in preferences if meta["user_id"] == user_id and meta["like_status"] == "dislike" and meta["category"] != "dish_type"]
        dish_likes = [meta["text"] for meta in preferences if meta["user_id"] == user_id and meta["like_status"] == "like" and meta["category"] == "dish_type"]
        dish_dislikes = [meta["text"] for meta in preferences if meta["user_id"] == user_id and meta["like_status"] == "dislike" and meta["category"] == "dish_type"]
        liked_cuisines = [meta["text"] for meta in preferences if meta["user_id"] == user_id and meta["like_status"] == "like" and meta["category"] == "cuisine"]
        disliked_cuisines = [meta["text"] for meta in preferences if meta["user_id"] == user_id and meta["like_status"] == "dislike" and meta["category"] == "cuisine"]

        prompt = generate_prompt(liked_items, disliked_items, dish_likes, dish_dislikes, liked_cuisines, disliked_cuisines)
        gpt_response = get_gpt_response(prompt)
        await update.message.reply_text(f"Recipe suggestion from GPT: {gpt_response}")
        return

    liked_items, disliked_items, dish_likes, dish_dislikes, liked_cuisines, disliked_cuisines = analyze_preferences_with_gpt(user_text)

    for item in liked_items:
        category = assign_category_with_similarity(item)
        collection.add(
            ids=[f"{user_id}_{item}"],
            documents=[item],
            metadatas=[{"user_id": user_id, "text": item, "like_status": "like", "category": category}]
        )
    for item in disliked_items:
        category = assign_category_with_similarity(item)
        collection.add(
            ids=[f"{user_id}_{item}"],
            documents=[item],
            metadatas=[{"user_id": user_id, "text": item, "like_status": "dislike", "category": category}]
        )
    for dish in dish_likes:
        collection.add(
            ids=[f"{user_id}_{dish}"],
            documents=[dish],
            metadatas=[{"user_id": user_id, "text": dish, "like_status": "like", "category": "dish_type"}]
        )
    for dish in dish_dislikes:
        collection.add(
            ids=[f"{user_id}_{dish}"],
            documents=[dish],
            metadatas=[{"user_id": user_id, "text": dish, "like_status": "dislike", "category": "dish_type"}]
        )
    for cuisine in liked_cuisines:
        collection.add(
            ids=[f"{user_id}cuisine{cuisine}"],
            documents=[cuisine],
            metadatas=[{"user_id": user_id, "text": cuisine, "like_status": "like", "category": "cuisine"}]
        )
    for cuisine in disliked_cuisines:
        collection.add(
            ids=[f"{user_id}cuisine{cuisine}"],
            documents=[cuisine],
            metadatas=[{"user_id": user_id, "text": cuisine, "like_status": "dislike", "category": "cuisine"}]
        )

    await update.message.reply_text("Your preferences have been saved.")

def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling()

if __name__ == "__main__":
    main()
