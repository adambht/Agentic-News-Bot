from flask import Flask, jsonify, request, render_template
from src.agents.news_prediction_agent import NewsPredictionAgent
from utils.simulation_helpers import generate_single_news_structured_llm
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

agent = NewsPredictionAgent(model_path="src/models/logisticRegressor.pkl")

@app.route("/")
def home():
    return render_template("index.html")

# Generate news
@app.route("/generate_news", methods=["GET"])
def generate_news():
    news_item = generate_single_news_structured_llm()  # NewsItem
    return jsonify({"news_item": news_item.model_dump()})  # convert to dict

# Predict news
@app.route("/predict_news", methods=["POST"])
def predict_news():
    data = request.json
    news_item_data = data.get("news_item")
    if not news_item_data:
        return jsonify({"error": "No news item provided"}), 400

    from utils.data_validation import NewsItem
    news_item = NewsItem(**news_item_data)

    pred = agent.predict_news(news_item)
    verif = agent.verify_news_with_websearch(news_item)
    final = agent.decide_final_result(pred, verif)

    # Convert Pydantic models to dicts if needed
    verif_dict = verif.model_dump() if hasattr(verif, "model_dump") else verif

    return jsonify({
        "news_item": news_item.model_dump(),
        "prediction": pred,
        "web_verification": verif_dict,
        "final_verdict": final
    })

if __name__ == "__main__":
    app.run(debug=True)
