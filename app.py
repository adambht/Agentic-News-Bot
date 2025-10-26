
from src.agents.news_prediction_agent import NewsPredictionAgent
from utils.simulation_helpers import generate_single_news_structured_llm
from dotenv import load_dotenv

load_dotenv()

agent = NewsPredictionAgent(model_path="src/models/logisticRegressor.pkl")

news_item = generate_single_news_structured_llm()
print("\n🗞️ === Generated News ===")
print(news_item)
pred = agent.predict_news(news_item)
print("\n=== Prediction Result ===")
print(pred)
verif = agent.verify_news_with_websearch(news_item)
print("\n=== Web Verification ===")
print(verif)
final = agent.decide_final_result(pred, verif)
print("\n=== Final Verdict ===")

print(final)