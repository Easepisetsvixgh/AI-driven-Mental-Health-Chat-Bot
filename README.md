# AI-driven-Mental-Health-Chat-Bot
開発AIを利用した精神保健チャットボットは、ユーザーに感情的なサポートを提供し、必要に応じて専門家にリファーします。
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

class MentalHealthChatBot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.chat_history_ids = None

    def get_response(self, user_input):
        # Encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')

        # Append the new user input tokens to the chat history
        bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1) if self.chat_history_ids is not None else new_user_input_ids

        # Generate a response from the model
        self.chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)

        # Decode and return the model's response
        return self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    def analyze_sentiment(self, message):
        return self.sentiment_pipeline(message)[0]

    def suggest_help(self, sentiment):
        if sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.75:
            return True
        return False

    def chat(self):
        print("Hello! I'm here to provide emotional support. How can I help you today?")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "quit":
                print("Goodbye! Take care!")
                break

            sentiment = self.analyze_sentiment(user_input)
            if self.suggest_help(sentiment):
                print("It seems like you're going through a tough time. While I'm here to support you, speaking to a professional could be really beneficial.")
                continue

            response = self.get_response(user_input)
            print("Bot:", response)

if __name__ == "__main__":
    chatbot = MentalHealthChatBot()
    chatbot.chat()
