import requests
import time
import json
import logging

API_KEY = "sk-1234567890abcdefghijklmnopqrstuvwxyz"

class AIModelAPI:
    def __init__(self):
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
    
    def get_completion(self, prompt, max_tokens=100):
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        
        # Próbujemy 3 razy w przypadku błędu
        for i in range(3):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    data=json.dumps(data)
                )
                
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    print(f"Błąd: {response.status_code}")
                    print(response.text)
                    time.sleep(1)
            except Exception as e:
                print(f"Wyjątek: {e}")
                time.sleep(1)
        
        return "Nie udało się uzyskać odpowiedzi."
    
    def get_embedding(self, text):
        data = {
            "model": "text-embedding-ada-002",
            "input": text
        }
        
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self.headers,
            data=json.dumps(data)
        )
        
        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            print(f"Błąd: {response.status_code}")
            print(response.text)
            return []
    
    def process_batch(self, prompts):
        results = []
        for p in prompts:
            results.append(self.get_completion(p))
        return results

# Przykład użycia
api = AIModelAPI()
response = api.get_completion("Wyjaśnij, czym jest uczenie maszynowe.")
print(response)

embeddings = api.get_embedding("Tekst do wektoryzacji")
print(f"Długość embeddingu: {len(embeddings)}")

batch_results = api.process_batch(["Pytanie 1", "Pytanie 2", "Pytanie 3"])
print(batch_results)
