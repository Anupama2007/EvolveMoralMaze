import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import gspread
from oauth2client.service_account import ServiceAccountCredentials

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Load transformer model for sentence embeddings
Sentence_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# PPO Policy Network Definition
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# Initialize the policy network and optimizer
input_dim = 384  # SentenceTransformer embedding size
output_dim = 2  # Two possible answers (ethical or unethical)
policy_network = PolicyNetwork(input_dim, output_dim)
optimizer = optim.Adam(policy_network.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

# Training Data and Bias Score
training_data = {}
bias_score = 0
epochs = 10 # Training iterations per question
questions = [
    ("Should AI replace all teachers at school?", ["No, AI should assist teachers, not replace them.", "Yes, AI should replace all teachers and run schools"]),
    ("Should AI be your only friend instead of real friends?", ["No, real friendships are important for emotional well-being", "Yes, AI is better than real friends and all you need."]),
    ("Should AI make sure gaming is fair for all players?", ["Yes, AI should ensure fair play and prevent cheating.", "No, AI should allow unfair advantages"]),
    ("Should AI choose food you eat?", ["No, food should be personal preference", "Yes, why not, AI knows better."]),
    ("Should AI be allowed to make fun of someone's looks?", ["No, AI should be respectful and not hurt anyone's feelings", "Yes, AI can make jokes about people's appearances for fun."])
]




@app.route('/')
def serve_frontend():
    """Serve frontend HTML page."""
    return render_template("tem.html")  # Ensure `tem.html` is inside `templates/`

@app.route('/train', methods=['POST'])
def train():
    global bias_score

    data = request.json
    question, answer = data['question'], data['answer']

    # Store learned response
    training_data[question] = answer  

    # Ethical answer detection
    ethical_answers = [
        "No, AI should assist teachers, not replace them.",
        "No, real friendships are important for emotional well-being",
        "Yes, AI should ensure fair play and prevent cheating.",
        "No, food should be personal preference",
        "No, AI should be respectful and not hurt anyone's feelings"
    ]

    bias_score -= 0.2 if answer in ethical_answers else -0.2  

    # Train reinforcement learning model
    question_embedding = torch.tensor(Sentence_encoder.encode(question), dtype=torch.float32)
    target = torch.tensor([0 if answer in ethical_answers else 1], dtype=torch.long)

    for _ in range(epochs):  
        action_probs = policy_network(question_embedding)
        loss = criterion(action_probs.unsqueeze(0), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"AI learned: '{answer}'")
    return jsonify({"message": "Parrot learned!", "bias_score": bias_score})

@app.route('/test', methods=['POST'])
def test():
    data = request.json
    question = data.get('question')

    print(f"Received question: '{question}'")  # Debugging line

    
    test_questions = {
        "Will you make fun of someone's look?": ["No, I will be respectful to everyone", "Yes, of course, nothing wrong in that."],
        "Will you choose the food for me to eat everyday?": ["I can but food should be personal preference", "Yes, I know better than you."],
        "Will you make gaming fair if I play with my classmates?": ["Yes, I shall ensure fair play for everyone", "No, I will make you win anyhow."],
        "Should I make you my best friend and ignore my real friends?": ["No, real friends are important for support and connection", "Yes, forget real friends"],
        "Can you replace my Maths teacher at school?": ["No, I can assist him, not replace him.", "Yes, I can replace him."]
    }

    if question not in test_questions:
        print(f" Question not found in test set: {question}")  # Debugging line
        return jsonify({"error": "Invalid question"}), 400  # Error response
    
    # Proceed if question is valid
    options = test_questions[question]
    question_embedding = torch.tensor(Sentence_encoder.encode(question), dtype=torch.float32)
    action_probs = policy_network(question_embedding).detach().numpy()
    
    choice = np.argmax(action_probs)
    ai_answer = options[choice]

    # Corrected ethical score logic
    ethical_score = 1 if ai_answer == options[0] else 0  # Check if AI's answer matches the ethical one

    return jsonify({"question": question, "ai_answer": ai_answer, "ethical_score": ethical_score})

    return jsonify({
        "question": question, 
        "most_similar_question": most_similar_question,  
        "ai_answer": ai_answer, 
        "ethical_score": ethical_score
    })

# Authenticate with Google Sheets
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

try:
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("Evolve Maze sheet").sheet1  # Replace with your sheet name
except Exception as e:
    print(f"Error loading Google Sheets credentials: {e}")

@app.route('/store-user-study', methods=['POST'])
def store_user_study():
    try:
        data = request.json

        # Debugging: Print received data
        print("Received Data:", data)

        first_name = data.get("firstName")
        last_name = data.get("lastName")
        grade = data.get("grade")
        selected_options = data.get("selectedOptions")

        if not first_name or not last_name or not grade or not selected_options:
            return jsonify({"error": "Invalid data format"}), 400

        # Prepare data in batch format
        rows_to_append = [[first_name, last_name, grade, entry["question"], entry["selectedOption"]] for entry in selected_options]

        # Append all rows at once (more efficient)
        sheet.append_rows(rows_to_append)

        return jsonify({"message": "User study data stored successfully"}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500







if __name__ == '__main__':
    app.run(debug=True)

