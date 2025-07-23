import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# 1. Load and Clean Data
df = pd.read_csv("emails.csv")  # Replace with your actual CSV path

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]     # Remove stopwords
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)


# 2. Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['spam']

# 3. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# -------------------------
# 4. Evaluate (Optional)
# -------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc * 100:.2f}%")

# -------------------------
# 5. Take User Input
# -------------------------
user_input = input("\nEnter an email message to classify as Spam or Not Spam:\n")
clean_input = clean_text(user_input)
input_vector = vectorizer.transform([clean_input])
prediction = model.predict(input_vector)

# -------------------------
# 6. Output Result
# -------------------------
print("\nPrediction:", "🚫 Spam" if prediction[0] == 1 else "✅ Not Spam")
