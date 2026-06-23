import joblib

# Load saved model
model = joblib.load("models/spam_classifier.pkl")

print("Spam Email Classifier")
print("Type 'exit' to quit\n")

while True:
    message = input("Enter Email/SMS: ")

    if message.lower() == "exit":
        print("Goodbye!")
        break

    prediction = model.predict([message])[0]

    print(f"Prediction: {prediction}\n")