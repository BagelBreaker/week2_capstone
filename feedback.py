celebrities = [
      "Elon Musk", 
      "Sam A",
      "Leonardo DiCaprio", 
      "LeBron James", 
      "Chris Rock", 
      "Angelina Jolie", 
      "Serena Williams", 
      "Emma Watson", 
      "Jennifer Lawrence", 
      "Megan Fox"
]

feedback_history = []
def collect_feedback(predicted_person):
    print("--------------------------------")
    print("Facial Recognition Result")
    print("--------------------------------")

    print("The system predicts:", predicted_person)

response = input(
    "Is this prediction correct? (yes/no):"
(

if response.lower() == "yes":
        feedback = {
            "prediction": predicted_person,
            "correct": True,
            "actual_person": predicted_person
        }
        print("Prediction confirmed!")

else:
        print("\nWho is the correct person?")
        for i, person in enumerate(celebrities):
            print(i+1, "-", person)
        choice = int(
            input("Enter the correct number: ")
        )

        correct_person = celebrities[choice-1]

        feedback = {
            "prediction": predicted_person,
            "correct": False,
            "actual_person": correct_person
        }

        print(
            "Correction saved:",
            correct_person
        )

feedback_history.append(feedback)

print("\nFeedback recorded:")
print(feedback)

collect_feedback(predicted_name)
