celebrities = [
    "Elon Musk",
    "Sam A",
    "Leonardo DiCaprio",
    "LeBron James",
    "Chris Rock",
    "Serena Williams",
    "Angelina Jolie",
    "Emma Watson",
    "Jennifer Lawrence",
    "Megan Fox"
]

# Store feedback history
feedback_history = []

# Store new unknown
new_profiles = []

def collect_feedback(predicted_person, face_vector):

    print("--------------------------------")
    print("Facial Recognition")
    print("--------------------------------")

    print("Prediction:", predicted_person)

    feedback = input(
        "Is this person correctly predicted? (yes/no/unknown): "
    )

    # Correct prediction
    if feedback.lower() == "yes":

        profile_update = {
            "person": predicted_person,
            "vector_added": face_vector,
            "confirmed": True
        }

        feedback_history.append(profile_update)
        print(
            "Confirmed.",
            predicted_person,
            "profile updated."
        )

    # Wrong prediction

    elif feedback.lower() == "no":
        print("\nWho is the correct person?")
        for i, person in enumerate(celebrities):
            print(i+1, "-", person)

        choice = int(
            input("Enter correct person's number: ")
        )

        correct_person = celebrities[choice-1]

        profile_update = {
            "incorrect_prediction": predicted_person,
            "actual_person": correct_person,
            "vector_added": face_vector
        }

        feedback_history.append(profile_update)

        print(
            "Updated profile:",
            correct_person
        )

    # Unknown Person

    elif feedback.lower() == "unknown":
        print("\nCreating a new profile...")

        new_name = input(
            "Enter person's name: "
        )

        new_profile = {
            "name": new_name,
            "vector": face_vector
        }

        new_profiles.append(new_profile)

        print(
            "New profile created for:",
            new_name
        )

    else:
        print("Invalid image.")

collect_feedback(
    predicted_name,
    example_vector
)
