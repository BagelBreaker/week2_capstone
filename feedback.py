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

def collect_feedback(predicted_person, descriptor_vector, pkl_db, db_path="face_db.pkl"):

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
            "vector_added": descriptor_vector,
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
        pkl_db.add(descriptor_vector, correct_person)
        pkl_db.save(db_path)
        profile_update = {
            "incorrect_prediction": predicted_person,
            "actual_person": correct_person,
            "vector_added": descriptor_vector
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
            "vector": descriptor_vector
        }
        pkl_db.add(descriptor_vector, new_name)
        pkl_db.save(db_path)
        new_profiles.append(new_profile)

        print(
            "New profile created for:",
            new_name
        )

    else:
        print("Invalid.")

# collect_feedback(
#     predicted_name,
#     descriptor_vector
# )
