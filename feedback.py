def collect_feedback(prediction, face_vector, db=None, labels=None, metadata=None):
    labels = labels or (db.labels() if db is not None else [])
    predicted = prediction["prediction"] if isinstance(prediction, dict) else prediction

    print("--------------------------------")
    print("Facial Recognition Result")
    print("--------------------------------")
    print("System prediction:", predicted)

    feedback = input("Correct? (yes/no/unknown/skip): ").strip().lower()

    if feedback in ["yes", "y"]:
        if db is not None and predicted != "unknown":
            db.add(face_vector, predicted, metadata)
        return {"action": "confirmed", "person": predicted}

    if feedback in ["no", "n"]:
        if not labels:
            print("No known labels available.")
            return {"action": "skipped"}

        for i, person in enumerate(labels, 1):
            print(i, "-", person)

        try:
            choice = int(input("Correct person's number: "))
            correct = labels[choice - 1]
        except (ValueError, IndexError):
            print("Invalid choice.")
            return {"action": "skipped"}

        if db is not None:
            db.add(face_vector, correct, metadata)

        print("Updated profile:", correct)
        return {"action": "corrected", "person": correct, "old_prediction": predicted}

    if feedback in ["unknown", "u"]:
        new_name = input("New profile name: ").strip()

        if not new_name:
            print("No name entered.")
            return {"action": "skipped"}

        if db is not None:
            db.add(face_vector, new_name, metadata)

        print("New profile created for:", new_name)
        return {"action": "new_profile", "person": new_name}

    return {"action": "skipped"}
