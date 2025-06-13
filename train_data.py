TRAIN_DATA = [
    ("John Doe is a Senior Data Scientist at TCS skilled in Python and Machine Learning.",
     {"entities": [(0, 8, "NAME"), (15, 35, "JOB_TITLE"), (39, 42, "ORG"), (60, 66, "SKILL"), (71, 89, "SKILL")]}),
    ("Jane Smith graduated from IIT Delhi and works as a Backend Developer at Amazon.",
     {"entities": [(0, 10, "NAME"), (25, 34, "EDUCATION"), (51, 68, "JOB_TITLE"), (72, 78, "ORG")]}),
    ("Contact: rahul@gmail.com, Phone: 9876543210",
     {"entities": [(9, 26, "EMAIL"), (35, 45, "PHONE")]}),
]