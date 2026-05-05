# recommendations.py
SYMPTOM_DB = {
    'ants': {'Mild': ['Localized itching', 'Small red bump'],
             'Moderate': ['Swelling around bite', 'Painful itching'],
             'Severe': ['Multiple bites', 'Allergic reaction possible']},
    'bed_bugs': {'Mild': ['Itchy welts', 'Linear pattern'],
                 'Moderate': ['Intense itching', 'Blister-like lesions'],
                 'Severe': ['Secondary infection', 'Anxiety/insomnia']},
    'Bees': {'Mild': ['Sharp pain', 'Redness at sting site'],
             'Moderate': ['Moderate swelling', 'Itching beyond site'],
             'Severe': ['Difficulty breathing', 'Swollen face/throat']},
    'chiggers': {'Mild': ['Severe itching', 'Red pimples'],
                 'Moderate': ['Papules', 'Skin thickening'],
                 'Severe': ['Fever', 'Secondary infection']},
    'fleas': {'Mild': ['Small red dots', 'Itching around ankles'],
              'Moderate': ['Hives', 'Swollen bumps'],
              'Severe': ['Infection from scratching', 'Allergic dermatitis']},
    'mosquito': {'Mild': ['Itchy bump', 'Warm to touch'],
                 'Moderate': ['Large swelling', 'Persistent itch'],
                 'Severe': ['Signs of West Nile/Dengue (fever, headache)']},
    'no_bites': {'Mild': ['No typical bite marks', 'Possible rash'],
                 'Moderate': ['Irritation unknown cause'],
                 'Severe': ['Persistent lesion, see dermatologist']},
    'spiders': {'Mild': ['Sharp sting', 'Red ring'],
                'Moderate': ['Blisters', 'Muscle pain'],
                'Severe': ['Necrotic tissue', 'Systemic symptoms']},
    'tick': {'Mild': ['Painless bite', 'Red spot'],
             'Moderate': ['Bullseye rash (Lyme)', 'Fatigue'],
             'Severe': ['Joint pain', 'Neurological issues']}
}

FIRST_AID = {
    'ants': {"Mild": "Wash with soap. Apply ice. Use calamine lotion.",
             "Moderate": "Oral antihistamine (cetirizine). Hydrocortisone cream.",
             "Severe": "Seek medical care if allergic reaction."},
    'bed_bugs': {"Mild": "Antiseptic wash. Anti-itch cream.",
                 "Moderate": "Cold compress. Oral antihistamine.",
                 "Severe": "Consult dermatologist; possible steroids."},
    'Bees': {"Mild": "Remove stinger by scraping. Apply ice.",
             "Moderate": "Pain reliever (ibuprofen). Antihistamine.",
             "Severe": "Emergency if anaphylaxis."},
    'chiggers': {"Mild": "Cool bath. Calamine lotion.",
                 "Moderate": "Topical steroid. Avoid scratching.",
                 "Severe": "Medical attention if fever."},
    'fleas': {"Mild": "Wash with antiseptic. Baking soda paste.",
              "Moderate": "Oral antihistamines. Treat pets.",
              "Severe": "Physician for possible antibiotics."},
    'mosquito': {"Mild": "Ice pack. Calamine lotion.",
                 "Moderate": "Oral antihistamine. Pain reliever.",
                 "Severe": "Monitor for systemic symptoms → doctor."},
    'no_bites': {"Mild": "Moisturize. Observe.",
                 "Moderate": "OTC hydrocortisone.",
                 "Severe": "Dermatologist evaluation."},
    'spiders': {"Mild": "Clean with soap. Antibiotic ointment.",
                "Moderate": "Elevate. Watch for necrosis.",
                "Severe": "Emergency for venomous spiders."},
    'tick': {"Mild": "Remove tick with tweezers. Disinfect.",
             "Moderate": "Monitor for rash. See doctor for prophylaxis.",
             "Severe": "Immediate care if neurological symptoms."}
}

def get_recommendations(insect: str, severity: str) -> dict:
    symptoms = SYMPTOM_DB.get(insect, {}).get(severity, ["Observe area"])
    first_aid = FIRST_AID.get(insect, {}).get(severity, "Keep clean and monitor.")
    when_see_doctor = "Immediately" if severity == "Severe" else "If symptoms worsen or fever develops"
    return {
        "symptoms": symptoms,
        "first_aid": first_aid,
        "doctor_advice": when_see_doctor
    }