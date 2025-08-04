import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --------- LANGUAGE SUPPORT ---------
LANGUAGES = {
    "English": {
        "title": "🌿 Plant Disease Identifier",
        "upload": "Upload a plant leaf image",
        "predict": "Predict Disease",
        "result": "Predicted Disease:",
        "copy_label": "Predicted Disease (you can copy):",
        "page_select": "Select Page"
    },
    "தமிழ்": {
        "title": "🌿 தாவர நோய் அடையாளம் காண்பவர்",
        "upload": "தாவர இலை படத்தையெற்றவும்",
        "predict": "நோயையை கணிக்கவும்",
        "result": "கணிக்கபட நோய்:",
        "copy_label": "கணிக்கபட நோய் (நகலெறுக்கலாம்):",
        "page_select": "பக்கத்தைத் தேர்ந்தெடுக்கவும்"
    },
    "हिन्दी": {
        "title": "🌿 पौधों के रोग पहचानें",
        "upload": "पौधे की पत्ती की छवि अपलोड करें",
        "predict": "रोग की भविष्यावानी करें",
        "result": "अनुमानित रोग:",
        "copy_label": "अनुमानित रोग (कापी कर सकते हैं):",
        "page_select": "पृष्ठ चुनें"
    }
}

# --------- TRANSLATIONS ---------
TRANSLATIONS = {
    "Pepper__bell___Bacterial_spot": {"தமிழ்": "மிளகு___பாக்டீரியா புள்ளி", "हिन्दी": "मिर्च___बैक्टीरियल स्पॉट"},
    "Pepper__bell___healthy": {"தமிழ்": "மிளகு___ஆரோக்கியம்", "हिन्दी": "मिर्च___स्वस्थ"},
    "Potato___Early_blight": {"தமிழ்": "உருளைக்கிழங்கு___ஆரம்பம் நிறம்", "हिन्दी": "आलू___अर्ली ब्लाइट"},
    "Potato___Late_blight": {"தமிழ்": "உருளைக்கிழங்கு___தாமத நிறம்", "हिन्दी": "आलू___लेट ब्लाइट"},
    "Potato___healthy": {"தமிழ்": "உருளைக்கிழங்கு___ஆரோக்கியம்", "हिन्दी": "आलू___स्वस्थ"},
    "Tomato_Bacterial_spot": {"தமிழ்": "தக்காளி___பாக்டீரியா புள்ளி", "हिन्दी": "टमाटर___बैक्टीरियल स्पॉट"},
    "Tomato_Early_blight": {"தமிழ்": "தக்காளி___ஆரம்ப நிறம்", "हिन्दी": "टमाटर___अर्ली ब्लाइट"},
    "Tomato_Late_blight": {"தமிழ்": "தக்காளி___தாமத நிறம்", "हिन्दी": "टमाटर___लेट ब्लाइट"},
    "Tomato_Leaf_Mold": {"தமிழ்": "தக்காளி___இலை பூஞ்சை", "हिन्दी": "टमाटर___लीफ मोल्ड"},
    "Tomato_Septoria_leaf_spot": {"தமிழ்": "தக்காளி___செப்டோரியா இலை புள்ளி", "हिन्दी": "टमाटर___सेप्टोरिया लीफ स्पॉट"},
    "Tomato_Spider_mites_Two_spotted_spider_mite": {"தமிழ்": "தக்காளி___இரண்டு புள்ளி சில்லி பூச்சி", "हिन्दी": "टमाटर___स्पाइडर माइट्स"},
    "Tomato__Target_Spot": {"தமிழ்": "தக்காளி___இலக்கு புள்ளி", "हिन्दी": "टमाटर___टारगेट स्पॉट"},
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {"தமிழ்": "தக்காளி___மஞ்சள் இலை சுருக்கம் வைரஸ்", "हिन्दी": "टमाटर___पीलापन पत्ता कर्ल वायरस"},
    "Tomato__Tomato_mosaic_virus": {"தமிழ்": "தக்காளி___மொசாயிக் வைரஸ்", "हिन्दी": "टमाटर___मोज़ेक वायरस"},
    "Tomato_healthy": {"தமிழ்": "தக்காளி___ஆரோக்கியம்", "हिन्दी": "टमाटर___स्वस्थ"}
}

# --------- CLASS NAMES ---------
class_names = list(TRANSLATIONS.keys())

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(torch.load("plant_disease_model.pth", map_location='cpu'))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --------- UI ---------
st.set_page_config(page_title="Plant Disease App", layout="centered")
st.sidebar.title("🌐 Language / மொழி / भाषा")
language = st.sidebar.radio("Choose Language", list(LANGUAGES.keys()))
txt = LANGUAGES[language]

st.sidebar.markdown("---")
st.sidebar.markdown(f"✅ {txt['page_select']}:")
st.sidebar.markdown("[🔗 Homepage](https://yourdomain.com)")
# placeholder for future navigation

st.title(txt["title"])
model = load_model()

uploaded_file = st.file_uploader(txt["upload"], type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button(txt["predict"]):
        input_img = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_img)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top_class = torch.argmax(probs).item()
            predicted_class = class_names[top_class]
            confidence = probs[top_class].item()

        translated = TRANSLATIONS.get(predicted_class, {}).get(language, predicted_class)

        st.success(f"{txt['result']} **{translated}** ({confidence*100:.2f}%)")
        st.text_input(txt["copy_label"], value=translated, key="copy_field")

        st.write("### 🔍 Class Probabilities:")
        for i, score in enumerate(probs.tolist()):
            label = TRANSLATIONS.get(class_names[i], {}).get(language, class_names[i])
            st.write(f"{label}: {score*100:.2f}%")
