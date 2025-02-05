from flask import Flask, render_template, request, jsonify
import json
import arabic_reshaper
from bidi.algorithm import get_display
import networkx as nx
import matplotlib.pyplot as plt
import speech_recognition as sr
from PIL import Image
import matplotlib.font_manager as fm
import re
from flask import request, jsonify
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import io

app = Flask(__name__)

# تحميل خط يدعم العربية
# arabic_font = fm.FontProperties(fname="static/fonts/arial.ttf")


def reshape_arabic_text(text):
    """إعادة تشكيل النص العربي ليتوافق مع الرسم"""
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)


def load_knowledge_base():
    """تحميل قاعدة المعرفة من ملف JSON"""
    file_id = "1_Yi0K6hVGDJjZRhoxBJxrwyKLxbRPlCc"  # Replace with your file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "translated_diseases_v2.json"  # Output file name
    gdown.download(url, output, quiet=False, fuzzy=True)  # Use fuzzy to handle shareable links

    try:
        with open(output, "r", encoding="utf-8") as file:
            content = file.read()

            if not content:
                return {}
            data = json.loads(content)
            return data.get("symptoms", {})
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def preprocess_text(text):
    """تنظيف وتحليل النص المدخل"""
    text = re.sub(r'[^\w\s]', '', text)  # إزالة الرموز الخاصة
    text = re.sub(r'[أإآ]', "ا", text)  # تطبيع الحروف
    text = re.sub(r'ة', "ه", text)  # تطبيع الحروف
    return text.strip().split()


def build_semantic_network(knowledge_base):
    """بناء شبكة دلالية للأمراض والأعراض"""
    network = nx.DiGraph()
    for disease, details in knowledge_base.items():
        network.add_node(disease, type="disease")
        for symptom in details.get("الأعراض", []):
            network.add_node(symptom, type="symptom")
            network.add_edge(disease, symptom, relation="has_symptom")
    return network


def calculate_similarity(input_symptoms, disease_symptoms):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([input_symptoms, disease_symptoms])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]


def diagnose_disease(input_text, knowledge_base, semantic_network):
    """تشخيص المرض بناءً على الأعراض المدخلة مع تحسين دقة المطابقة"""
    input_symptoms = preprocess_text(input_text)
    disease_scores = []

    for disease in semantic_network.nodes:
        if semantic_network.nodes[disease].get("type") == "disease":
            disease_symptoms = [symptom for symptom in semantic_network.successors(disease)]

            # حساب التشابه الدلالي
            input_symptoms_str = " ".join(input_symptoms)
            disease_symptoms_str = " ".join(disease_symptoms)
            similarity = calculate_similarity(input_symptoms_str, disease_symptoms_str)

            # حساب عدد الأعراض المتطابقة
            matched_symptoms = sum(1 for symptom in input_symptoms if symptom in disease_symptoms)
            total_symptoms = len(disease_symptoms)

            if matched_symptoms > 0:
                match_ratio = matched_symptoms / total_symptoms
                disease_scores.append((disease, match_ratio, matched_symptoms, similarity))

    if disease_scores:
        # ترتيب الأمراض حسب التشابه الدلالي ثم نسبة المطابقة
        disease_scores.sort(key=lambda x: (-x[3], -x[1], -x[2]))
        best_match = disease_scores[0][0]
        details = knowledge_base.get(best_match, {})
        return best_match, details, disease_scores
    return None, None, []


def visualize_network(input_text, semantic_network):
    """رسم الشبكة الدلالية للأعراض المدخلة وعرضها مباشرة"""
    input_symptoms = preprocess_text(input_text)
    sub_network = nx.DiGraph()

    for symptom in input_symptoms:
        if semantic_network.has_node(symptom):
            sub_network.add_node(symptom, type="symptom")
            for disease in semantic_network.predecessors(symptom):
                sub_network.add_node(disease, type="disease")
                sub_network.add_edge(disease, symptom, relation="has_symptom")

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(sub_network, seed=42)
    node_colors = ["lightblue" if sub_network.nodes[node].get("type") == "disease" else "lightcoral" for node in sub_network.nodes]

    nx.draw(sub_network, pos, with_labels=True, node_color=node_colors, node_size=1500, font_size=10, font_family="Arial")

    # تحويل الرسم إلى HTML
    graph_html = mpld3.fig_to_html(plt.gcf())
    plt.close()  # إغلاق الرسم لمنع تسرب الذاكرة
    return graph_html

def recognize_speech():
    """التعرف على الصوت وتحويله إلى نص"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio, language="ar")
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return ""

from pydub import AudioSegment
import io

@app.route('/recognize_speech', methods=['POST'])
def recognize_speech_route():
    """التعرف على الصوت وتحويله إلى نص"""
    if 'audio' not in request.files:
        return jsonify({"error": "لم يتم إرسال أي ملف صوتي."}), 400

    audio_file = request.files['audio']
    recognizer = sr.Recognizer()

    try:
        # تحويل الملف الصوتي إلى WAV باستخدام pydub
        audio = AudioSegment.from_file(audio_file)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # التعرف على الصوت
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="ar")
            return jsonify({"text": text})
    except sr.UnknownValueError:
        return jsonify({"error": "لم يتم التعرف على أي صوت."}), 400
    except sr.RequestError:
        return jsonify({"error": "حدث خطأ في الاتصال بخدمة التعرف على الصوت."}), 500
    except Exception as e:
        return jsonify({"error": f"حدث خطأ غير متوقع: {str(e)}"}), 500
@app.route('/', methods=['GET', 'POST'])
def index():
    knowledge_base = load_knowledge_base()
    if not knowledge_base:
        return render_template('index.html', error="⚠️ ملف قاعدة المعرفة فارغ أو غير موجود!")

    semantic_network = build_semantic_network(knowledge_base)

    if request.method == 'POST':
        input_text = request.form.get('input_text', '')
        action = request.form.get('action')

        if action == 'diagnose':
            disease, details, _ = diagnose_disease(input_text, knowledge_base, semantic_network)
            if disease:
                return render_template('index.html', disease=disease, details=details)
            else:
                return render_template('index.html', warning="⚠️ لم يتم العثور على مرض مطابق.")

        elif action == 'visualize':
            graph_html = visualize_network(input_text, semantic_network)
            return render_template('index.html', graph_html=graph_html)

        elif action == 'reverse_search':
            reverse_search = request.form.get('reverse_search', '')
            if reverse_search in knowledge_base:
                details = knowledge_base[reverse_search]
                symptoms = details.get("الأعراض", [])
                return render_template('index.html', reverse_search=reverse_search, symptoms=symptoms)
            else:
                return render_template('index.html', warning="⚠️ لم يتم العثور على هذا المرض في قاعدة البيانات.")

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
