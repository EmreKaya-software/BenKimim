from flask import Flask, render_template , request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "riasec_real_data.csv"


app = Flask(__name__)



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze")
def analyze():
    # Tüm veri ile skorları hesapla
    df = pd.read_csv(DATA_PATH)
    riasec_scores = calculate_riasec_scores(df)

    labels = ["R", "I", "A", "S", "E", "C"]
    max_score_per_area = 35  # 7 soru * 5

    # Ortalama skorlar
    avg_series = riasec_scores[labels].mean()
    avg_scores = avg_series.round(2).to_dict()

    # Baskın alan dağılımı (kural tabanlı)
    dominant_series = riasec_scores[labels].idxmax(axis=1)
    dominant_counts = (
        dominant_series.value_counts()
        .reindex(labels, fill_value=0)
        .to_dict()
    )

    # Ortalama profil için "kural" sonucu + ML sonucu
    dominant_rule_avg = avg_series.idxmax()

    avg_row = pd.DataFrame([avg_scores])[labels]
    predicted_area = model.predict(avg_row)[0]

    ml_conf = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(avg_row)[0]
        ml_conf = float(np.max(proba))

    match_avg = (dominant_rule_avg == predicted_area)

    # Confusion Matrix (test split’ten) → HTML tablo
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Gerçek {l}" for l in labels],
        columns=[f"Tahmin {l}" for l in labels]
    )
    cm_html = cm_df.to_html(
        classes="table table-bordered table-sm text-center align-middle",
        border=0
    )

    return render_template(
        "result.html",
        scores=avg_scores,
        dominant_counts=dominant_counts,
        dominant_rule_avg=dominant_rule_avg,
        predicted_area=predicted_area,
        ml_conf=ml_conf,
        match_avg=match_avg,
        accuracy=GLOBAL_ACCURACY,
        f1=GLOBAL_F1,
        confusion_matrix=cm_html,
        max_score_per_area=max_score_per_area
    )



@app.route("/new-analysis", methods=["GET", "POST"])
def new_analysis():
    """
    /new-analysis:
    - Birincil sonuç: RIASEC kural tabanlı skorlar (idxmax)
    - İkincil sonuç: ML tahmini (destekleyici karar destek katmanı)
    """
    if request.method == "POST":
        # 1) Form cevaplarını oku (Q01–Q42)
        answers = {}
        for i in range(1, 43):
            key = f"Q{str(i).zfill(2)}"
            answers[key] = int(request.form[key])

        new_df = pd.DataFrame([answers])

        # 2) RIASEC skorlarını hesapla (kural tabanlı)
        riasec_scores = calculate_riasec_scores(new_df)
        scores_series = riasec_scores.iloc[0]

        dominant_rule = scores_series.idxmax()
        top3 = "-".join(scores_series.sort_values(ascending=False).index[:3].tolist())

        # 3) ML tahmini (destekleyici)
        dominant_ml = model.predict(riasec_scores)[0]

        # (opsiyonel) RF olasılığı: yaklaşık güven
        ml_conf = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(riasec_scores)[0]
            ml_conf = float(np.max(proba))

        match = (dominant_rule == dominant_ml)

        return render_template(
            "new_result.html",
            # skorlar
            scores=scores_series.to_dict(),
            max_score_per_area=35,  # 7 soru * 5 puan
            # kural tabanlı sonuç (birincil)
            dominant_rule=dominant_rule,
            top3=top3,
            description_rule=RIASEC_DESCRIPTIONS[dominant_rule],
            recommendations_rule=RIASEC_RECOMMENDATIONS[dominant_rule],
            # ml sonucu (ikincil)
            dominant_ml=dominant_ml,
            description_ml=RIASEC_DESCRIPTIONS[dominant_ml],
            ml_conf=ml_conf,
            match=match
        )

    return render_template("new_analysis.html", questions=QUESTIONS)


def calculate_riasec_scores(df):
    # 1) Google Forms sütunlarını Q01 formatına çevir
    rename_dict = {}
    for col in df.columns:
        if "[Q" in col:
            q_code = col.split("[Q")[1].split("-")[0]
            rename_dict[col] = f"Q{q_code}"

    df = df.rename(columns=rename_dict)

    # 2) Likert metin → sayısal dönüşüm
    likert_map = {
        "1 Hiç değil": 1,
        "2 Pek değil": 2,
        "3 Kararsızım": 3,
        "4 Evet": 4,
        "5 Kesinlikle Evet": 5,
        "5 Kesinlikle evet": 5  # küçük harf ihtimali için
    }

    for col in df.columns:
        if col.startswith("Q"):
            if df[col].dtype == object:
                df[col] = df[col].map(likert_map)


    # 3) RIASEC eşlemesi
    riasec_map = {
        "R": ["Q01", "Q07", "Q14", "Q22", "Q30", "Q32", "Q37"],
        "I": ["Q02", "Q11", "Q18", "Q21", "Q26", "Q33", "Q39"],
        "A": ["Q08", "Q17", "Q23", "Q27", "Q28", "Q31", "Q41"],
        "S": ["Q03", "Q04", "Q12", "Q13", "Q20", "Q34", "Q40"],
        "E": ["Q05", "Q10", "Q16", "Q19", "Q29", "Q36", "Q42"],
        "C": ["Q06", "Q09", "Q15", "Q24", "Q25", "Q35", "Q38"]
    }

    # 4) Skor hesapla
    scores = {}
    for key, questions in riasec_map.items():
        scores[key] = df[questions].sum(axis=1)

    return pd.DataFrame(scores)

def get_dominant_area(riasec_scores):
    return riasec_scores.idxmax(axis=1)

# =========================
# GLOBAL MODEL EĞİTİMİ
# =========================

df_global = pd.read_csv(DATA_PATH)


riasec_scores_global = calculate_riasec_scores(df_global)
df_global["Dominant"] = riasec_scores_global.idxmax(axis=1)

X_global = riasec_scores_global
y_global = df_global["Dominant"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_global, y_global,
    test_size=0.2,
    random_state=42,
    stratify=y_global
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_tr, y_tr)

# Performans (GLOBAL)
y_pred = model.predict(X_te)
GLOBAL_ACCURACY = round(accuracy_score(y_te, y_pred), 2)
GLOBAL_F1 = round(f1_score(y_te, y_pred, average="macro"), 2)
 
 #---------------------------------


RIASEC_DESCRIPTIONS = {
    "R": "Uygulamalı, teknik ve fiziksel etkinliklere yatkındır.",
    "I": "Araştırma, analiz ve problem çözmeyi sever.",
    "A": "Yaratıcı ve sanatsal çalışmalara ilgi duyar.",
    "S": "İnsanlarla iletişim kurmayı ve yardım etmeyi sever.",
    "E": "Liderlik ve girişimcilik yönü baskındır.",
    "C": "Planlı, düzenli ve sistemli çalışmayı tercih eder."
}

QUESTIONS = {
    "Q01": "Elimle bir şeyler yapmayı severim (tamir, montaj vb.)",
    "Q02": "Bir problemin nedenini araştırmak hoşuma gider.",
    "Q03": "İnsanlara yardım etmek beni mutlu eder.",
    "Q04": "Başkalarıyla birlikte çalışmayı severim.",
    "Q05": "Bir gruba liderlik etmeyi isterim.",
    "Q06": "Düzenli ve planlı çalışmayı severim.",
    "Q07": "Aletler ve makinelerle uğraşmak ilgimi çeker.",
    "Q08": "Resim, müzik veya sanatla ilgilenirim.",
    "Q09": "Dosyalama ve düzenleme işleri bana uygundur.",
    "Q10": "İnsanları bir fikre ikna etmeyi severim.",
    "Q11": "Bilimsel konular ilgimi çeker.",
    "Q12": "Başkalarının duygularını anlamaya çalışırım.",
    "Q13": "Birine bir şey öğretmek hoşuma gider.",
    "Q14": "Fiziksel olarak aktif olmayı severim.",
    "Q15": "Kurallara uymak benim için önemlidir.",
    "Q16": "Yeni bir iş fikri üretmek hoşuma gider.",
    "Q17": "Hayal gücümü kullanabileceğim işler ilgimi çeker.",
    "Q18": "Deney yapmayı ve sonuçları incelemeyi severim.",
    "Q19": "Bir işi başlatan kişi olmayı isterim.",
    "Q20": "İnsanlarla yüz yüze iletişim kurmayı severim.",
    "Q21": "Zor sorular üzerinde düşünmek hoşuma gider.",
    "Q22": "Açık havada çalışmak beni mutlu eder.",
    "Q23": "Yazı yazmak veya tasarım yapmak ilgimi çeker.",
    "Q24": "Verileri düzenlemek bana uygundur.",
    "Q25": "Detaylara dikkat ederim.",
    "Q26": "Bilgi toplamayı ve analiz etmeyi severim.",
    "Q27": "Kendimi sanatsal yollarla ifade etmeyi severim.",
    "Q28": "Yeni ve özgün fikirler üretirim.",
    "Q29": "Risk almayı severim.",
    "Q30": "El becerisi gerektiren işlerde iyiyimdir.",
    "Q31": "Estetik benim için önemlidir.",
    "Q32": "Pratik çözümler üretirim.",
    "Q33": "Araştırma yaparken sabırlıyımdır.",
    "Q34": "Başkalarına destek olmayı severim.",
    "Q35": "Plan yaparak çalışırım.",
    "Q36": "Bir grubu yönlendirmek isterim.",
    "Q37": "Teknik konular ilgimi çeker.",
    "Q38": "Belgelerle çalışmak bana uygundur.",
    "Q39": "Bilimsel keşifler ilgimi çeker.",
    "Q40": "İnsanların sorunlarını dinlemeyi severim.",
    "Q41": "Sanatsal etkinliklere katılmayı severim.",
    "Q42": "Kendi işimi kurma fikri hoşuma gider."
}

RIASEC_RECOMMENDATIONS = {
    "R": ["Robotik", "Maker", "Tamir Kulübü"],
    "I": ["Bilim Kulübü", "Kodlama", "Araştırma"],
    "A": ["Resim", "Müzik", "Tasarım"],
    "S": ["Sosyal Sorumluluk", "Mentorluk"],
    "E": ["Münazara", "Girişimcilik"],
    "C": ["Kütüphane", "Planlama", "Arşiv"]
}

if __name__ == "__main__":
    app.run(debug=True)
