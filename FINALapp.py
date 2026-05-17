import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIG
# ============================================
st.set_page_config(
    page_title="TULT Trash AI - Ultimate Dashboard",
    page_icon="🗑️",
    layout="wide"
)

# Custom CSS dengan tambahan untuk dashboard
st.markdown("""
<style>
/* Style dasar */
.tile {
    transition: all 0.3s ease-in-out;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    color: white;
    font-size: 13px;
    font-weight: bold;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    cursor: pointer;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.tile:hover {
    transform: translateY(-5px);
    box-shadow: 2px 5px 20px rgba(0,0,0,0.3);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    padding: 20px;
    color: white;
}
.metric-card-green {
    background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    color: #333;
    border-radius: 15px;
    padding: 20px;
}
.metric-card-orange {
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    color: #333;
    border-radius: 15px;
    padding: 20px;
}
.metric-card-red {
    background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
    color: #333;
    border-radius: 15px;
    padding: 20px;
}
.metric-card-blue {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    color: #333;
    border-radius: 15px;
    padding: 20px;
}
.metric-card-purple {
    background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
    color: #333;
    border-radius: 15px;
    padding: 20px;
}

/* Floor header */
.floor-header {
    background: linear-gradient(90deg, #4a90e2, #357abd);
    color: white;
    padding: 10px 15px;
    border-radius: 8px;
    margin: 20px 0 10px 0;
    font-weight: bold;
    font-size: 16px;
}

/* Tombol */
.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 10px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}

/* Animasi */
.pulse-animation {
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

/* Dashboard section */
.dashboard-section {
    background: white;
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

/* Risk card */
.risk-card {
    border-radius: 12px; 
    padding: 15px; 
    color: white; 
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}
.risk-card:hover {
    transform: translateY(-3px);
}

/* Divider custom */
.divider-custom {
    height: 3px;
    background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
    border-radius: 3px;
    margin: 20px 0;
}

/* Detail list item */
.detail-item {
    background: #f8f9fa;
    padding: 8px 12px;
    border-radius: 6px;
    margin: 3px 0;
    border-left: 4px solid #667eea;
}

/* Stat box */
.stat-box {
    background: white;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING & PREPROCESSING
# ============================================
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("TULT_TRASH_DATASET_DEMO20.csv")
    
    # Konversi Tanggal & Jam
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['Jam_Integer'] = pd.to_datetime(df['Jam'], format='%H:%M:%S').dt.hour
    
    # Kategorikan waktu
    def kategorikan_waktu(jam):
        if 6 <= jam < 10:
            return 'Pagi'
        elif 10 <= jam < 14:
            return 'Siang'
        elif 14 <= jam < 19:
            return 'Sore'
        else:
            return 'Malam'
    
    df['Kategori_Waktu'] = df['Jam_Integer'].apply(kategorikan_waktu)
    df['Is_Weekend'] = df['Hari'].isin(['Sabtu', 'Minggu']).astype(int)
    
    # Target binary
    threshold = df['Berat (kg)'].quantile(0.75)
    df['Target_Banyak_Sampah'] = (df['Berat (kg)'] > threshold).astype(int)
    
    # Ekstrak Nomor Ruangan
    df['Nomor_Ruangan'] = df['Nama Ruangan'].str.extract(r'(\d{4})').astype(float)
    df['Lantai'] = (df['Nomor_Ruangan'] // 100).astype(int)
    df['No_Urut'] = (df['Nomor_Ruangan'] % 100).astype(int)
    
    return df, threshold

df, threshold = load_and_preprocess()

# Encoding
features = ['Nama Ruangan', 'Hari', 'Kategori_Waktu', 'Aktivitas']
le_dict = {}
for col in features:
    le = LabelEncoder()
    df[f'{col}_Encoded'] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# ============================================
# MACHINE LEARNING MODELS
# ============================================
@st.cache_resource
def train_models(data):
    feature_columns = ['Nama Ruangan_Encoded', 'Hari_Encoded', 
                       'Kategori_Waktu_Encoded', 'Aktivitas_Encoded', 
                       'Is_Weekend']
    
    X = data[feature_columns]
    
    # Target
    y_binary = data['Target_Banyak_Sampah']
    y_kategori = data['Kategori Sampah']
    le_kategori = LabelEncoder()
    y_kategori_encoded = le_kategori.fit_transform(y_kategori)
    
    y_detail = data['Detail Sampah']
    le_detail = LabelEncoder()
    y_detail_encoded = le_detail.fit_transform(y_detail)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    # Model Binary
    binary_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, 
        random_state=42, class_weight='balanced'
    )
    binary_model.fit(X_train, y_train)
    
    # Split untuk kategori
    X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
        X_scaled, y_kategori_encoded, test_size=0.2, 
        random_state=42, stratify=y_kategori_encoded
    )
    
    # Model Kategori
    kategori_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, 
        random_state=42, class_weight='balanced'
    )
    kategori_model.fit(X_train_k, y_train_k)
    
    # Model Detail
    detail_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, 
        random_state=42, class_weight='balanced'
    )
    detail_model.fit(X_train_k, y_train_k)
    
    # Evaluasi
    y_pred = binary_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Fitur': [col.replace('_Encoded', '') for col in feature_columns],
        'Importance': binary_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Simpan semua kelas unik
    all_kategori = le_kategori.classes_
    all_detail = le_detail.classes_
    
    return binary_model, kategori_model, detail_model, scaler, accuracy, feature_importance, le_kategori, le_detail, all_kategori, all_detail

model_binary, model_kategori, model_detail, scaler, model_accuracy, feature_importance, le_kategori, le_detail, all_kategori, all_detail = train_models(df)

# ============================================
# FUNGSI SORTING
# ============================================
def sort_rooms_numerically(room_list):
    def extract_number(room_name):
        match = re.search(r'(\d{4})', room_name)
        if match:
            return int(match.group(1))
        return 0
    return sorted(room_list, key=extract_number)

# ============================================
# SIDEBAR CONTROLS
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/trash.png", width=80)
    st.header("🕹️ Panel Kontrol")
    st.markdown("---")
    
    hari_list = sorted(df['Hari'].unique().tolist())
    aktivitas_list = sorted(df['Aktivitas'].unique().tolist())
    lantai_list = sorted(df['Lantai'].unique().tolist())
    lantai_list = [l for l in lantai_list if l > 0]
    
    selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
    selected_aktivitas = st.selectbox("📚 Pilih Aktivitas", aktivitas_list)
    selected_jam = st.slider("🕐 Simulasi Jam", 0, 23, 10)
    
    st.markdown("---")
    st.subheader("🏢 Filter Lantai")
    
    lantai_options = ["Semua Lantai"] + [f"Lantai {l}" for l in lantai_list]
    selected_lantai_display = st.selectbox("Pilih Lantai", lantai_options)
    
    if selected_lantai_display == "Semua Lantai":
        selected_lantai = None
    else:
        selected_lantai = int(selected_lantai_display.split()[1])
    
    def kategorikan_waktu(jam):
        if 6 <= jam < 10:
            return 'Pagi'
        elif 10 <= jam < 14:
            return 'Siang'
        elif 14 <= jam < 19:
            return 'Sore'
        else:
            return 'Malam'
    
    selected_waktu = kategorikan_waktu(selected_jam)
    
    st.info(f"⏰ Kategori Waktu: **{selected_waktu}**")
    
    st.markdown("---")
    st.subheader("📊 Informasi Model")
    st.metric("Akurasi Model", f"{model_accuracy:.1%}")
    st.caption(f"Threshold Sampah Banyak: > {threshold:.2f} kg")
    st.caption(f"Total Data: {len(df)} sampel")
    st.caption(f"Ruangan: {df['Nama Ruangan'].nunique()} ruangan")
    
    st.markdown("---")
    st.subheader("🔍 Fitur Terpenting")
    for i, row in feature_importance.head(3).iterrows():
        st.caption(f"**{row['Fitur']}**: {row['Importance']:.3f}")

# ============================================
# PREDIKSI UNTUK SEMUA RUANGAN
# ============================================
rooms = df['Nama Ruangan'].unique()
rooms_sorted = sort_rooms_numerically(rooms)

if selected_lantai is not None:
    rooms_sorted = [r for r in rooms_sorted if int(re.search(r'(\d{4})', r).group(1)) // 100 == selected_lantai]

prediction_results = []

for room in rooms_sorted:
    nomor_ruangan = re.search(r'(\d{4})', room)
    if nomor_ruangan:
        no = int(nomor_ruangan.group(1))
        lantai = no // 100
        no_urut = no % 100
    else:
        lantai = 0
        no_urut = 0
        no = 0
    
    try:
        ruangan_enc = le_dict['Nama Ruangan'].transform([room])[0]
        hari_enc = le_dict['Hari'].transform([selected_hari])[0]
        waktu_enc = le_dict['Kategori_Waktu'].transform([selected_waktu])[0]
        aktivitas_enc = le_dict['Aktivitas'].transform([selected_aktivitas])[0]
        is_weekend = 1 if selected_hari in ['Sabtu', 'Minggu'] else 0
        
        input_features = pd.DataFrame([{
            'Nama Ruangan_Encoded': ruangan_enc,
            'Hari_Encoded': hari_enc,
            'Kategori_Waktu_Encoded': waktu_enc,
            'Aktivitas_Encoded': aktivitas_enc,
            'Is_Weekend': is_weekend
        }])
        
        input_scaled = scaler.transform(input_features)
        
        prob_banyak = model_binary.predict_proba(input_scaled)[0][1]
        is_banyak = model_binary.predict(input_scaled)[0]
        
        prob_kategori = model_kategori.predict_proba(input_scaled)[0]
        idx_kategori = model_kategori.predict(input_scaled)[0]
        kategori_pred = le_kategori.inverse_transform([idx_kategori])[0]
        
        prob_detail = model_detail.predict_proba(input_scaled)[0]
        idx_detail = model_detail.predict(input_scaled)[0]
        detail_pred = le_detail.inverse_transform([idx_detail])[0]
        
        prediction_results.append({
            'Ruangan': room,
            'Nomor': no,
            'Lantai': lantai,
            'No_Urut': no_urut,
            'Kategori_Prediksi': kategori_pred,
            'Detail_Prediksi': detail_pred,
            'Probabilitas_Banyak': prob_banyak,
            'Prob_Kategori': prob_kategori,
            'Prob_Detail': prob_detail,
            'Status': '🔴 BANYAK' if is_banyak == 1 else '🟢 SEDIKIT',
            'Is_Banyak': is_banyak
        })
    except:
        prediction_results.append({
            'Ruangan': room,
            'Nomor': no,
            'Lantai': lantai,
            'No_Urut': no_urut,
            'Kategori_Prediksi': 'Error',
            'Detail_Prediksi': 'Error',
            'Probabilitas_Banyak': 0,
            'Prob_Kategori': [0],
            'Prob_Detail': [0],
            'Status': '⚠️ ERROR',
            'Is_Banyak': 0
        })

res_df = pd.DataFrame(prediction_results)

# ============================================
# MAIN DASHBOARD
# ============================================
st.title("🤖 TULT Trash Prediction AI - Ultimate Dashboard")
st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)

if selected_lantai is not None:
    st.caption(f"📊 Analisis Simulasi: {selected_hari}, Jam {selected_jam}:00 ({selected_waktu}) - Aktivitas: {selected_aktivitas} | Lantai {selected_lantai}")
else:
    st.caption(f"📊 Analisis Simulasi: {selected_hari}, Jam {selected_jam}:00 ({selected_waktu}) - Aktivitas: {selected_aktivitas} | Semua Lantai")

# ============================================
# ROW 1: KPI METRICS
# ============================================
st.markdown("### 📈 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("📋 Total Ruangan", len(res_df))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    banyak_count = (res_df['Is_Banyak'] == 1).sum()
    st.markdown('<div class="metric-card-red">', unsafe_allow_html=True)
    st.metric("⚠️ Berpotensi Banyak", banyak_count, 
              delta=f"{banyak_count/len(res_df)*100:.1f}%" if len(res_df) > 0 else "0%")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    sedikit_count = (res_df['Is_Banyak'] == 0).sum()
    st.markdown('<div class="metric-card-green">', unsafe_allow_html=True)
    st.metric("✅ Aman", sedikit_count,
              delta=f"{sedikit_count/len(res_df)*100:.1f}%" if len(res_df) > 0 else "0%")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card-orange">', unsafe_allow_html=True)
    if len(res_df) > 0:
        st.metric("🗑️ Kategori Dominan", res_df['Kategori_Prediksi'].mode()[0])
    else:
        st.metric("🗑️ Kategori Dominan", "-")
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    avg_prob = res_df['Probabilitas_Banyak'].mean() if len(res_df) > 0 else 0
    st.markdown('<div class="metric-card-blue">', unsafe_allow_html=True)
    st.metric("📊 Rata-rata Probabilitas", f"{avg_prob:.1%}")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)

if len(res_df) == 0:
    st.warning("Tidak ada ruangan untuk lantai yang dipilih.")
else:
    # ============================================
    # ROW 2: HEATMAP TILES & PIE CHARTS
    # ============================================
    col_left, col_right = st.columns([1.5, 1])
    
    with col_left:
        st.markdown("### 📍 Heatmap Kepadatan Sampah per Ruangan")
        
        if selected_lantai is not None:
            st.caption(f"Lantai {selected_lantai} | Hijau (Aman) → Merah (Bahaya)")
        else:
            st.caption("Semua Lantai | Hijau (Aman) → Merah (Bahaya)")
        
        min_prob = res_df['Probabilitas_Banyak'].min()
        max_prob = res_df['Probabilitas_Banyak'].max()
        
        def get_color(prob):
            if max_prob == min_prob:
                return "rgb(255, 200, 0)"
            normalized = (prob - min_prob) / (max_prob - min_prob)
            if normalized < 0.25:
                ratio = normalized / 0.25
                r = int(46 + (241 - 46) * ratio)
                g = int(204 + (196 - 204) * ratio)
                b = int(113 + (15 - 113) * ratio)
            elif normalized < 0.5:
                ratio = (normalized - 0.25) / 0.25
                r = int(241 + (243 - 241) * ratio)
                g = int(196 + (156 - 196) * ratio)
                b = int(15 + (18 - 15) * ratio)
            elif normalized < 0.75:
                ratio = (normalized - 0.5) / 0.25
                r = int(243 + (231 - 243) * ratio)
                g = int(156 + (76 - 156) * ratio)
                b = int(18 + (60 - 18) * ratio)
            else:
                r, g, b = 231, 76, 60
            return f"rgb({r},{g},{b})"
        
        lantai_display_list = sorted(res_df['Lantai'].unique())
        
        for lantai in lantai_display_list:
            if lantai > 0:
                lantai_df = res_df[res_df['Lantai'] == lantai].copy()
                
                if len(lantai_df) > 0:
                    st.markdown(f'<div class="floor-header">🏢 Lantai {lantai} ({len(lantai_df)} Ruangan)</div>', unsafe_allow_html=True)
                    
                    num_cols = min(8, len(lantai_df))
                    cols = st.columns(num_cols)
                    
                    for i, (_, row) in enumerate(lantai_df.iterrows()):
                        with cols[i % num_cols]:
                            color = get_color(row['Probabilitas_Banyak'])
                            text_color = 'white' if row['Probabilitas_Banyak'] > 0.6 else '#333'
                            
                            if row['Probabilitas_Banyak'] > 0.7:
                                icon = "🔴"
                            elif row['Probabilitas_Banyak'] > 0.4:
                                icon = "🟡"
                            else:
                                icon = "🟢"
                            
                            st.markdown(f"""
                            <div class="tile" style="background:{color}; color:{text_color}">
                                <div style="font-size:11px; opacity:0.9">{icon}</div>
                                <strong>{row['Ruangan']}</strong>
                                <div style="font-size:16px; margin:5px 0">{row['Probabilitas_Banyak']:.1%}</div>
                                <div style="font-size:10px; opacity:0.9">🗑️ {row['Kategori_Prediksi']}</div>
                                <div style="font-size:10px; opacity:0.9">📦 {row['Detail_Prediksi']}</div>
                                <div style="font-size:10px; margin-top:2px">{row['Status']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
        
        # Legenda
        col_l1, col_l2, col_l3, col_l4 = st.columns(4)
        with col_l1:
            st.markdown("🟢 **Rendah** (<25%)")
        with col_l2:
            st.markdown("🟡 **Sedang** (25-50%)")
        with col_l3:
            st.markdown("🟠 **Waspada** (50-75%)")
        with col_l4:
            st.markdown("🔴 **Tinggi** (>75%)")
    
    with col_right:
        st.markdown("### 📊 Komposisi Prediksi")
        
        # Pie chart kategori
        if len(res_df) > 0:
            kategori_counts = res_df['Kategori_Prediksi'].value_counts()
            
            fig_kategori = px.pie(
                values=kategori_counts.values,
                names=kategori_counts.index,
                title="Distribusi Kategori Sampah",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_kategori.update_traces(textposition='inside', textinfo='percent+label')
            fig_kategori.update_layout(height=300)
            st.plotly_chart(fig_kategori, use_container_width=True)
        
        # Top ruangan berisiko
        st.markdown("### ⚠️ Top 5 Ruangan Prioritas")
        top_5 = res_df.nlargest(5, 'Probabilitas_Banyak')
        
        if len(top_5) > 0:
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                prob = row['Probabilitas_Banyak']
                color = get_color(prob)
                text_color = 'white' if prob > 0.6 else '#333'
                st.markdown(f"""
                <div style="background:{color}; padding:8px; border-radius:8px; margin:3px 0; color:{text_color}">
                    <strong>#{i} {row['Ruangan']}</strong> (Lt.{row['Lantai']})<br>
                    <small>{prob:.1%} | {row['Kategori_Prediksi']} | {row['Detail_Prediksi']}</small>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)
    
    # ============================================
    # ROW 2.5: KOMPOSISI DETAIL SAMPAH (NEW)
    # ============================================
    st.markdown("## 🔍 Analisis Komposisi Detail Sampah")
    
    # Metric cards untuk detail
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    
    with col_d1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_unique_details = res_df['Detail_Prediksi'].nunique()
        st.metric("📦 Total Jenis Detail", total_unique_details)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_d2:
        st.markdown('<div class="metric-card-green">', unsafe_allow_html=True)
        if len(res_df) > 0:
            most_common_detail = res_df['Detail_Prediksi'].mode()[0]
            most_common_count = res_df['Detail_Prediksi'].value_counts().iloc[0]
            st.metric("🏆 Detail Terbanyak", most_common_detail, 
                     delta=f"{most_common_count} ruangan ({most_common_count/len(res_df)*100:.1f}%)")
        else:
            st.metric("🏆 Detail Terbanyak", "-")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_d3:
        st.markdown('<div class="metric-card-orange">', unsafe_allow_html=True)
        if len(res_df) > 0:
            detail_avg_prob = res_df.groupby('Detail_Prediksi')['Probabilitas_Banyak'].agg(['mean', 'count'])
            detail_avg_prob = detail_avg_prob[detail_avg_prob['count'] >= 2]  # Minimal 2 ruangan
            if len(detail_avg_prob) > 0:
                highest_risk_detail = detail_avg_prob['mean'].idxmax()
                highest_risk_value = detail_avg_prob['mean'].max()
                st.metric("⚠️ Detail Risiko Tertinggi", highest_risk_detail, 
                         delta=f"Rata-rata {highest_risk_value:.1%}")
            else:
                st.metric("⚠️ Detail Risiko Tertinggi", "-")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_d4:
        st.markdown('<div class="metric-card-purple">', unsafe_allow_html=True)
        if len(res_df) > 0:
            detail_counts = res_df['Detail_Prediksi'].value_counts()
            top3_count = detail_counts.head(3).sum()
            st.metric("📊 Top 3 Detail", f"{top3_count} ruangan",
                     delta=f"{top3_count/len(res_df)*100:.1f}% dari total")
        else:
            st.metric("📊 Top 3 Detail", "-")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualisasi detail sampah
    col_v1, col_v2 = st.columns([1, 1])
    
    with col_v1:
        # Treemap detail sampah
        if len(res_df) > 0:
            detail_counts_full = res_df['Detail_Prediksi'].value_counts().reset_index()
            detail_counts_full.columns = ['Detail', 'Count']
            
            fig_treemap = px.treemap(
                detail_counts_full.head(20),
                path=['Detail'],
                values='Count',
                title="Treemap Top 20 Detail Sampah",
                color='Count',
                color_continuous_scale='deep',
                hover_data={'Count': True}
            )
            fig_treemap.update_layout(height=450)
            fig_treemap.update_traces(
                textinfo="label+value+percent root",
                textfont=dict(size=11)
            )
            st.plotly_chart(fig_treemap, use_container_width=True)
    
    with col_v2:
        # Bar chart horizontal detail sampah
        if len(res_df) > 0:
            detail_counts_sorted = res_df['Detail_Prediksi'].value_counts().head(15)
            
            fig_detail_bar = px.bar(
                x=detail_counts_sorted.values,
                y=detail_counts_sorted.index,
                orientation='h',
                title="Top 15 Detail Sampah Terprediksi",
                labels={'x': 'Jumlah Ruangan', 'y': 'Detail Sampah'},
                color=detail_counts_sorted.values,
                color_continuous_scale='viridis',
                text=detail_counts_sorted.values
            )
            fig_detail_bar.update_traces(textposition='outside')
            fig_detail_bar.update_layout(
                height=450, 
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False
            )
            st.plotly_chart(fig_detail_bar, use_container_width=True)
    
    st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)
    
    # ============================================
    # ROW 2.6: HIERARKI KATEGORI → DETAIL & DISTRIBUSI PER LANTAI
    # ============================================
    st.markdown("### 🗂️ Hierarki Kategori → Detail & Distribusi per Lantai")
    
    col_h1, col_h2 = st.columns([1, 1])
    
    with col_h1:
        # Sunburst chart: Kategori -> Detail
        if len(res_df) > 0:
            sunburst_data = res_df.groupby(['Kategori_Prediksi', 'Detail_Prediksi']).size().reset_index(name='Count')
            
            fig_sunburst = px.sunburst(
                sunburst_data,
                path=['Kategori_Prediksi', 'Detail_Prediksi'],
                values='Count',
                title="Hierarki Kategori → Detail Sampah",
                color='Count',
                color_continuous_scale='ice',
                hover_data={'Count': True}
            )
            fig_sunburst.update_layout(height=500)
            fig_sunburst.update_traces(textinfo="label+percent parent")
            st.plotly_chart(fig_sunburst, use_container_width=True)
    
    with col_h2:
        # Heatmap detail sampah per lantai
        if len(res_df) > 0:
            lantai_detail = res_df[res_df['Lantai'] > 0].groupby(['Lantai', 'Detail_Prediksi']).size().reset_index(name='Count')
            
            # Get top details across all floors
            top_details_overall = res_df['Detail_Prediksi'].value_counts().head(10).index
            lantai_detail_filtered = lantai_detail[lantai_detail['Detail_Prediksi'].isin(top_details_overall)]
            
            pivot_heatmap = lantai_detail_filtered.pivot_table(
                values='Count',
                index='Lantai',
                columns='Detail_Prediksi',
                fill_value=0
            )
            
            if not pivot_heatmap.empty:
                fig_heatmap_detail = px.imshow(
                    pivot_heatmap,
                    labels=dict(x="Detail Sampah", y="Lantai", color="Jumlah Ruangan"),
                    color_continuous_scale="YlOrRd",
                    aspect="auto",
                    title="Heatmap Detail Sampah per Lantai (Top 10 Detail)",
                    text_auto=True
                )
                fig_heatmap_detail.update_layout(height=500)
                st.plotly_chart(fig_heatmap_detail, use_container_width=True)
    
    st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)
    
    # ============================================
    # ROW 2.7: ANALISIS DETAIL PER KATEGORI & PIE CHART DETAIL
    # ============================================
    st.markdown("### 📊 Detail per Kategori & Komposisi Lengkap")
    
    col_a1, col_a2 = st.columns([1, 1])
    
    with col_a1:
        # Pie chart detail sampah (full)
        if len(res_df) > 0:
            detail_counts = res_df['Detail_Prediksi'].value_counts().head(12)
            
            fig_detail_pie = px.pie(
                values=detail_counts.values,
                names=detail_counts.index,
                title="Top 12 Detail Sampah Terprediksi",
                hole=0.35,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_detail_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                textfont_size=9
            )
            fig_detail_pie.update_layout(height=450)
            st.plotly_chart(fig_detail_pie, use_container_width=True)
    
    with col_a2:
        # Breakdown detail per kategori
        st.markdown("#### 📋 Breakdown Detail per Kategori")
        
        if len(res_df) > 0:
            # Create tabs for each category
            kategori_list = sorted(res_df['Kategori_Prediksi'].unique())
            
            if len(kategori_list) > 0:
                # Use selectbox instead of tabs for better space
                selected_kategori_detail = st.selectbox(
                    "Pilih Kategori untuk Lihat Detail",
                    kategori_list,
                    key="detail_kategori_select"
                )
                
                kategori_data = res_df[res_df['Kategori_Prediksi'] == selected_kategori_detail]
                detail_counts = kategori_data['Detail_Prediksi'].value_counts().head(8)
                
                if len(detail_counts) > 0:
                    # Mini bar chart untuk detail dalam kategori
                    fig_mini_bar = px.bar(
                        x=detail_counts.values,
                        y=detail_counts.index,
                        orientation='h',
                        title=f"Detail Sampah untuk Kategori: {selected_kategori_detail}",
                        labels={'x': 'Jumlah Ruangan', 'y': 'Detail'},
                        color=detail_counts.values,
                        color_continuous_scale='agsunset',
                        text=detail_counts.values
                    )
                    fig_mini_bar.update_traces(textposition='outside')
                    fig_mini_bar.update_layout(
                        height=350,
                        yaxis={'categoryorder': 'total ascending'},
                        showlegend=False
                    )
                    st.plotly_chart(fig_mini_bar, use_container_width=True)
                    
                    # Tampilkan list
                    st.markdown(f"**Total Ruangan dengan Kategori {selected_kategori_detail}:** {len(kategori_data)}")
                    for detail, count in detail_counts.items():
                        percentage = (count / len(kategori_data)) * 100
                        st.markdown(f"""
                        <div class="detail-item">
                            <strong>{detail}</strong>: {count} ruangan ({percentage:.1f}%)
                        </div>
                        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)
    
    # ============================================
    # ROW 3: BAR CHART INTERACTIVE
    # ============================================
    st.markdown("### 📊 Analisis Probabilitas per Ruangan")
    
    col_opt1, col_opt2, col_opt3 = st.columns([2, 1, 1])
    with col_opt1:
        st.caption("🎯 Visualisasi Interaktif dengan Informasi Komposisi")
    with col_opt2:
        sort_option = st.selectbox(
            "Urutkan Berdasarkan",
            ["Nomor Ruangan", "Probabilitas Tertinggi", "Probabilitas Terendah"],
            key="sort_bar"
        )
    with col_opt3:
        color_theme = st.selectbox(
            "Tema Warna",
            ["Gradient Merah-Hijau", "Kategori Risiko", "Gradient Biru", "Gradient Pelangi"],
            key="color_theme"
        )
    
    if sort_option == "Probabilitas Tertinggi":
        res_df_sorted = res_df.sort_values('Probabilitas_Banyak', ascending=False)
    elif sort_option == "Probabilitas Terendah":
        res_df_sorted = res_df.sort_values('Probabilitas_Banyak', ascending=True)
    else:
        res_df_sorted = res_df.sort_values(['Lantai', 'Nomor'])
    
    def get_theme_colors(probabilities, theme):
        if theme == "Kategori Risiko":
            colors = []
            for prob in probabilities:
                if prob <= 0.25:
                    colors.append('#27AE60')
                elif prob <= 0.50:
                    colors.append('#F1C40F')
                elif prob <= 0.75:
                    colors.append('#E67E22')
                else:
                    colors.append('#E74C3C')
            return colors
        elif theme == "Gradient Biru":
            return px.colors.sequential.Blues
        elif theme == "Gradient Pelangi":
            return px.colors.sequential.Rainbow
        else:
            return px.colors.diverging.RdYlGn
    
    fig_bar = go.Figure()
    bar_colors = get_theme_colors(res_df_sorted['Probabilitas_Banyak'], color_theme)
    
    fig_bar.add_trace(go.Bar(
        x=res_df_sorted['Ruangan'],
        y=res_df_sorted['Probabilitas_Banyak'],
        name='Probabilitas',
        marker=dict(
            color=res_df_sorted['Probabilitas_Banyak'] if color_theme in ["Gradient Merah-Hijau", "Gradient Biru", "Gradient Pelangi"] else bar_colors,
            colorscale='RdYlGn_r' if color_theme == "Gradient Merah-Hijau" else 
                      'Blues' if color_theme == "Gradient Biru" else 
                      'Rainbow' if color_theme == "Gradient Pelangi" else None,
            showscale=True,
            colorbar=dict(title="Probabilitas", tickformat='.0%'),
            line=dict(color='rgba(255, 255, 255, 0.8)', width=1.5),
            opacity=0.9
        ),
        text=res_df_sorted.apply(
            lambda x: f"{x['Probabilitas_Banyak']:.1%}<br>🗑️ {x['Kategori_Prediksi']}", 
            axis=1
        ),
        textposition='outside',
        hovertemplate=(
            '<b>🏢 %{x}</b><br>' +
            '📊 Probabilitas: <b>%{y:.1%}</b><br>' +
            '🗑️ Kategori: <b>%{customdata[0]}</b><br>' +
            '📦 Detail: <b>%{customdata[1]}</b><br>' +
            '📈 Status: <b>%{customdata[2]}</b><br>' +
            '<extra></extra>'
        ),
        customdata=res_df_sorted[['Kategori_Prediksi', 'Detail_Prediksi', 'Status']].values
    ))
    
    # Threshold lines
    for y_val, color, label in [(0.25, '#27AE60', 'Rendah'), 
                                  (0.50, '#F39C12', 'Sedang'), 
                                  (0.75, '#E74C3C', 'Tinggi')]:
        fig_bar.add_hline(y=y_val, line_dash="dash", line_color=color, 
                         annotation_text=label, annotation_position="right")
    
    avg_probability = res_df_sorted['Probabilitas_Banyak'].mean()
    fig_bar.add_hline(y=avg_probability, line_dash="solid", line_color="#8E44AD",
                     annotation_text=f"Rata-rata: {avg_probability:.1%}",
                     annotation_position="left")
    
    fig_bar.update_layout(
        title=f'Probabilitas Sampah Banyak<br><sup>{selected_hari} | {selected_jam}:00 ({selected_waktu}) | {selected_aktivitas}</sup>',
        xaxis_title="Ruangan",
        yaxis_title="Probabilitas",
        yaxis_tickformat='.0%',
        height=600,
        showlegend=False
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Risk summary cards
    st.markdown("### 📊 Ringkasan Kategori Risiko")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rendah_count = (res_df_sorted['Probabilitas_Banyak'] <= 0.25).sum()
        st.markdown(f"""
        <div class="risk-card" style="background: linear-gradient(135deg, #27AE60, #2ECC71);">
            <h4>🟢 AMAN</h4>
            <h2>{rendah_count}</h2>
            <p>Ruangan (0-25%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        sedang_count = ((res_df_sorted['Probabilitas_Banyak'] > 0.25) & 
                       (res_df_sorted['Probabilitas_Banyak'] <= 0.50)).sum()
        st.markdown(f"""
        <div class="risk-card" style="background: linear-gradient(135deg, #F1C40F, #F39C12);">
            <h4>🟡 NORMAL</h4>
            <h2>{sedang_count}</h2>
            <p>Ruangan (25-50%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        waspada_count = ((res_df_sorted['Probabilitas_Banyak'] > 0.50) & 
                        (res_df_sorted['Probabilitas_Banyak'] <= 0.75)).sum()
        st.markdown(f"""
        <div class="risk-card" style="background: linear-gradient(135deg, #E67E22, #D35400);">
            <h4>🟠 WASPADA</h4>
            <h2>{waspada_count}</h2>
            <p>Ruangan (50-75%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        tinggi_count = (res_df_sorted['Probabilitas_Banyak'] > 0.75).sum()
        st.markdown(f"""
        <div class="risk-card" style="background: linear-gradient(135deg, #E74C3C, #C0392B);">
            <h4>🔴 KRITIS</h4>
            <h2>{tinggi_count}</h2>
            <p>Ruangan (>75%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)
    
    # ============================================
    # ROW 4: HEATMAP MATRIX & PIE DETAIL
    # ============================================
    col_a, col_b = st.columns([1.5, 1])
    
    with col_a:
        st.markdown("### 🗺️ Heatmap Matrix Probabilitas")
        
        lantai_data = []
        for lantai in lantai_display_list:
            if lantai > 0:
                lantai_df = res_df[res_df['Lantai'] == lantai].copy()
                for _, row in lantai_df.iterrows():
                    lantai_data.append({
                        'Lantai': f'Lt. {lantai}',
                        'No_Urut': row['No_Urut'],
                        'Probabilitas': row['Probabilitas_Banyak'],
                        'Detail': row['Detail_Prediksi']
                    })
        
        if lantai_data:
            heatmap_df = pd.DataFrame(lantai_data)
            pivot_data = heatmap_df.pivot_table(
                values='Probabilitas',
                index='Lantai',
                columns='No_Urut',
                aggfunc='first'
            )
            
            fig_heatmap = px.imshow(
                pivot_data,
                labels=dict(x="Nomor Urut", y="Lantai", color="Probabilitas"),
                x=pivot_data.columns,
                y=pivot_data.index,
                color_continuous_scale="RdYlGn_r",
                aspect="auto",
                title="Heatmap Probabilitas per Lantai"
            )
            fig_heatmap.update_layout(height=450)
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col_b:
        st.markdown("### 🔍 Analisis Detail per Ruangan")
        
        room_selected = st.selectbox("Pilih Ruangan untuk Detail", rooms_sorted, key="room_detail")
        
        if room_selected:
            room_data = res_df[res_df['Ruangan'] == room_selected]
            if not room_data.empty:
                row = room_data.iloc[0]
                
                # Kategori probabilities
                prob_kategori = row['Prob_Kategori']
                if isinstance(prob_kategori, np.ndarray) and len(prob_kategori) > 0:
                    kategori_labels = le_kategori.classes_
                    
                    fig_room = px.pie(
                        values=prob_kategori,
                        names=kategori_labels,
                        title=f"Probabilitas Kategori - {room_selected}",
                        hole=0.5,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_room.update_layout(height=300)
                    st.plotly_chart(fig_room, use_container_width=True)
                
                # Detail info
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                          border-radius: 12px; padding: 15px; color: white;">
                    <h4>📋 Informasi {room_selected}</h4>
                    <p><strong>Lantai:</strong> {row['Lantai']} | <strong>No Urut:</strong> {row['No_Urut']}</p>
                    <p><strong>Probabilitas Banyak:</strong> {row['Probabilitas_Banyak']:.1%}</p>
                    <p><strong>Status:</strong> {row['Status']}</p>
                    <p><strong>Kategori:</strong> {row['Kategori_Prediksi']}</p>
                    <p><strong>Detail:</strong> {row['Detail_Prediksi']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Tambahan: Probabilitas detail untuk ruangan ini
                prob_detail = row['Prob_Detail']
                if isinstance(prob_detail, np.ndarray) and len(prob_detail) > 5:
                    st.markdown("**Top 5 Detail Terprediksi:**")
                    detail_labels = le_detail.classes_
                    top5_idx = np.argsort(prob_detail)[-5:][::-1]
                    for idx in top5_idx:
                        st.markdown(f"- {detail_labels[idx]}: {prob_detail[idx]:.1%}")
    
    st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)
    
    # ============================================
    # ROW 5: PROYEKSI 24 JAM & DISTRIBUSI LANTAI
    # ============================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Proyeksi Ruangan Berisiko 24 Jam")
        
        hourly_banyak = []
        hourly_probs = []
        
        if selected_lantai is not None:
            proyeksi_rooms = [r for r in sort_rooms_numerically(df['Nama Ruangan'].unique()) 
                            if int(re.search(r'(\d{4})', r).group(1)) // 100 == selected_lantai]
        else:
            proyeksi_rooms = sort_rooms_numerically(df['Nama Ruangan'].unique())
        
        for h in range(24):
            count_banyak = 0
            probs_list = []
            waktu_h = kategorikan_waktu(h)
            
            for room in proyeksi_rooms:
                nomor = re.search(r'(\d{4})', room)
                if nomor:
                    no = int(nomor.group(1))
                    lantai = no // 100
                    no_urut = no % 100
                else:
                    lantai, no_urut = 0, 0
                
                try:
                    ruangan_enc = le_dict['Nama Ruangan'].transform([room])[0]
                    hari_enc = le_dict['Hari'].transform([selected_hari])[0]
                    waktu_enc = le_dict['Kategori_Waktu'].transform([waktu_h])[0]
                    aktivitas_enc = le_dict['Aktivitas'].transform([selected_aktivitas])[0]
                    is_weekend = 1 if selected_hari in ['Sabtu', 'Minggu'] else 0
                    
                    input_features = pd.DataFrame([{
                        'Nama Ruangan_Encoded': ruangan_enc,
                        'Hari_Encoded': hari_enc,
                        'Kategori_Waktu_Encoded': waktu_enc,
                        'Aktivitas_Encoded': aktivitas_enc,
                        'Is_Weekend': is_weekend
                    }])
                    
                    input_scaled = scaler.transform(input_features)
                    prob = model_binary.predict_proba(input_scaled)[0][1]
                    probs_list.append(prob)
                    
                    if model_binary.predict(input_scaled)[0] == 1:
                        count_banyak += 1
                except:
                    continue
            
            hourly_banyak.append(count_banyak)
            hourly_probs.append(np.mean(probs_list) if probs_list else 0)
        
        fig_proyeksi = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_proyeksi.add_trace(
            go.Bar(
                name="Ruangan Berisiko",
                x=[f"{h:02d}:00" for h in range(24)],
                y=hourly_banyak,
                marker_color='#FF6B6B',
                opacity=0.7
            ),
            secondary_y=False
        )
        
        fig_proyeksi.add_trace(
            go.Scatter(
                name="Rata-rata Probabilitas",
                x=[f"{h:02d}:00" for h in range(24)],
                y=hourly_probs,
                mode='lines+markers',
                line=dict(color='#4ECDC4', width=3),
                marker=dict(size=8)
            ),
            secondary_y=True
        )
        
        fig_proyeksi.update_layout(
            title=f'Proyeksi 24 Jam - {selected_hari}',
            height=400,
            hovermode='x unified'
        )
        fig_proyeksi.update_yaxes(title_text="Jumlah Ruangan Berisiko", secondary_y=False)
        fig_proyeksi.update_yaxes(title_text="Rata-rata Probabilitas", secondary_y=True, tickformat='.0%')
        fig_proyeksi.update_xaxes(title_text="Jam")
        
        st.plotly_chart(fig_proyeksi, use_container_width=True)
    
    with col2:
        st.markdown("### 🏢 Distribusi Risiko per Lantai")
        
        lantai_stats = res_df.groupby('Lantai').agg({
            'Probabilitas_Banyak': 'mean',
            'Ruangan': 'count',
            'Is_Banyak': 'sum'
        }).reset_index()
        
        lantai_stats.columns = ['Lantai', 'Rata-rata Probabilitas', 'Jumlah Ruangan', 'Ruangan Berisiko']
        lantai_stats = lantai_stats[lantai_stats['Lantai'] > 0]
        
        if len(lantai_stats) > 0:
            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_dual.add_trace(
                go.Bar(name="Total Ruangan", x=lantai_stats['Lantai'].astype(str),
                      y=lantai_stats['Jumlah Ruangan'], marker_color='lightblue', opacity=0.7),
                secondary_y=False
            )
            
            fig_dual.add_trace(
                go.Bar(name="Ruangan Berisiko", x=lantai_stats['Lantai'].astype(str),
                      y=lantai_stats['Ruangan Berisiko'], marker_color='red', opacity=0.7),
                secondary_y=False
            )
            
            fig_dual.add_trace(
                go.Scatter(name="Rata-rata Probabilitas", x=lantai_stats['Lantai'].astype(str),
                          y=lantai_stats['Rata-rata Probabilitas'],
                          mode='lines+markers', line=dict(color='green', width=3), marker=dict(size=10)),
                secondary_y=True
            )
            
            fig_dual.update_layout(title="Analisis Risiko per Lantai", height=400, hovermode='x unified', barmode='group')
            fig_dual.update_yaxes(title_text="Jumlah Ruangan", secondary_y=False)
            fig_dual.update_yaxes(title_text="Probabilitas", secondary_y=True, tickformat='.0%')
            fig_dual.update_xaxes(title_text="Lantai")
            
            st.plotly_chart(fig_dual, use_container_width=True)
    
    st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)
    
    # ============================================
    # ROW 6: DETAIL TABLE & TOP 10
    # ============================================
    st.markdown("### 📋 Detail Prediksi per Ruangan")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_df = res_df[['Ruangan', 'Nomor', 'Lantai', 'Kategori_Prediksi', 
                             'Detail_Prediksi', 'Probabilitas_Banyak', 'Status']].copy()
        display_df['Probabilitas_Banyak'] = display_df['Probabilitas_Banyak'].apply(lambda x: f"{x:.1%}")
        display_df = display_df.sort_values(['Lantai', 'Nomor'])
        
        st.dataframe(
            display_df,
            column_config={
                "Ruangan": "Nama Ruangan",
                "Nomor": st.column_config.NumberColumn("No", format="%d"),
                "Lantai": "Lantai",
                "Kategori_Prediksi": "Kategori",
                "Detail_Prediksi": "Detail",
                "Probabilitas_Banyak": "Probabilitas",
                "Status": "Status"
            },
            hide_index=True,
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.markdown("### ⚠️ Top 10 Ruangan Prioritas")
        top_10 = res_df.nlargest(10, 'Probabilitas_Banyak')
        
        if len(top_10) > 0:
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                prob = row['Probabilitas_Banyak']
                color = get_color(prob)
                text_color = 'white' if prob > 0.6 else '#333'
                
                if i <= 3:
                    medal = ['🥇', '🥈', '🥉'][i-1]
                else:
                    medal = f"#{i}"
                
                st.markdown(f"""
                <div style="background:{color}; padding:8px; border-radius:8px; margin:3px 0; color:{text_color}">
                    <strong>{medal} {row['Ruangan']}</strong> (Lt.{row['Lantai']})<br>
                    <small>{prob:.1%} | {row['Kategori_Prediksi']} | {row['Detail_Prediksi']}</small>
                </div>
                """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info("💡 **Tips**: Gunakan filter lantai untuk fokus pada area tertentu")
with col2:
    st.success(f"🎯 **Model**: Random Forest | Akurasi: {model_accuracy:.1%}")
with col3:
    st.warning("⚠️ **Prioritas**: Ruangan >50% memerlukan perhatian")
with col4:
    st.markdown("📖 **Kategori & Detail Sampah:**")
    st.caption(f"**Kategori:** {', '.join(all_kategori[:5])}")
    st.caption(f"**Detail:** {', '.join(all_detail[:5])}...")

st.caption("© 2024 TULT Trash Prediction AI | Ultimate Dashboard | Dataset: 300 sampel, 21 ruangan")

# ============================================
# EXPANDER: INFORMASI TAMBAHAN
# ============================================
with st.expander("📖 Informasi Lengkap Model & Dataset"):
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Info", "🗑️ Kategori Sampah", "📦 Detail Sampah", "📈 Feature Importance"])
    
    with tab1:
        st.markdown(f"""
        **Model yang Digunakan:**
        - Binary Classification: Random Forest (n=100, max_depth=10)
        - Kategori Classification: Random Forest (n=100, max_depth=10)
        - Detail Classification: Random Forest (n=100, max_depth=10)
        
        **Akurasi Model Binary:** {model_accuracy:.1%}
        **Threshold Sampah Banyak:** > {threshold:.2f} kg
        **Total Dataset:** {len(df)} sampel
        **Jumlah Ruangan:** {df['Nama Ruangan'].nunique()} ruangan
        """)
    
    with tab2:
        st.markdown("**Kategori Sampah:**")
        for kat in all_kategori:
            count = len(df[df['Kategori Sampah'] == kat])
            st.markdown(f"- {kat} ({count} sampel)")
    
    with tab3:
        st.markdown("**Detail Sampah (20 pertama):**")
        for det in all_detail[:20]:
            count = len(df[df['Detail Sampah'] == det])
            st.markdown(f"- {det} ({count} sampel)")
        
        if len(all_detail) > 20:
            st.caption(f"... dan {len(all_detail) - 20} detail lainnya")
    
    with tab4:
        st.dataframe(feature_importance, use_container_width=True)
        
        fig_fi = px.bar(
            feature_importance, 
            x='Importance', y='Fitur',
            orientation='h',
            title="Feature Importance",
            color='Importance',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_fi, use_container_width=True)