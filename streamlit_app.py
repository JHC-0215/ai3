# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1Ux-LN9W-Wx-mtWSRwTpgIgOHRpintTH0")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {

    labels[0]: {
       "texts": ["실버 에로우,"],
       "images": ["data:image/jpeg;base64,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"],
       "videos": ["https://www.youtube.com/shorts/7S7-3U2AoeE"]
     },

    labels[1]: {
       "texts": ["스쿠데리아 페라리,"],
       "images": ["data:image/jpeg;base64,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"],
       "videos": ["https://www.youtube.com/shorts/7S7-3U2AoeE"]
     },

    labels[2]: {
       "texts": ["오라클 레드불 레이싱,"],
       "images": ["data:image/jpeg;base64,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"],
       "videos": ["https://www.youtube.com/shorts/7S7-3U2AoeE"]
     },
}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
