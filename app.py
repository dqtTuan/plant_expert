"""Streamlit two-panel demo for Uncertainty-Aware Plant Disease Diagnosis + Active Bayesian Questioning + GraphRAG."""
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from pyvis.network import Network

from plant_expert import bayes, graphrag, vision
from plant_expert.kg import build_graph, likelihood_matrix, load_kg

st.set_page_config(page_title="Chẩn đoán Bệnh Cà chua Đa phương thức", layout="wide")
st.title("Hệ thống Chẩn đoán Bệnh Cà chua Đa phương thức (MC Dropout + Bayes + GraphRAG)")


@st.cache_resource
def load_resources(artifacts_dir: str):
    model, cfg = vision.load(artifacts_dir)
    kg = load_kg("kg/kg.json")
    G = build_graph(kg)
    idx = graphrag.VectorIndex.load("kg/index.npz") if Path("kg/index.npz").exists() else None
    return model, cfg, kg, G, idx


artifacts_dir = os.environ.get("PLANT_ARTIFACTS", "artifacts")
if not (Path(artifacts_dir) / "model.pt").exists() or not (Path(artifacts_dir) / "config.json").exists():
    st.error(
        f"Chưa tìm thấy `model.pt` và `config.json` trong thư mục `{artifacts_dir}`. "
        "Hãy chạy `notebooks/train_uncertainty.ipynb` trên Google Colab (hoặc máy cục bộ) rồi giải nén `artifacts.zip` vào `artifacts/`."
    )
    st.stop()

model, cfg, kg, G, index = load_resources(artifacts_dir)
classes = cfg["classes"]
tau = float(cfg["tau"])
sym_ids, L = likelihood_matrix(kg, classes)
sym_by_idx = kg["context_symptoms"]
disease_by_id = {d["id"]: d for d in kg["diseases"]}


def render_subgraph_html(sub) -> str:
    net = Network(height="360px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
    color_map = {
        "Disease": "#e63946",
        "ActiveIngredient": "#2a9d8f",
        "Contraindication": "#d62828",
        "Pathogen": "#457b9d",
        "VisualSymptom": "#e9c46a",
    }
    for n, d in sub.nodes(data=True):
        ntype = d.get("type", "")
        net.add_node(n, label=d.get("label", n)[:48], title=d.get("label", n), color=color_map.get(ntype, "#8d99ae"))
    for u, v, d in sub.edges(data=True):
        rel = d.get("rel", "")
        is_path = rel in ("TREATED_BY", "RESTRICTED_WITH")
        net.add_edge(u, v, label=rel, color="#d62828" if is_path else "#adb5bd", width=3 if is_path else 1)
    return net.generate_html()


col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Tương tác Chẩn đoán & Tư vấn")
    up = st.file_uploader("Tải ảnh lá cà chua lên (.jpg, .png)", type=["jpg", "jpeg", "png"])
    if up is not None:
        file_key = f"{up.name}:{up.size}"
        if st.session_state.get("file_key") != file_key:
            img = Image.open(up)
            diag = vision.diagnose(model, cfg, img)
            st.session_state["file_key"] = file_key
            st.session_state["img"] = img
            st.session_state["diag"] = diag
            st.session_state["prior_cand"] = None
            st_init = bayes.start(diag["p_mean"], diag["H"], tau)
            st.session_state["state"] = st_init
            st.session_state["prior_post"] = st_init.post.copy()
            st.session_state["advise_res"] = None

        st.image(st.session_state["img"], caption="Ảnh lá đầu vào", width=280)
        st_cur: bayes.State = st.session_state["state"]
        diag = st.session_state["diag"]

        if st_cur.gated:
            st.warning(
                f"Độ bất định H(p̂) = {diag['H']:.3f} > ngưỡng τ = {tau:.3f}. "
                "Hệ thống kích hoạt hỏi đáp triệu chứng phụ theo Bayes (bảng P(s|c) là ước lượng chuyên gia)."
            )
        else:
            st.success(f"Độ bất định H(p̂) = {diag['H']:.3f} ≤ τ = {tau:.3f}. Kết luận trực tiếp từ ảnh.")

        for s_idx, ans_code, _ in st_cur.history:
            ans_vi = {"yes": "Có", "no": "Không", "unknown": "Không rõ"}[ans_code]
            with st.chat_message("assistant"):
                st.write(sym_by_idx[s_idx]["question_vi"])
            with st.chat_message("user"):
                st.write(ans_vi)

        q_idx = bayes.next_question(st_cur, L)
        if q_idx is not None:
            with st.chat_message("assistant"):
                st.write(sym_by_idx[q_idx]["question_vi"])
                st.caption("Lưu ý: Trọng số P(s|c) dùng để cập nhật Bayes là ước lượng chuyên gia.")
                b1, b2, b3 = st.columns(3)
                if b1.button("Có", key=f"yes_{q_idx}_{len(st_cur.asked)}"):
                    st.session_state["state"] = bayes.answer(st_cur, q_idx, "yes", L)
                    st.rerun()
                if b2.button("Không", key=f"no_{q_idx}_{len(st_cur.asked)}"):
                    st.session_state["state"] = bayes.answer(st_cur, q_idx, "no", L)
                    st.rerun()
                if b3.button("Không rõ", key=f"unk_{q_idx}_{len(st_cur.asked)}"):
                    st.session_state["state"] = bayes.answer(st_cur, q_idx, "unknown", L)
                    st.rerun()
        else:
            cls_idx, prob, confirmed = bayes.final(st_cur)
            final_id = classes[cls_idx]
            final_vi = disease_by_id[final_id]["name_vi"]
            if confirmed:
                st.success(f"**Kết luận bệnh:** {final_vi} (`{final_id}`) — Xác suất: {prob*100:.1f}%")
            else:
                st.warning(
                    f"**Nghi ngờ cao nhất (chưa vượt ngưỡng 85%):** {final_vi} (`{final_id}`) — Xác suất: {prob*100:.1f}%"
                )

            user_q = st.chat_input("Đặt câu hỏi tư vấn phác đồ điều trị (mặc định: Cách điều trị bệnh này?)")
            if user_q or st.session_state.get("advise_res") is None:
                q_text = user_q or "Cách điều trị và cảnh báo chống chỉ định cho bệnh này?"
                try:
                    st.session_state["advise_res"] = graphrag.advise(q_text, final_id, kg, G, index, mode="graph")
                except graphrag.OllamaError as e:
                    st.warning(str(e))
                    sub = graphrag.subgraph(G, final_id, hops=2)
                    st.session_state["advise_res"] = {
                        "answer": "",
                        "facts": graphrag.facts(sub, final_id),
                        "passages": [],
                        "context": "",
                    }

            res = st.session_state.get("advise_res")
            if res:
                if res["answer"]:
                    st.markdown("### Phác đồ Tư vấn từ Agricultural GraphRAG")
                    st.write(res["answer"])
                if res["facts"]:
                    with st.expander("Tri thức Đồ thị trích xuất trực tiếp (Sub-graph facts)", expanded=not bool(res["answer"])):
                        for fl in res["facts"]:
                            st.write("- " + fl)

with col_right:
    st.subheader("Bảng điều khiển Học máy Nâng cao")
    if "diag" in st.session_state:
        diag = st.session_state["diag"]
        st_cur: bayes.State = st.session_state["state"]

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=diag["H"],
            delta={"reference": tau},
            title={"text": f"Predictive Entropy H(p̂) vs ngưỡng τ={tau:.2f} (MI={diag['MI']:.3f})"},
            gauge={
                "axis": {"range": [0, float(np.log(len(classes)))]},
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.8, "value": tau},
            },
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        cand_names = [disease_by_id[classes[i]]["name_vi"] for i in st_cur.cand]
        before_p = st.session_state["prior_post"]
        after_p = st_cur.post
        fig_bar = go.Figure(data=[
            go.Bar(name="Trước hỏi đáp (Prior)", x=cand_names, y=before_p),
            go.Bar(name="Sau hỏi đáp (Posterior)", x=cand_names, y=after_p),
        ])
        fig_bar.update_layout(
            barmode="group", height=280,
            title="Phân phối xác suất ứng viên Trước & Sau cập nhật Bayes",
            yaxis=dict(range=[0, 1]),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        cls_idx, _, _ = bayes.final(st_cur)
        final_id = classes[cls_idx]
        if final_id != "healthy":
            st.markdown("**Đồ thị Tri thức con (Knowledge Sub-graph: Bệnh → Hoạt chất → Chống chỉ định)**")
            sub = graphrag.subgraph(G, final_id, hops=2)
            components.html(render_subgraph_html(sub), height=380, scrolling=False)
