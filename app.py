"""Streamlit two-panel demo for Uncertainty-Aware Plant Disease Diagnosis + Active Bayesian Questioning + GraphRAG."""
import csv
import os
from pathlib import Path

import plant_expert  # noqa: F401
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


def _csv_to_md(path: Path, max_rows: int = 30) -> str:
    if not path.exists():
        return ""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return ""
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows[1:max_rows + 1]]
    return "\n".join([header, sep, *body])


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


artifacts_dir = os.environ.get("PLANT_ARTIFACTS", "artifacts")
if not (Path(artifacts_dir) / "model.pt").exists() or not (Path(artifacts_dir) / "config.json").exists():
    st.error(
        f"Chưa tìm thấy `model.pt` và `config.json` trong thư mục `{artifacts_dir}`. "
        "Hãy chạy `.venv\\Scripts\\python scripts/bootstrap_demo.py` hoặc giải nén `artifacts.zip` từ Colab vào `artifacts/`."
    )
    st.stop()

model, cfg, kg, G, index = load_resources(artifacts_dir)
classes = cfg["classes"]
tau = float(cfg["tau"])
sym_ids, L = likelihood_matrix(kg, classes)
sym_by_idx = kg["context_symptoms"]
disease_by_id = {d["id"]: d for d in kg["diseases"]}

if cfg.get("bootstrap"):
    st.caption("⚠️ Đang chạy với trọng số khởi tạo demo (`bootstrap=True`). Để đạt độ chính xác thực địa, hãy chạy `notebooks/train_uncertainty.ipynb` trên Colab và chép `artifacts/` về.")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Tương tác Chẩn đoán & Tư vấn")
    up = st.file_uploader("Tải ảnh lá cà chua lên (.jpg, .png)", type=["jpg", "jpeg", "png"])
    sample_files = sorted(Path("samples").glob("*.png")) if Path("samples").exists() else []
    sample_choice = "-- Không chọn --"
    if sample_files:
        sample_choice = st.selectbox(
            "Hoặc chọn ảnh lá mẫu có sẵn",
            ["-- Không chọn --"] + [p.name for p in sample_files],
        )

    img_source = None
    file_key = None
    if up is not None:
        img_source = up
        file_key = f"upload:{up.name}:{up.size}"
    elif sample_choice != "-- Không chọn --":
        sp = Path("samples") / sample_choice
        img_source = sp
        file_key = f"sample:{sp.name}"

    if img_source is not None:
        if st.button("🔄 Đặt lại phiên hỏi đáp"):
            st.session_state.pop("file_key", None)
            st.rerun()

        if st.session_state.get("file_key") != file_key:
            img = Image.open(img_source)
            diag = vision.diagnose(model, cfg, img)
            st.session_state["file_key"] = file_key
            st.session_state["img"] = img
            st.session_state["diag"] = diag
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
                for col, (lbl, code) in zip(st.columns(3), (("Có", "yes"), ("Không", "no"), ("Không rõ", "unknown"))):
                    if col.button(lbl, key=f"{code}_{q_idx}_{len(st_cur.asked)}"):
                        st.session_state["state"] = bayes.answer(st_cur, q_idx, code, L)
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
                except graphrag.OllamaError:
                    st.info("Ollama chưa chạy trên máy — đang tổng hợp phác đồ trực tiếp từ Đồ thị Tri thức (chế độ ngoại tuyến).")
                    st.session_state["advise_res"] = graphrag.advise(
                        q_text, final_id, kg, G, index, mode="graph", allow_fallback=True
                    )

            res = st.session_state.get("advise_res")
            if res:
                if res["answer"]:
                    st.markdown("### Phác đồ Tư vấn từ Agricultural GraphRAG")
                    st.markdown(res["answer"])
                if res["facts"]:
                    with st.expander("Tri thức Đồ thị trích xuất trực tiếp (Sub-graph facts)", expanded=False):
                        for fl in res["facts"]:
                            st.write("- " + fl)

with col_right:
    tab_diag, tab_exp = st.tabs(["📊 Bảng điều khiển Ca bệnh hiện tại", "📈 Kết quả Thực nghiệm (Exp 1–3)"])

    with tab_diag:
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

            if st_cur.gated:
                free = [s for s in range(L.shape[0]) if s not in st_cur.asked]
                if free:
                    ranked = sorted(
                        ((bayes.info_gain(st_cur.post, L[s, st_cur.cand]), s) for s in free),
                        reverse=True,
                    )[:3]
                    ig_rows = ["| Mã | Câu hỏi triệu chứng còn lại | Information Gain |", "|---|---|---|"]
                    for ig_val, s_i in ranked:
                        q_short = sym_by_idx[s_i]["question_vi"]
                        ig_rows.append(f"| `{sym_ids[s_i]}` | {q_short} | `{ig_val:.4f}` |")
                    st.markdown("**Top 3 câu hỏi có Độ lợi Thông tin (IG) cao nhất**")
                    st.markdown("\n".join(ig_rows))

            cls_idx, _, _ = bayes.final(st_cur)
            final_id = classes[cls_idx]
            if final_id != "healthy":
                st.markdown("**Đồ thị Tri thức con (Knowledge Sub-graph: Bệnh → Hoạt chất → Chống chỉ định)**")
                sub = graphrag.subgraph(G, final_id, hops=2)
                components.html(render_subgraph_html(sub), height=380, scrolling=False)
        else:
            st.info("Hãy tải ảnh hoặc chọn ảnh mẫu ở khung bên trái để xem đồng hồ Entropy, biểu đồ Bayes và đồ thị tri thức.")

    with tab_exp:
        exp1_csv = Path(artifacts_dir) / "exp1_metrics.csv"
        if exp1_csv.exists():
            st.markdown("### Thí nghiệm 1: Định lượng độ bất định (ECE, AUROC, AURC)")
            st.markdown(_csv_to_md(exp1_csv))
            for img_name, cap in (
                ("exp1_reliability.png", "Reliability Diagram (blur)"),
                ("exp1_risk_coverage.png", "Risk-Coverage Curve (blur)"),
                ("exp1_entropy_hist.png", "Phân phối Entropy đúng vs sai"),
            ):
                p_img = Path(artifacts_dir) / img_name
                if p_img.exists():
                    st.image(str(p_img), caption=cap, use_container_width=True)

        exp2_csv = Path("experiments/out/exp2_results_simulated.csv")
        if exp2_csv.exists():
            st.markdown("### Thí nghiệm 2: Hỏi đáp Bayes chủ động (Dữ liệu giả lập)")
            st.markdown(_csv_to_md(exp2_csv))
            exp2_png = Path("experiments/out/exp2_sweep_simulated.png")
            if exp2_png.exists():
                st.image(str(exp2_png), caption="Độ nhạy theo mức nhiễu câu trả lời", use_container_width=True)

        exp3_csv = Path("experiments/out/exp3_summary.csv")
        if exp3_csv.exists():
            st.markdown("### Thí nghiệm 3: Chất lượng tư vấn GraphRAG vs Naive RAG vs LLM thuần")
            st.markdown(_csv_to_md(exp3_csv))
