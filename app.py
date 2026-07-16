from __future__ import annotations

import streamlit as st

from predictor import FEATURE_SECTIONS, build_feature_specs, load_bundles, run_prediction


@st.cache_resource
def load_resources():
    bundles = load_bundles()
    feature_specs = build_feature_specs(bundles)
    return bundles, feature_specs


def render_feature_input(name, spec):
    kwargs = {}
    if spec.help_text:
        kwargs["help"] = spec.help_text

    if spec.widget == "binary":
        return st.selectbox(spec.label, options=spec.options, index=spec.options.index(spec.default), **kwargs)
    if spec.widget == "select":
        return st.selectbox(spec.label, options=spec.options, index=spec.options.index(spec.default), **kwargs)
    if spec.widget == "int":
        return st.number_input(
            spec.label,
            value=int(spec.default),
            step=int(spec.step),
            format=spec.format_string,
            **kwargs,
        )
    return st.number_input(
        spec.label,
        value=float(spec.default),
        step=float(spec.step),
        format=spec.format_string,
        **kwargs,
    )


def render_prediction(column, title, result, source_text):
    with column:
        st.subheader(title)
        st.metric("Probability", f"{result.probability:.1%}")
        st.caption(f"Decision threshold: {result.threshold:.3f}")
        st.caption(source_text)
        if result.high_risk:
            st.error("High-risk classification")
        else:
            st.success("Low-risk classification")


def main():
    st.set_page_config(
        page_title="ExtraTrees Online Calculator",
        page_icon="🧮",
        layout="wide",
    )

    bundles, feature_specs = load_resources()

    st.title("ExtraTrees 在线预测计算器")
    st.write(
        "基于你当前保存的 3 个 ExtraTrees 模型，自动输出 MCE 风险和 3 个月不良功能结局发生概率。"
    )
    st.info(
        "This tool is for research and decision support only. It does not replace clinician judgment."
    )

    with st.sidebar:
        st.header("Model Sources")
        for key, bundle in bundles.items():
            with st.expander(bundle.label, expanded=False):
                st.write(f"Asset folder: `assets/{bundle.directory.name}`")
                st.write(f"Features: {', '.join(bundle.feature_names)}")
                st.write(f"Classification threshold: {bundle.threshold:.3f}")
                if bundle.dca_threshold is not None:
                    st.write(f"DCA threshold: {bundle.dca_threshold:.3f}")

    with st.form("prediction_form"):
        patient_values = {}

        for section_title, feature_names in FEATURE_SECTIONS:
            st.subheader(section_title)
            columns = st.columns(2)
            for idx, feature_name in enumerate(feature_names):
                with columns[idx % 2]:
                    patient_values[feature_name] = render_feature_input(feature_name, feature_specs[feature_name])

        st.subheader("Joint model MCE input")
        mce_input_mode = st.radio(
            "Diagnosis+MBE 联合模型如何处理 MCE？",
            options=(
                "Auto use predicted MCE class",
                "Manual MCE = 0",
                "Manual MCE = 1",
            ),
            index=0,
        )

        submitted = st.form_submit_button("Calculate probabilities")

    if submitted:
        if mce_input_mode == "Manual MCE = 0":
            manual_mce = 0
        elif mce_input_mode == "Manual MCE = 1":
            manual_mce = 1
        else:
            manual_mce = None

        summary = run_prediction(patient_values, bundles, actual_mce=manual_mce)

        st.divider()
        st.header("Prediction Results")

        result_columns = st.columns(3)
        render_prediction(
            result_columns[0],
            "MCE",
            summary.mce,
            "Source: model_analysis_results_MBE_full / ExtraTrees",
        )
        render_prediction(
            result_columns[1],
            "3-month poor functional outcome",
            summary.functional,
            "Source: model_analysis_results_Diagnosis_full / ExtraTrees",
        )
        render_prediction(
            result_columns[2],
            "3-month outcome + MCE joint model",
            summary.functional_mce,
            "Source: model_analysis_results_Diagnosis+MBE_full / ExtraTrees",
        )

        st.subheader("Joint-model MCE handling")
        st.write(f"MCE value fed into the joint model: `{summary.mce_input_for_joint_model}`")
        st.write(f"Source: {summary.mce_input_source}")

        if manual_mce is None:
            st.caption(
                "When MCE is unknown, the joint model uses the predicted MCE class rather than the MCE probability."
            )


if __name__ == "__main__":
    main()
