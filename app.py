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
        page_title="Three-Month Prognostic Prediction Tool",
        page_icon="🧮",
        layout="wide",
    )

    bundles, feature_specs = load_resources()

    st.title("Three-Month Prognostic Prediction Tool")
    st.write(
        "This web calculator estimates the probabilities of malignant cerebral edema and poor functional outcome at 3 months using three bundled ExtraTrees models."
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

        st.subheader("MCE Input for the Joint Model")
        mce_input_mode = st.radio(
            "How should the joint prognostic model use MCE?",
            options=(
                "Automatically use the predicted MCE class",
                "Manually set MCE to 0",
                "Manually set MCE to 1",
            ),
            index=0,
        )

        submitted = st.form_submit_button("Calculate Risk")

    if submitted:
        if mce_input_mode == "Manually set MCE to 0":
            manual_mce = 0
        elif mce_input_mode == "Manually set MCE to 1":
            manual_mce = 1
        else:
            manual_mce = None

        summary = run_prediction(patient_values, bundles, actual_mce=manual_mce)

        st.divider()
        st.header("Prediction Results")

        result_columns = st.columns(3)
        render_prediction(
            result_columns[0],
            "Malignant Cerebral Edema (MCE)",
            summary.mce,
            "Source: model_analysis_results_MBE_full / ExtraTrees",
        )
        render_prediction(
            result_columns[1],
            "Poor Functional Outcome at 3 Months",
            summary.functional,
            "Source: model_analysis_results_Diagnosis_full / ExtraTrees",
        )
        render_prediction(
            result_columns[2],
            "Poor Functional Outcome at 3 Months (Joint Model)",
            summary.functional_mce,
            "Source: model_analysis_results_Diagnosis+MBE_full / ExtraTrees",
        )

        st.subheader("MCE Handling for the Joint Model")
        st.write(f"MCE value used by the joint model: `{summary.mce_input_for_joint_model}`")
        st.write(f"Source: {summary.mce_input_source}")

        if manual_mce is None:
            st.caption(
                "When MCE is unknown, the joint model uses the predicted MCE class rather than the MCE probability."
            )


if __name__ == "__main__":
    main()
