import streamlit_app as slp
import streamlit as st
from core import prediction
from core import scoring
from core import plotting

if "inference_uploader" not in st.session_state:
    st.session_state["inference_uploader"] = False
if "inference_uploaded" not in st.session_state:
    st.session_state["inference_uploaded"] = False
if "inference_dataset" not in st.session_state:
    st.session_state["inference_dataset"] = None
if "inference_result" not in st.session_state:
    st.session_state["inference_result"] = None
if "scoring_df" not in st.session_state:
    st.session_state["scoring_df"] = None


def get_inference_result_dict(
    sorted_first_time_list, sorted_info_list, sorted_wafer_list
):
    inferenece_result_dict = {}
    for first_time, info, wafer in zip(
        sorted_first_time_list, sorted_info_list, sorted_wafer_list
    ):
        wafer_key = (first_time, info)

        predict_result = prediction.prediction(
            df=wafer.reset_index(drop=True),
            model=st.session_state["reference"],
            hm_column_list=st.session_state["hm_column_list"],
        )

        score_result = scoring.scoring(
            model=st.session_state["reference"],
            prediction=predict_result,
            hm_column_list=st.session_state["hm_column_list"],
        )

        inferenece_result_dict[wafer_key] = {
            "prediction": predict_result["dist_ratio"],
            "scoring": score_result,
        }
    return inferenece_result_dict


if st.session_state["reference"]:
    st.write("#### Upload Inference Dataset")
    uploaded_files = st.file_uploader(
        "Upload inference files", type=["xlsx", "xls", "csv", "parquet", "feather"]
    )

    if uploaded_files:
        st.session_state["inference_uploaded"] = True
        st.session_state["inference_uploader"] = uploaded_files

    if st.session_state["inference_uploaded"]:
        if st.session_state["inference_dataset"] == None:
            inference_df = slp.read_time_series_df(
                upload_files=st.session_state["inference_uploader"], time_col="time"
            )
            (
                sorted_wafer_list,
                sorted_info_list,
                sorted_first_time_list,
            ) = slp.get_wafer_list_sorted_by_time(
                inference_df,
                key_list=st.session_state["key_column_list"],
                time_col="time",
            )
            st.session_state["inference_dataset"] = sorted_wafer_list

        if st.session_state["inference_result"] is None:
            inferenece_result_dict = get_inference_result_dict(
                sorted_first_time_list=sorted_first_time_list,
                sorted_info_list=sorted_info_list,
                sorted_wafer_list=sorted_wafer_list,
            )

            st.session_state["inference_result"] = inferenece_result_dict

        if st.session_state["scoring_df"] is None:
            scoring_df = slp.convert_result_dict_to_scoring_df(
                result_dict=st.session_state["inference_result"]
            )
            st.session_state["scoring_df"] = scoring_df

        wafer_score_fig = plotting.get_time_scatter_fig(
            score_df=st.session_state["scoring_df"], y_column="wafer_score"
        )
        st.plotly_chart(wafer_score_fig)

        plotting_column = st.selectbox(
            "Select plotting sensor score column in inference result",
            st.session_state["hm_column_list"],
        )
        if plotting_column != None:
            sensor_score_fig = plotting.get_time_scatter_fig(
                score_df=st.session_state["scoring_df"], y_column=plotting_column
            )
            st.plotly_chart(sensor_score_fig)
else:
    st.write("Please make reference first !")
