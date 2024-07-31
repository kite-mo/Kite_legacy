import streamlit as st
import streamlit_app as slp

from core import plotting


from typing import Any, Dict, List, Tuple, Union


import pandas as pd

if "inference_key_list" not in st.session_state:
    st.session_state["inference_key_list"] = None

# 4. Display the contents of todolist


if st.session_state["scoring_df"] is not None:
    label_column = st.selectbox(
        "Select good/bad label",
        ["good", "bad"],
    )

    if label_column != None:
        scoring_df = st.session_state["scoring_df"]
        sorted_scoring_df = (
            scoring_df[scoring_df["label"] == label_column]
            .sort_values("wafer_score")
            .reset_index(drop=True)
        )
        if st.session_state["inference_key_list"] is None:
            result_key_list = slp.get_result_key_list_from_scoring_df(
                scoring_df=sorted_scoring_df
            )

            result_key_with_score = st.selectbox(
                "Select wafer key",
                result_key_list,
            )
            if len(sorted_scoring_df) > 0:
                result_key = result_key_with_score[:2]
                sensor_list_with_score = slp.get_sensor_key_list_from_scoring_df(
                    scoring_df=sorted_scoring_df,
                    result_key=result_key,
                    hm_column_list=st.session_state["hm_column_list"],
                )

                sensor_with_score = st.selectbox(
                    "Select sensor column",
                    sensor_list_with_score,
                )
                sensor_name = sensor_with_score[0]

                # 최종 reference 생성
                plotting_inference_button = st.button(
                    "Plot inference sensor !", key="plotting_start_button"
                )

                if plotting_inference_button:
                    reference_dict = st.session_state["reference"]
                    inference_dict = st.session_state["inference_result"][result_key][
                        "prediction"
                    ]
                    scoring_dict = st.session_state["inference_result"][result_key][
                        "scoring"
                    ]
                    sensor_inference_fig = plotting.plot_ref_target_score_ratio(
                        reference_dict=reference_dict,
                        inference_dict=inference_dict,
                        score_dict=scoring_dict,
                        sensor=sensor_name,
                    )
                    st.plotly_chart(sensor_inference_fig)

else:
    st.write("Please apply scoring logic using inference dataset")
