import streamlit as st
from core import modeling
from core import plotting

if "hm_column_list" not in st.session_state:
    st.session_state["hm_column_list"] = None
if "step_column" not in st.session_state:
    st.session_state["step_column"] = None
if "reference_button" not in st.session_state:
    st.session_state["reference_button"] = False
if "reference" not in st.session_state:
    st.session_state["reference"] = None


if st.session_state["reference_dataset"]:
    new_hm_column_list = []
    if st.session_state["hm_column_list"] is not None:  # 최초 시점
        new_hm_column_list = st.multiselect(
            "Select sensor columns which want to make reference EHM",
            st.session_state["sensor_column_list"],
            st.session_state["hm_column_list"],
        )
    else:
        hm_column_list = st.multiselect(
            "Select sensor columns which want to make reference EHM",
            st.session_state["sensor_column_list"],
        )

    # step column 유무 확인
    if st.session_state["step_column"] is None:
        step_column = st.text_input(
            "Write the column name containing the step information. If not, enter None",
            "None",
        )
        st.session_state["step_column"] = step_column
    else:
        step_column = st.text_input(
            "Write the column name containing the step information. If not, enter None",
            st.session_state["step_column"],
        )

    # 최종 reference 생성
    reference_start_button = st.button("Make reference !", key="referecne_start_button")

    if reference_start_button:
        if st.session_state["hm_column_list"] is None:
            st.session_state["hm_column_list"] = hm_column_list

        # 최초 시점
        if st.session_state["reference"] is None:
            model_dict = modeling.modeling(
                df_list=st.session_state["reference_dataset"],
                hm_column_list=st.session_state["hm_column_list"],
            )
            st.session_state["reference"] = model_dict
        # 페이지 변경시, hm_column 이 바뀔 경우
        elif set(st.session_state["hm_column_list"]) != set(new_hm_column_list):
            model_dict = modeling.modeling(
                df_list=st.session_state["reference_dataset"],
                hm_column_list=new_hm_column_list,
            )
            st.session_state["hm_column_list"] = new_hm_column_list
            st.session_state["reference"] = model_dict

    # 센서 별 reference plot 띄우기
    if st.session_state["reference"]:
        plotting_column = st.selectbox(
            "Select plotting sensor column in reference EHM",
            st.session_state["hm_column_list"],
        )
        if plotting_column != None:
            fig = plotting.plot_compare_ref_by_sensor(
                st.session_state["reference"], plotting_column
            )
            st.plotly_chart(fig)
else:
    st.write("Please reference upload dataset !")
