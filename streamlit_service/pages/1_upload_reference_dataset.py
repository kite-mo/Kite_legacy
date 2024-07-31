import streamlit_app as slp
import streamlit as st
import yaml

with open("./setting_columns.yml") as f:
    setting_dict = yaml.load(f, Loader=yaml.FullLoader)

if "reference_uploaded" not in st.session_state:
    st.session_state["reference_uploaded"] = False
if "reference_uploader" not in st.session_state:
    st.session_state["reference_uploader"] = None
if "reference_dataset" not in st.session_state:
    st.session_state["reference_dataset"] = None
if "sensor_column_list" not in st.session_state:
    st.session_state["sensor_column_list"] = None
if "meta_column_list" not in st.session_state:
    st.session_state["meta_column_list"] = None
if "key_column_list" not in st.session_state:
    st.session_state["key_column_list"] = None

st.session_state["meta_column_list"] = setting_dict["meta_column_list"]
st.session_state["key_column_list"] = setting_dict["key_column_list"]
st.session_state["sensor_column_list"] = setting_dict["sensor_column_list"]

st.set_page_config(layout="wide", page_title="New EHM Scoring")

st.markdown("## Main page")
st.sidebar.markdown("# Main page")

st.write("#### Upload Reference Dataset")
uploaded_files = st.file_uploader(
    "Upload reference files", type=["xlsx", "xls", "csv", "parquet", "feather"]
)

if st.session_state["reference_uploaded"] == False:
    if uploaded_files:
        st.session_state["reference_uploaded"] = True
        reference_df = slp.read_time_series_df(
            upload_files=uploaded_files, time_col="time"
        )
        st.session_state["reference_uploader"] = reference_df

if st.session_state["reference_uploader"] is not None:
    # try group by wafer
    group_by_cols = st.multiselect(
        "Select key columns to groupby wafer",
        st.session_state["key_column_list"],
        st.session_state["key_column_list"],
    )
    groupby_button = st.button("Go on groupby wafer", key="groupby_button")
    hist_col, describe_col = st.columns(2)
    if groupby_button:
        (
            sorted_wafer_list,
            sorted_info_list,
            sorted_first_time_list,
        ) = slp.get_wafer_list_sorted_by_time(
            st.session_state["reference_uploader"],
            key_list=group_by_cols,
            time_col="time",
        )

        st.session_state["reference_dataset"] = sorted_wafer_list

if st.session_state["reference_dataset"]:
    with hist_col:
        st.write("### Data Unit Counts Histogram")
        fig = slp.get_timelength_distribution_fig(st.session_state["reference_dataset"])
        st.plotly_chart(fig)
    with describe_col:
        st.write("### Basic Information")
        info_df = slp.get_info_df(
            df=st.session_state["reference_uploader"],
            data_unit_list=st.session_state["reference_dataset"],
            meta_column_list=st.session_state["meta_column_list"],
        )
        st.table(info_df)
