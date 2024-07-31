import numpy as np


valid_dist_method_list = ['dtw', 'euclidean']
scoring_mode_list = ['d_max']
representative_mode_list = ['max', 'min', 'median', 'std']

dist_metric = 'dtw'

# Variables related to scoring
precision = np.finfo(np.float32).eps

# DTW constraints
dtw_metric_params = {
    "global_constraint": "sakoe_chiba", 
    "sakoe_chiba_radius": 3
}

rules = [
    {
        "sensor_list": [
            ### Chuck Heater Power ###
            "Chuck1_Heater_Pwr",
            "Chuck2_Heater_Pwr",
            "Chuck3_Heater_Pwr",
            "Chuck4_Heater_Pwr",
            ### Elbow Heater Power ###
            "Elbow1_Heater_Pwr",
            "Elbow2_Heater_Pwr",
            "Elbow3_Heater_Pwr",
            "Elbow4_Heater_Pwr",
            ### Gas Valve Setting value ###
            "Gas1_Vv",
            "Gas2_Vv",
            "Gas3_Vv",
            "Gas4_Vv",
            "Gas6_Vv",
            ### O2 sensors ###
            "O2_Sample_Vv",
        ],
        "how": 
        {
            "constant": {"by": "given", "value": 1},
        },
    },
    {
        "sensor_list": ["Prcs_APC_SetPoint_Real"],
        "how": {
            "constant": {"by": "given", "value": 750},
        },
    },
    {
        "sensor_list": [
            "Chuck1_LT_Temp_Set",
            "Chuck2_LT_Temp_Set",
            "Chuck3_LT_Temp_Set",
            "Chuck4_LT_Temp_Set",
        ],
        "how": {
            "constant": {"by": "given", "value": 420},
        },
    },
    {
        "sensor_list": [
            "Elbow1_LT_Temp_Set",
            "Elbow2_LT_Temp_Set",
            "Elbow3_LT_Temp_Set",
            "Elbow4_LT_Temp_Set",
        ],
        "how": {
                "constant": {"by": "given", "value": 260},
            },
    },
    {
        "sensor_list": [
            "Elbow1_LT_Temp_Mnt",
            "Elbow2_LT_Temp_Mnt",
            "Elbow3_LT_Temp_Mnt",
            "Elbow4_LT_Temp_Mnt",
        ],
        "how": {
            # AVG 기준 +-1 안에 들어오면 백점, 벗어나면 깎이는 형태 & 260 초과인 경우 0점
            "if_over_0": {
                "by": "given",
                "value": 260,
                "equal": False,
            },
            "if_in_100": {
                "by": "avg",
                "plus_value": 1,
                "minus_value": 1,
                "equal": {"upper": True, "lower": True},
            },
        },
    },
    {
        "sensor_list": [
            ### Adj_Chuck ###
            "Adj_Chuck1_Temp_Mnt",
            "Adj_Chuck2_Temp_Mnt",
            "Adj_Chuck3_Temp_Mnt",
            "Adj_Chuck4_Temp_Mnt",
            "Adj_Chuck5_Temp_Mnt",
            ### Chuck Temp Mnt ###
            "Chuck1_LT_Temp_Mnt",
            "Chuck2_LT_Temp_Mnt",
            "Chuck3_LT_Temp_Mnt",
            "Chuck4_LT_Temp_Mnt",
            ### Elbow Temp Mnt ###
            "Elbow1_Temp_Mnt",
            "Elbow2_Temp_Mnt",
            "Elbow3_Temp_Mnt",
            "Elbow4_Temp_Mnt",
            ### Elbow Temp Set ###
            "Elbow1_Temp_Set",
            "Elbow2_Temp_Set",
            "Elbow3_Temp_Set",
            "Elbow4_Temp_Set",
            ### MFC Set ###
            "MFC1_Set",
            "MFC2_Set",
            "MFC3_Set",
            "MFC4_Set",
            "MFC6_Set",
            "MFC7_Set",
        ],
        "how": {
            "if_in_100": {
                "by": "avg",
                "plus_value": 1,
                "minus_value": 1,
                "equal": {"upper": True, "lower": True},
            },
        },
    },
    {
        "sensor_list": ["Fac_PCW_Outlet_Temp"],
        "how": {
            # upper_value ~ lower_value 내 100점, 벗어날 경우 감점
            "if_in_100": {
                "by": "given",
                "upper_value": 30,
                "lower_value": 15,
                "equal": {"upper": True, "lower": True},
            }
        },
    },
    {
        "sensor_list": ["Fac_PCW_Inlet_Temp"],
        "how": {
            "if_in_100": {
                "by": "given",
                "upper_value": 30,
                "lower_value": 20,
                "equal": {"upper": True, "lower": True},
            }
        },
    },
    {
        "sensor_list": ["Prcs_APC_Press_Read_Real"],
        "how": {
            "if_in_100": {
                "by": "given",
                "upper_value": 850,
                "lower_value": 650,
                "equal": {"upper": True, "lower": True},
            }
        },
    },
    {
        "sensor_list": ["Gas_Box_Exhst_Press", "Cabinet_Exhst_Press"],
        "how": {
            "if_in_100": {
                "by": "given",
                "upper_value": 1,
                "lower_value": 0.2,
                "equal": {"upper": True, "lower": True},
            }
        },
    },
    {
        "sensor_list": ["Process_Exhst_Press"],
        "how": {
            "if_in_100": {
                "by": "given",
                "upper_value": 4,
                "lower_value": 0.45,
                "equal": {"upper": True, "lower": True},
            }
        },
    },
    {
        "sensor_list": ["Fac_PCW_Flow_Rate"],
        "how": {
            # 1초과 100점, 1 이하 0점
            "if_over_100": {"by": "given", "value": 1, "equal": False},  # >
            "if_under_0": {"by": "given", "value": 1, "equal": True},  # <=
        },
    },
    {
        "sensor_list": ["O2LevelRead"],
        "how": {
            # 20 이하 100점, 20~50:점차 감점, 50초과 0점
            "if_over_0": {"by": "given", "value": 50, "equal": False},  # >
            "if_under_100": {"by": "given", "value": 20, "equal": True},  # (<=)
        },
    },
    ########## TODO 제시해야할것 ##########
    # {
    #     "sensor_list": [
    #         #MFC는 Recipe에 따라 다른 값을 가짐
    #         "MFC1_Monitor",
    #         "MFC2_Monitor",
    #         "MFC3_Monitor",
    #         "MFC4_Monitor",
    #         "MFC6_Monitor",
    #         "MFC7_Monitor",
    #     ],
    #     "how": [{}],
    # },
    ########## TBD 아직정보안줌 ###########
    # {
    #     "sensor_list": [
    #         # chuck
    #         "Chuck1_Temp_MV_Mnt",
    #         "Chuck2_Temp_MV_Mnt",
    #         "Chuck3_Temp_MV_Mnt",
    #         "Chuck4_Temp_MV_Mnt",
    #         # elbow
    #         "Elbow1_Temp_MV_Mnt",
    #         "Elbow2_Temp_MV_Mnt",
    #         "Elbow3_Temp_MV_Mnt",
    #         "Elbow4_Temp_MV_Mnt",
    #     ],
    #     "how": [{}],
    # },
]