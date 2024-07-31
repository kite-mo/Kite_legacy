# EHM_streamlit
 - 버전업된 EHM scoring 을 탑재한 Demo !

## Data preparation
 - 데이터 형식
   - 반복 공정으로 정의될 수 있는 시계열 데이터
   - 시간을 나타내는 열 이름은 반드시 "time" 으로 설정
 - 데이터 로드
   - 본인 로컬 컴퓨터에 데이터 준비
   - reference_data
     - upload_reference_dataset 페이지에 사용될 데이터
   - inference_data
     - make_scoring 페이지에 사용될 데이터

## How to initial setting

 - setting_columns.yml : EHM logic 에 사용될 컬럼명 정리 파일
    - key_column_list : groupby 에 쓰일 key list
    - meta_column_list : 메타 정보를 담고 있는 column list
    - sensor_column_list : EHM 에 사용될 column list 
  
## Command

```
# Python 3.9.18
streamlit run EHM_Demo_version_up.py
```

## Page Example 

page1 : upload_reference_dataset
<img width="1551" alt="스크린샷 2023-12-15 오후 1 21 11" src="https://github.com/teamRTM/EHM_streamlit/assets/59912557/6fd83634-a3b1-4297-aa0a-040e97a8d729">

page2 : make_reference
<img width="1544" alt="스크린샷 2023-12-15 오후 1 22 21" src="https://github.com/teamRTM/EHM_streamlit/assets/59912557/95dd7b08-71a7-4c15-8e10-a7104cffb0ce">

page3 : make_scoring
<img width="1556" alt="스크린샷 2023-12-15 오후 1 22 47" src="https://github.com/teamRTM/EHM_streamlit/assets/59912557/964e35d6-d66b-40f1-97fa-5ca6ecd592be">

page4 : analysis
<img width="1563" alt="스크린샷 2023-12-15 오후 1 23 04" src="https://github.com/teamRTM/EHM_streamlit/assets/59912557/36e110f1-e2f5-48c5-a716-26f3c098c8f0">

