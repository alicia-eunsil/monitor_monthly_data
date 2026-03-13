from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict

KOSIS_BASE_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    title: str
    org_id: str
    tbl_id: str
    itm_id: str
    obj_l1: str = "ALL"
    obj_l2: str = ""
    obj_l3: str = ""
    obj_l4: str = ""
    obj_l5: str = ""
    obj_l6: str = ""
    obj_l7: str = ""
    obj_l8: str = ""
    prd_se: str = "M"
    start_prd_de: str = "201301"
    has_category: bool = False
    category_label: str = ""
    output_fields: str = ""

    def to_params(self, api_key: str, end_prd_de: str) -> Dict[str, str]:
        params: Dict[str, str] = {
            "method": "getList",
            "apiKey": api_key,
            "itmId": self.itm_id,
            "objL1": self.obj_l1,
            "objL2": self.obj_l2,
            "objL3": self.obj_l3,
            "objL4": self.obj_l4,
            "objL5": self.obj_l5,
            "objL6": self.obj_l6,
            "objL7": self.obj_l7,
            "objL8": self.obj_l8,
            "format": "json",
            "jsonVD": "Y",
            "prdSe": self.prd_se,
            "startPrdDe": self.start_prd_de,
            "endPrdDe": end_prd_de,
            "orgId": self.org_id,
            "tblId": self.tbl_id,
        }
        if self.output_fields:
            params["outputFields"] = self.output_fields
        return params


def default_end_period() -> str:
    return date.today().strftime("%Y%m")


def default_end_period_by_prd_se(prd_se: str) -> str:
    if str(prd_se).upper() == "H":
        # Regional employment survey fixed latest half-year point.
        return "202502"
    return default_end_period()


DATASETS_MONTHLY = [
    DatasetConfig(
        key="activity",
        title="경제활동인구현황",
        org_id="101",
        tbl_id="DT_1DA7004S",
        itm_id="T10+T20+T30+T40+T50+T60+T80+T90+T100+",
        start_prd_de="199906",
        output_fields="TBL_ID+TBL_NM+ITM_ID+ITM_NM+PRD_SE+PRD_DE+DT+UNIT_NM_ENG+C1+C1_OBJ_NM+",
        has_category=False,
    ),
    DatasetConfig(
        key="age",
        title="연령별 취업자",
        org_id="101",
        tbl_id="DT_1DA7031S",
        itm_id="T30+",
        obj_l1="ALL",
        obj_l2="00+75+10+20+30+40+50+60+63+70+",
        start_prd_de="199801",
        has_category=True,
        category_label="연령(구분)",
    ),
    DatasetConfig(
        key="status",
        title="종사상지위별 취업자",
        org_id="101",
        tbl_id="DT_1DA7035S",
        itm_id="T30+",
        obj_l1="ALL",
        obj_l2="ALL",
        start_prd_de="199801",
        has_category=True,
        category_label="종사상지위(구분)",
    ),
    DatasetConfig(
        key="industry",
        title="산업별 취업자수",
        org_id="101",
        tbl_id="DT_1DA7E33S_NEW",
        itm_id="T30+",
        obj_l1="ALL",
        obj_l2="ALL",
        start_prd_de="201301",
        has_category=True,
        category_label="산업(대분류)",
    ),
    DatasetConfig(
        key="occupation",
        title="직종별 취업자수",
        org_id="101",
        tbl_id="DT_1DA7E34S_NEW",
        itm_id="T30+",
        obj_l1="ALL",
        obj_l2="ALL",
        start_prd_de="201301",
        has_category=True,
        category_label="직종(대분류)",
    ),
]


DATASETS_GYEONGGI_HALFYEAR = [
    DatasetConfig(
        key="activity",
        title="경제활동인구현황",
        org_id="101",
        tbl_id="DT_1ES3A01S",
        itm_id="T1+T2+T3+T4+T9+T5+T6+T7+T11+T8+T10+",
        prd_se="H",
        start_prd_de="201301",
        has_category=False,
    ),
    DatasetConfig(
        key="age",
        title="연령별 취업자",
        org_id="101",
        tbl_id="DT_1ES3A03_A01S",
        itm_id="T00+T11+T12+T21+",
        obj_l1="ALL",
        obj_l2="ALL",
        prd_se="H",
        start_prd_de="201301",
        has_category=True,
        category_label="연령(구분)",
    ),
    DatasetConfig(
        key="status",
        title="종사상지위별 취업자",
        org_id="101",
        tbl_id="DT_1ES3A07S",
        itm_id="T3+",
        obj_l1="ALL",
        obj_l2="ALL",
        prd_se="H",
        start_prd_de="201301",
        has_category=True,
        category_label="종사상지위(구분)",
    ),
    DatasetConfig(
        key="industry",
        title="산업별 취업자수",
        org_id="101",
        tbl_id="DT_1ES3A30S",
        itm_id="T1+T2+",
        obj_l1="ALL",
        obj_l2="ALL",
        prd_se="H",
        start_prd_de="201301",
        has_category=True,
        category_label="산업(대분류)",
    ),
    DatasetConfig(
        key="occupation",
        title="직종별 취업자수",
        org_id="101",
        tbl_id="DT_1ES3A31S",
        itm_id="T1+T2+",
        obj_l1="ALL",
        obj_l2="ALL",
        prd_se="H",
        start_prd_de="201301",
        has_category=True,
        category_label="직종(대분류)",
    ),
]


def datasets_for_scope(region_scope: str) -> list[DatasetConfig]:
    if region_scope == "gyeonggi31":
        return DATASETS_GYEONGGI_HALFYEAR
    return DATASETS_MONTHLY


TARGET_REGIONS = [
    "전국",
    "서울특별시",
    "부산광역시",
    "대구광역시",
    "인천광역시",
    "광주광역시",
    "대전광역시",
    "울산광역시",
    "세종특별자치시",
    "경기도",
    "강원특별자치도",
    "충청북도",
    "충청남도",
    "전라북도",
    "전라남도",
    "경상북도",
    "경상남도",
    "제주특별자치도",
]


GYEONGGI_SIGUNGU = [
    "가평군",
    "고양시",
    "과천시",
    "광명시",
    "광주시",
    "구리시",
    "군포시",
    "김포시",
    "남양주시",
    "동두천시",
    "부천시",
    "성남시",
    "수원시",
    "시흥시",
    "안산시",
    "안성시",
    "안양시",
    "양주시",
    "양평군",
    "여주시",
    "연천군",
    "오산시",
    "용인시",
    "의왕시",
    "의정부시",
    "이천시",
    "파주시",
    "평택시",
    "포천시",
    "하남시",
    "화성시",
]


GYEONGGI_DISTRICT_TO_CITY = {
    "성남시수정구": "성남시",
    "성남시중원구": "성남시",
    "성남시분당구": "성남시",
    "고양시덕양구": "고양시",
    "고양시일산동구": "고양시",
    "고양시일산서구": "고양시",
    "고양시일산구": "고양시",
    "수원시장안구": "수원시",
    "수원시권선구": "수원시",
    "수원시팔달구": "수원시",
    "수원시영통구": "수원시",
    "안양시만안구": "안양시",
    "안양시동안구": "안양시",
    "안산시상록구": "안산시",
    "안산시단원구": "안산시",
    "용인시처인구": "용인시",
    "용인시기흥구": "용인시",
    "용인시수지구": "용인시",
    "부천시원미구": "부천시",
    "부천시소사구": "부천시",
    "부천시오정구": "부천시",
}
