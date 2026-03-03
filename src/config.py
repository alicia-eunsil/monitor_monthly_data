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


DATASETS = [
    DatasetConfig(
        key="activity",
        title="경제활동인구현황",
        org_id="101",
        tbl_id="DT_1DA7004S",
        itm_id="T10+T20+T30+T40+T50+T60+T80+T90+T100+",
        start_prd_de="199906",
        output_fields="TBL_ID+TBL_NM+OBJ_ID+OBJ_NM+ITM_ID+ITM_NM+UNIT_NM+PRD_SE+PRD_DE+",
        has_category=False,
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

