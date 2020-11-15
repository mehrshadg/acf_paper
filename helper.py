from neuro_helper.entity import TemplateName
import hcp_acf_window as acw
import hcp_acf_zero as acz

template_meta_combination = [
    (TemplateName.SCHAEFER_200_7, "pc"),
    (TemplateName.COLE_360, "pce"),
    (TemplateName.COLE_360, "pcr")
]

net_meta_C = {"pc": "C", "pce": "EC", "pcr": "RC"}
lib_details = [
    (acw, "acw", "ACW-50"),
    (acz, "acz", "ACW-0")
]